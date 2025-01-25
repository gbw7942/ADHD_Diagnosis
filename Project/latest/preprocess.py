import os
import numpy as np
import nibabel as nib
from nilearn import image, masking, signal
from scipy import stats, signal as scipy_signal
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

class ADHD200Preprocessor:
    def __init__(self, target_shape=(28, 28, 28), time_points=30):
        self.target_shape = target_shape
        self.time_points = time_points
        self.scaler = StandardScaler()
        
    def preprocess_single_subject(self, func_file, anat_file=None, confounds=None):
        """
        完整的单个受试者预处理流程
        """
        # 1. 基本数据加载
        func_img = nib.load(func_file)
        data = func_img.get_fdata()
        print(f"\nInitial data shape: {data.shape}")
        
        # 2. 时间层面预处理
        # 2.1 去除前几个时间点（稳态）
        n_volumes_to_remove = 5
        if data.shape[-1] > n_volumes_to_remove:
            data = data[..., n_volumes_to_remove:]
            print(f"After removing first {n_volumes_to_remove} volumes: {data.shape}")
            
        # 2.2 时间层面的切片定时校正
        data = self._slice_timing_correction(data)
        print(f"After slice timing correction: {data.shape}")
        
        # 2.3 头动校正（如果提供了confounds）
        if confounds is not None:
            data = self._motion_correction(data, confounds)
            print(f"After motion correction: {data.shape}")
        
        # 3. 空间预处理
        # 3.1 空间平滑
        smooth_fwhm = 6.0  # 6mm FWHM
        img_smooth = image.smooth_img(
            nib.Nifti1Image(data, func_img.affine),
            smooth_fwhm
        )
        data = img_smooth.get_fdata()
        print(f"After spatial smoothing: {data.shape}")
        
        # 3.2 空间标准化（重采样到目标大小）
        data = self._spatial_normalization(data, func_img.affine)
        print(f"After spatial normalization: {data.shape}")
        
        # 4. 信号处理
        # 4.1 带通滤波
        data = self._bandpass_filter(data)
        print(f"After bandpass filtering: {data.shape}")
        
        # 4.2 去线性趋势
        data = self._detrend(data)
        print(f"After detrending: {data.shape}")
        
        # 4.3 全局信号回归
        data = self._global_signal_regression(data)
        print(f"After global signal regression: {data.shape}")
        
        # 5. 时间点选择
        data = self._select_timepoints(data)
        print(f"After timepoint selection: {data.shape}")
        
        # 6. 数据标准化
        data = self._normalize_data(data)
        print(f"Final data shape: {data.shape}")
        
        return data
    
    def _slice_timing_correction(self, data):
        """
        切片定时校正
        简化版本：使用插值方法进行时间校正
        """
        n_slices = data.shape[2]
        tr = 2.0  # TR时间，单位秒
        
        # 创建参考时间点（假设中间切片为参考）
        slice_times = np.linspace(0, tr, n_slices)
        reference_slice = n_slices // 2
        
        # 对每个体素进行时间插值
        corrected_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    time_course = data[i, j, k, :]
                    # 使用三次样条插值
                    if not np.all(np.isnan(time_course)):
                        time_points = np.arange(len(time_course)) * tr + slice_times[k]
                        reference_times = np.arange(len(time_course)) * tr + slice_times[reference_slice]
                        corrected_data[i, j, k, :] = np.interp(reference_times, time_points, time_course)
        
        return corrected_data
    
    def _motion_correction(self, data, confounds):
        """
        头动校正
        """
        # 从confounds中提取头动参数
        motion_params = confounds[['trans_x', 'trans_y', 'trans_z', 
                                 'rot_x', 'rot_y', 'rot_z']].values
        
        # 回归掉头动影响
        data_shape = data.shape
        data_2d = data.reshape(-1, data_shape[-1])
        
        # 使用线性回归去除头动影响
        for i in range(data_2d.shape[0]):
            if not np.all(np.isnan(data_2d[i])):
                beta = np.linalg.lstsq(motion_params, data_2d[i], rcond=None)[0]
                data_2d[i] = data_2d[i] - motion_params.dot(beta)
            
        return data_2d.reshape(data_shape)
    
    def _spatial_normalization(self, data, affine):
        """
        空间标准化
        """
        # 创建目标图像
        target_shape = self.target_shape + (data.shape[-1],)
        
        # 重采样到目标大小
        img = nib.Nifti1Image(data, affine)
        img_resized = image.resample_img(
            img,
            target_affine=np.eye(4),
            target_shape=target_shape,
            interpolation='continuous'
        )
        
        return img_resized.get_fdata()
    
    def _bandpass_filter(self, data, tr=2.0):
        """
        带通滤波 (0.01-0.1 Hz)
        """
        # 设置频率范围
        low_freq = 0.01
        high_freq = 0.1
        
        # 应用滤波器
        data_shape = data.shape
        data_2d = data.reshape(-1, data_shape[-1])
        
        # 设计带通滤波器
        nyquist = 1.0 / (2 * tr)
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = scipy_signal.butter(3, [low, high], btype='band')
        
        # 应用滤波器
        for i in range(data_2d.shape[0]):
            if not np.all(np.isnan(data_2d[i])):
                data_2d[i] = scipy_signal.filtfilt(b, a, data_2d[i])
            
        return data_2d.reshape(data_shape)
    
    def _detrend(self, data):
        """
        去除线性趋势
        """
        data_shape = data.shape
        data_2d = data.reshape(-1, data_shape[-1])
        
        # 应用线性去趋势
        for i in range(data_2d.shape[0]):
            if not np.all(np.isnan(data_2d[i])):
                data_2d[i] = scipy_signal.detrend(data_2d[i])
            
        return data_2d.reshape(data_shape)
    
    def _global_signal_regression(self, data):
        """
        全局信号回归
        """
        # 计算全局信号
        global_signal = np.nanmean(data, axis=(0,1,2))
        
        # 回归掉全局信号
        data_shape = data.shape
        data_2d = data.reshape(-1, data_shape[-1])
        
        for i in range(data_2d.shape[0]):
            if not np.all(np.isnan(data_2d[i])):
                beta = np.linalg.lstsq(
                    global_signal.reshape(-1,1),
                    data_2d[i].reshape(-1,1),
                    rcond=None
                )[0]
                data_2d[i] = data_2d[i] - global_signal * beta
            
        return data_2d.reshape(data_shape)
    
    def _select_timepoints(self, data):
        """
        选择时间点
        """
        if data.shape[-1] > self.time_points:
            # 均匀采样时间点
            indices = np.linspace(0, data.shape[-1]-1, self.time_points, dtype=int)
            data = data[..., indices]
        return data
    
    def _normalize_data(self, data):
        """
        数据标准化
        """
        # 应用StandardScaler
        data_shape = data.shape
        data_2d = data.reshape(-1, data_shape[-1])
        
        # 处理非NaN值
        mask = ~np.any(np.isnan(data_2d), axis=1)
        data_2d[mask] = self.scaler.fit_transform(data_2d[mask])
        
        return data_2d.reshape(data_shape)
    
    def quality_check(self, data):
        """
        质量检查
        """
        # 计算基本统计量（忽略NaN值）
        mean_signal = np.nanmean(data, axis=-1)
        std_signal = np.nanstd(data, axis=-1)
        snr = mean_signal / (std_signal + 1e-6)
        
        # 创建质量报告
        report = {
            'mean_intensity': np.nanmean(mean_signal),
            'std_intensity': np.nanmean(std_signal),
            'snr': np.nanmean(snr),
            'max_value': np.nanmax(data),
            'min_value': np.nanmin(data),
            'nan_percentage': (np.isnan(data).sum() / data.size) * 100
        }
        
        return report

def process_adhd200_dataset(data_dir, output_dir):
    """
    处理整个ADHD200数据集
    """
    preprocessor = ADHD200Preprocessor()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有受试者数据
    subjects = [f for f in os.listdir(data_dir) if f.endswith('bold.nii.gz')]
    print(f"\nFound {len(subjects)} subjects to process")
    
    # 使用tqdm创建进度条
    for subject in tqdm(subjects, desc="Processing subjects"):
        try:
            print(f"\nProcessing subject: {subject}")
            
            # 构建文件路径
            func_file = os.path.join(data_dir, subject)
            
            # 预处理
            processed_data = preprocessor.preprocess_single_subject(func_file)
            
            # 质量检查
            qc_report = preprocessor.quality_check(processed_data)
            print("\nQuality check results:")
            for metric, value in qc_report.items():
                print(f"{metric}: {value:.4f}")
            
            # 保存处理后的数据
            output_file = os.path.join(output_dir, f'processed_{subject}')
            np.save(output_file, processed_data)
            
            # 保存质量报告
            qc_file = os.path.join(output_dir, f'qc_{subject.replace(".nii.gz", ".json")}')
            pd.Series(qc_report).to_json(qc_file)
            
        except Exception as e:
            print(f"\nError processing {subject}: {str(e)}")
            continue

# 使用示例
if __name__ == "__main__":
    data_dir = "/root/autodl-tmp/CNNLSTM/Project/Data"
    output_dir = "/root/autodl-tmp/CNNLSTM/Project/preprocssed30_28"
    
    print("Starting ADHD200 dataset processing...")
    process_adhd200_dataset(data_dir, output_dir)
    print("\nProcessing completed!")