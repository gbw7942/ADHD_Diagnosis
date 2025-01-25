from math import sin
import torch  
import numpy as np  
import os  
from tqdm import tqdm  
from model import ImprovedCNNTransformer  
import glob  
from sklearn.metrics import classification_report, confusion_matrix  
import seaborn as sns  
import matplotlib.pyplot as plt  
import pandas as pd  

class Inferencer:  
    def __init__(self, model_path, device=None):  
        """  
        初始化推理器  
        Args:  
            model_path: 模型权重文件路径  
            device: 运行设备，默认None会自动选择  
        """  
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.model = self.load_model(model_path)  
        self.model.eval()  
        # print(f"Model loaded and running on {self.device}")  

    def load_model(self, model_path):  
        """  
        加载预训练模型  
        """  
        if not os.path.exists(model_path):  
            raise FileNotFoundError(f"Model file not found: {model_path}")  
        
        # 加载检查点  
        checkpoint = torch.load(model_path, map_location=self.device)  
        
        # 创建模型实例  
        model = ImprovedCNNTransformer()  
        
        # 加载模型权重  
        model.load_state_dict(checkpoint['model_state_dict'])  
        model = model.to(self.device)  
        
        return model  

    def preprocess_data(self, data):  
        """  
        预处理输入数据  
        Args:  
            data: numpy数组，形状为[time_steps, height, width, depth, channels]  
        Returns:  
            处理后的tensor  
        """  
        if not isinstance(data, torch.Tensor):  
            data = torch.tensor(data, dtype=torch.float32)  
        
        # 移动channels维度并添加batch维度  
        data = data.permute(0, 4, 1, 2, 3)  # [time_steps, channels, height, width, depth]  
        data = data.unsqueeze(0)  # [1, time_steps, channels, height, width, depth]  
        
        return data  

    def predict_single(self, data):  
        """  
        对单个样本进行预测  
        Args:  
            data: numpy数组或tensor  
        Returns:  
            预测类别和概率  
        """  
        with torch.no_grad():  
            # 预处理数据  
            data = self.preprocess_data(data)  
            data = data.to(self.device)  
            
            # 前向传播  
            outputs = self.model(data)  
            probabilities = torch.softmax(outputs, dim=1)  
            
            # 获取预测结果  
            all_probs = probabilities[0].cpu().numpy()
            pred_class = torch.argmax(probabilities, dim=1).item()  
            pred_prob = probabilities[0][pred_class].item()  
            
            return pred_class, pred_prob, all_probs 

    def predict_batch(self, data_list, labels=None):  
        """  
        批量预测  
        Args:  
            data_list: 数据文件路径列表或numpy数组列表  
            labels: 真实标签列表（可选）  
        Returns:  
            预测结果字典  
        """  
        predictions = []  
        probabilities = []  
        
        for data_item in tqdm(data_list, desc="Predicting"):  
            # 如果输入是文件路径，则加载数据  
            if isinstance(data_item, str):  
                data = np.load(data_item)  
            else:  
                data = data_item  
            
            # 预测  
            pred_class, pred_prob, all_prob = self.predict_single(data)
            print(pred_class, all_prob)  
            predictions.append(pred_class)  
            probabilities.append(pred_prob)  
        
        results = {  
            'predictions': predictions,  
            'probabilities': probabilities  
        }  
        
        # 如果提供了标签，计算评估指标  
        if labels is not None:  
            results['classification_report'] = classification_report(  
                labels, predictions, output_dict=True  
            )  
            results['confusion_matrix'] = confusion_matrix(labels, predictions)  
        
        return results  

    def visualize_results(self, results, save_dir=None):  
        """  
        可视化预测结果  
        Args:  
            results: predict_batch返回的结果字典  
            save_dir: 保存可视化结果的目录  
        """  
        if save_dir:  
            os.makedirs(save_dir, exist_ok=True)  
        
        # 绘制混淆矩阵  
        if 'confusion_matrix' in results:  
            plt.figure(figsize=(10, 8))  
            sns.heatmap(  
                results['confusion_matrix'],   
                annot=True,   
                fmt='d',  
                cmap='Blues'  
            )  
            plt.title('Confusion Matrix')  
            if save_dir:  
                plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))  
            plt.close()  
        
        # 绘制预测概率分布  
        plt.figure(figsize=(10, 6))  
        plt.hist(results['probabilities'], bins=20)  
        plt.title('Prediction Probability Distribution')  
        plt.xlabel('Probability')  
        plt.ylabel('Count')  
        if save_dir:  
            plt.savefig(os.path.join(save_dir, 'probability_distribution.png'))  
        plt.close()  
        
        # 保存详细的分类报告  
        if 'classification_report' in results:  
            report_df = pd.DataFrame(results['classification_report']).transpose()  
            if save_dir:  
                report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'))  

def main():  
    # 配置  
    model_path = "/root/autodl-tmp/CNNLSTM/Project/fMRI/ljhtest/experiments/20250111_120230/best_model.pth"  
    data_dir = "/root/autodl-tmp/CNNLSTM/Project/preprocssed_60_64"  
    save_dir = "inference_results"  
    singel_path = "/root/autodl-tmp/CNNLSTM/Project/preprocssed_60_64/sub-0010021_ses-1_task-rest_run-2_bold.nii.gz.npy"
    data_npy = np.load(singel_path)
    # 获取测试数据文件列表  
    test_files = glob.glob(os.path.join(data_dir, "*.npy"))  
    
    # 可选：准备标签（如果有）  
    # test_labels = [...]  # 从某处加载测试标签  
    
    # 初始化推理器  
    inferencer = Inferencer(model_path)  
    # singel_results, _ = inferencer.predict_single(data_npy)
    # 进行预测  
    results = inferencer.predict_batch(test_files)  # 如果有标签：, labels=test_labels  
    
    # 可视化结果  
    inferencer.visualize_results(results, save_dir)  
    # print(singel_results)

if __name__ == "__main__":  
    main()