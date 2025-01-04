from torch.utils.data import Dataset
import numpy as np
import os
import nibabel as nib
import torch
from scipy.ndimage import zoom
import config

class FMRIDataGenerator(Dataset):  
    def __init__(self, list_IDs, labels, dataset_dir, time_length=config.TIME_LENGTH, img_dim=(49, 58, 47), target_dim=(28, 28, 28)):  
        """  
        :param list_IDs: List of file IDs (paths to fMRI data)  
        :param labels: Dictionary mapping file IDs to labels  
        :param dataset_dir: Directory where the dataset is stored  
        :param time_length: Time length of the fMRI data  
        :param img_dim: Original dimensions of the fMRI image (x, y, z)  
        :param target_dim: Resized target dimensions (x, y, z)  
        """  
        self.list_IDs = list_IDs  
        self.labels = labels  
        self.dataset_dir = dataset_dir  
        self.time_length = time_length  
        self.img_dim = img_dim  
        self.target_dim = target_dim  

    def __len__(self):  
        return len(self.list_IDs)  

    def __getitem__(self, index):  
        img_path = self.list_IDs[index]  
        X = self.preprocess_image(img_path)  
        y = self.labels[img_path]  

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)  

    def preprocess_image(self, img_path):  
        """Preprocess the fMRI image"""  
        img = nib.load(os.path.join(self.dataset_dir, img_path))  

        # Handle the case where time length does not match  
        if img.shape[3] > self.time_length:  
            pp_img = self.truncate_image(img)  
        elif img.shape[3] < self.time_length:  
            pp_img = self.pad_image(img)  
        else:  
            pp_img = img.get_fdata()  

        # Dynamically get the actual spatial dimensions  
        actual_dim = pp_img.shape[:3]  
        new_x = self.target_dim[0] / actual_dim[0]  
        new_y = self.target_dim[1] / actual_dim[1]  
        new_z = self.target_dim[2] / actual_dim[2]  

        resized_imgs = []  
        for t in range(self.time_length):  
            # Rescale the image  
            z_img = zoom(pp_img[:, :, :, t], (new_x, new_y, new_z), order=1)  

            # Ensure the resized image matches the target size  
            if z_img.shape != self.target_dim:  
                z_img = self.fix_shape(z_img, self.target_dim)  

            resized_imgs.append(z_img.reshape((*self.target_dim, 1)))  

        return np.array(resized_imgs)  

    def fix_shape(self, img, target_dim):  
        """Fix the image shape to ensure it matches the target shape"""  
        # If the image is too large, crop it to the target size  
        if img.shape > target_dim:  
            slices = tuple(slice(0, s) for s in target_dim)  
            img = img[slices]  
        # If the image is too small, pad it to the target size  
        elif img.shape < target_dim:  
            pad_width = [(0, max(0, t - s)) for s, t in zip(img.shape, target_dim)]  
            img = np.pad(img, pad_width, mode='constant', constant_values=0)  
        return img  

    def truncate_image(self, img):  
        """Truncate the image to the desired time length"""  
        return img.get_fdata()[:, :, :, :self.time_length]  

    def pad_image(self, img):  
        """Pad the image to the desired time length"""  
        # Get the original image data  
        padded_img = img.get_fdata()  
        
        # Get the spatial dimensions of the original image  
        actual_dim = padded_img.shape[:3]  # (x, y, z)  
        
        # Create a padding array, ensuring its spatial dimensions match the original image  
        img_padding = np.zeros((*actual_dim, 1))  # Dynamically generate a padding array of shape (x, y, z, 1)  
        
        # Calculate how many time steps need to be filled  
        amt_to_fill = self.time_length - padded_img.shape[3]  
        
        # Append padding along the time axis (axis=3)  
        for _ in range(amt_to_fill):  
            padded_img = np.append(arr=padded_img, values=img_padding, axis=3)  

        return padded_img
