from torch.utils.data import Dataset
import os, torch


class ImageRewardDataset(Dataset):
    """
    Dataset for image preference annotations.
    Returns data as (image1, image2, annotation) tuples.
    image1/2 are tensors of shape (3,H,W), and annotation is a float.
    """

    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data_files = [os.path.join(data_folder,file) for file in os.listdir(data_folder) if file.endswith('.pt')]

        self.length = len(self.data_files)
    

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        return data['data1'], data['data2'], data['annotation']