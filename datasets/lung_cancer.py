import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import params
import torchvision.transforms as transforms

class LungCancerDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None, train_split=0.8):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Map string labels to integers
        self.label_map = {'Benign': 0, 'Malignant': 1, 'Normal': 2}

        # Split data into training and evaluation sets based on the train_split ratio
        split_index = int(len(self.data) * train_split)
        if train:
            self.data = self.data[:split_index]
        else:
            self.data = self.data[split_index:]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label_str = self.data.iloc[idx, 1]

        # Convert label from string to integer
        label = self.label_map[label_str]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale if necessary

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_lung_cancer(csv_file, train=True, batch_size=params.batch_size):
        """Get Lung Cancer dataset loader."""
        transform = transforms.Compose([
            transforms.Resize((params.image_size, params.image_size)), 
            transforms.ToTensor(),
            transforms.Normalize(
                                mean=params.dataset_mean,
                                std=params.dataset_std)
        ])

        dataset = LungCancerDataset(csv_file=csv_file, train=train, transform=transform)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)

        return data_loader
    