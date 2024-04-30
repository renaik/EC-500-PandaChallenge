import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


__author__ = "Reana Naik"


def apply_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)
    
    data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])
    return data_transforms



def show_image():
    pass



class ProstateCancerDataset(Dataset):
    """ 
    Load Prostate cANcer graDe Assessment (PANDA) dataset.
    https://www.kaggle.com/c/prostate-cancer-grade-assessment/data
    """
    def __init__(self, img_path, csv_path, transform=None):
        super(ProstateCancerDataset, self).__init__()
        self.csv_path = csv_path
        self.img_path = img_path
        self.transform = transform

        self.ids, self.labels = self.load_dataset()

    def load_dataset(self):
        Image.MAX_IMAGE_PIXELS = None

        ids = []
        imgs = []
        labels = []

        with open(self.csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                id = row['image_id']
                label = int(row['isup_grade'])
                
                ids.append(id)
                labels.append(label)

        return ids, labels
    
    def get_data(self):
        return self.ids, self.labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return id, label
    


class GraphDataset(Dataset):
    """
    Load graph dataset generated from original dataset.
    """
    def __init__(self, graph_path, ids, labels):
        super(GraphDataset, self).__init__()
        self.graph_path = graph_path
        self.ids = ids
        self.labels = labels

        self.features, self.adjs_s = self.load_dataset()

    def load_dataset(self):
        features = []
        adjs_s = []

        for id in self.ids:
            graph_path = os.path.join(self.graph_path, f'{id}')

            feature_path = graph_path + '/features.pt'
            if os.path.exists(feature_path):
                feature = torch.load(feature_path, map_location=lambda storage, loc: storage)
            else:
                print(feature_path + ' does not exist.')
                feature = torch.zeros(1, 512)

            adj_s_path = graph_path + '/adj_s.pt'
            if os.path.exists(adj_s_path):
                adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)
            else:
                print(adj_s_path + ' does not exist.')
                adj_s = torch.ones(feature.shape[0], feature.shape[0])

            features.append(feature)
            adjs_s.append(adj_s)
        
        return features, adjs_s
    
    def get_data(self):
        return self.features, self.labels, self.ids, self.adjs_s

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        id = self.ids[idx]
        adj_s = self.adjs_s[idx]

        return feature, label, id, adj_s