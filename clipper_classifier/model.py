import torch
import torch.nn as nn
import clip
from torch.utils.data import Dataset
from PIL import Image

class CLIPClassifierDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.classes = ["door", "window", "furniture", "unknown"]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=4):
        super().__init__()
        self.clip_model = clip_model
        # Freeze CLIP weights if desired
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(512, num_classes)  # assuming ViT-B/32 512-dim output
        
    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        image_features = image_features.float()
        logits = self.classifier(image_features)
        return logits
