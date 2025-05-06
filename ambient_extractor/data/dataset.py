import torch
from torch.utils.data import Dataset

# Sample binary classification dataset
class AmbientDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len,
            return_tensors=None
        )
        
        input_ids = torch.tensor(encodings['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encodings['attention_mask'], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return input_ids, attention_mask, label
