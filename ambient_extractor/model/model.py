from transformers import AutoModel
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, embed_dim=768):
        super(TextClassifier, self).__init__()
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]  

        return self.classifier(cls_embedding).squeeze(-1)