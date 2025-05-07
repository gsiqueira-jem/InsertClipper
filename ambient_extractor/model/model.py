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
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
        return self.classifier(cls_embedding).squeeze(-1)
    
    def save_onnx(self, model_path, input_ids, attention_mask):
        torch.onnx.export(
            self,
            (input_ids, attention_mask),  # tuple of inputs
            model_path,       # output file
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size"}
            },
            opset_version=13,
            do_constant_folding=True
        )