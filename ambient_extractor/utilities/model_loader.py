import torch
from transformers import AutoTokenizer
from model.model import TextClassifier

def save_onnx(model, model_path):
    model.to(torch.device("cpu"))
    SAMPLE_INPUT = "ROOM"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoding = tokenizer(
        SAMPLE_INPUT,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=32
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    model.save_onnx(model_path, input_ids, attention_mask)


def load_best_model(device, path="./checkpoints/best_model.pt"):
    model = TextClassifier()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.to(device)
    model.eval()
    
    return model