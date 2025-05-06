import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def test(model, test_loader, criterion, device):
    all_preds = []
    all_labels = []
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Batch:")
        for input_ids, attention_mask, labels in loop:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss=criterion(outputs, labels)
            total_loss+=loss.item()

            preds = (outputs > 0.5).int()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    
    acc = accuracy_score(all_labels, all_preds)
    f1sc = f1_score(all_labels, all_preds)

    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {acc}")
    print(f"Test F1-Score: {f1sc}")