import torch
from tqdm import tqdm
from model.model import TextClassifier
from sklearn.metrics import accuracy_score, f1_score

def train(model, train_loader, criterion, optimizer, device, num_epochs=10):
    best_f1=0.0
    loop = tqdm(range(num_epochs), desc="Epochs")
    for epoch in loop:
        
        all_preds = []
        all_labels = []
        total_loss = 0
        total_loss = 0
        
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = (outputs > 0.5).int()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1sc = f1_score(all_labels, all_preds)

        tqdm.write(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1sc:.4f}")
        
        if f1sc > best_f1:
            best_f1 = f1sc
            torch.save(model.state_dict(), "./checkpoints/best_model.pt")
            tqdm.write(f"New best model saved at epoch {epoch} (F1: {f1sc:.4f})")
            
        if (epoch  + 1) % 10 == 0:
            fname = f"./checkpoints/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, fname)
            tqdm.write(f"Saved model after {epoch+1} epochs → {fname}")