from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

def train(model, train_loader, criterion, optimizer, device, num_epochs=10):
    all_preds = []
    all_labels = []
    total_loss = 0
    
    loop = tqdm(range(num_epochs), desc="Epochs:")
    model.train()
    for _ in loop:
        total_loss = 0
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

        loop.set_postfix({"loss": f"{total_loss:.4f}"})
    
    acc = accuracy_score(all_labels, all_preds)
    f1sc = f1_score(all_labels, all_preds)

    print(f"Train Loss: {loss}")
    print(f"Train Accuracy: {acc}")
    print(f"Train F1-Score: {f1sc}")