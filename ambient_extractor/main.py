import torch
import torch.nn as nn
import torch.optim as optim
from utilities.train import train
from utilities.test import test
from model.model import TextClassifier
from data.dataloader import load_data

def main():
    DATA_PATH = "./dataset/cad_text_classification_dataset.csv"
    NUM_EPOCHS=50
    BATCH_SIZE=50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TextClassifier()
    model.to(device)
    train_loader, test_loader = load_data(data_path=DATA_PATH, batch_size=BATCH_SIZE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    train(model, train_loader, criterion, optimizer, device=device, num_epochs=NUM_EPOCHS)
    test(model, test_loader, criterion, device=device)
    
if __name__ == "__main__":
    main()