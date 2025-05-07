import torch
import torch.nn as nn
import torch.optim as optim
from utilities.train import train
from utilities.test import test
from utilities.model_loader import load_best_model, save_onnx
from model.model import TextClassifier
from data.dataloader import load_data

def main():
    DATA_PATH = "./dataset/cad_text_classification_dataset.csv"
    MODEL_PATH = "./onnx_checkpoint/ambient_classifier.onnx"
    NUM_EPOCHS=50
    BATCH_SIZE=50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TextClassifier()
    model.to(device)
    train_loader, test_loader = load_data(data_path=DATA_PATH, batch_size=BATCH_SIZE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    print("Start Training")
    train(model, train_loader, criterion, optimizer, device=device, num_epochs=NUM_EPOCHS)
    print("Getting best model, starting testing")
    model = load_best_model(device)
    test(model, test_loader, criterion, device=device)

    print(f"Saving ONNX model to {MODEL_PATH}")
    save_onnx(model, MODEL_PATH)
    
if __name__ == "__main__":
    main()