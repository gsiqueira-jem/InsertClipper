import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import os
from tqdm import tqdm
from torchvision import transforms
from model import CLIPClassifierDataset, CLIPClassifier
from data_utils import load_dataset
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP Fine-tuning for Open-Set Classification')
    parser.add_argument('--version', type=str, default='0.1',
                      help='Experiment version (e.g., 0.1, 0.2)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--save_every', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    return parser.parse_args()

def train_model(model, train_loader, val_loader, args, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    save_dir = os.path.join("checkpoints", f"v{args.version}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save experiment configuration
    config_path = os.path.join(save_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Experiment Version: v{args.version}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Number of Epochs: {args.num_epochs}\n")
        f.write(f"Save Every: {args.save_every}\n")
        f.write(f"Early Stopping Patience: {args.patience}\n")
        f.write(f"Number of Workers: {args.num_workers}\n")
    
    epoch_pbar = tqdm(range(args.num_epochs), desc='Training', position=0)
    epochs_no_improvement = 0
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}', position=1, leave=False)
        for images, labels in batch_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss/len(train_loader):.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'val_acc': f'{val_acc:.2f}%'
        })
        
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
        
        if val_acc - best_val_acc > 0.5:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model.pth'))
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= args.patience:
                print(f"No improvement in {epochs_no_improvement} epochs. Stopping training.")
                break

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
    
    model = CLIPClassifier(clip_model).to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    
    train_image_paths, train_labels, val_image_paths, val_labels = load_dataset()
    
    train_dataset = CLIPClassifierDataset(train_image_paths, train_labels, transform)
    val_dataset = CLIPClassifierDataset(val_image_paths, val_labels, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    train_model(model, train_loader, val_loader, args, device=device)

if __name__ == '__main__':
    main()
