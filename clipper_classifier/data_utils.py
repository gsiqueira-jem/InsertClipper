import os
from glob import glob
import numpy as np

def get_class_mapping():
    """Returns a dictionary mapping folder numbers to class indices"""
    mapping = {}
    # Door: 1-6 -> class 0
    for i in range(1, 7):
        mapping[str(i)] = 0
    # Window: 7-10 -> class 1
    for i in range(7, 11):
        mapping[str(i)] = 1
    # Furniture: 11-27 -> class 2
    for i in range(11, 28):
        mapping[str(i)] = 2
    # Unknown: 28-30 -> class 3
    for i in range(28, 36):
        mapping[str(i)] = 3
    return mapping

def load_folder_data(split, class_mapping, base_path="./data/dataset"):
    """Helper function to load data from a specific folder"""
    image_paths = []
    labels = []
    
    dataset_path = os.path.join(base_path, split, "clip")
    # Walk through all numbered folders
    for folder in sorted(os.listdir(dataset_path)):
        if not folder.isdigit():
            continue
            
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Get all images in the folder
        folder_image_paths = glob(os.path.join(folder_path, "*.*"))
        folder_image_paths = [p for p in folder_image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Map folder number to class index
        class_idx = class_mapping[folder]
        
        image_paths.extend(folder_image_paths)
        labels.extend([class_idx] * len(folder_image_paths))
    
    return np.array(image_paths), np.array(labels)

def load_dataset(base_path="./data/dataset", seed=42):
    """
    Load dataset from separate train and validation paths
    
    Args:
        train_path: Path to the training dataset directory
        val_path: Path to the validation dataset directory
        seed: Random seed for reproducibility
    
    Returns:
        train_image_paths, train_labels, val_image_paths, val_labels
    """
    np.random.seed(seed)
    class_mapping = get_class_mapping()
    
    # Load training data
    train_image_paths, train_labels = load_folder_data("train", class_mapping, base_path)
    
    # Load validation data
    val_image_paths, val_labels = load_folder_data("val", class_mapping, base_path)
    
    # Shuffle training data
    indices = np.random.permutation(len(train_image_paths))
    train_image_paths = train_image_paths[indices]
    train_labels = train_labels[indices]
    
    return train_image_paths, train_labels, val_image_paths, val_labels 