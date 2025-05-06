import pandas as pd
from data.dataset import AmbientDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def load_data(DATA_PATH, split=0.2):
    cad_dataset = pd.read_csv(DATA_PATH)
    cad_dataset["label"] = cad_dataset["label"] == "AMBIENT"
    cad_dataset["label"] = cad_dataset["label"].astype(int)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        cad_dataset['text'],
        cad_dataset['label'],
        test_size=split,
        random_state=42,
        stratify=cad_dataset['label']
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

     # Create Datasets
    train_dataset = AmbientDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    test_dataset = AmbientDataset(test_texts.tolist(), test_labels.tolist(), tokenizer)
    

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10)

    return train_loader, test_loader