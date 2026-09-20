import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

class SignSequenceDataset(Dataset):
    def __init__(self, X, meta_df, sequence_length=30):
        self.sequences = []
        self.labels = []
        
        videos = meta_df["video"].values
        y = meta_df["encoded_label"].values
        unique_videos = np.unique(videos)
        
        for vid in unique_videos:
            idx = np.where(videos == vid)[0]
            vid_X = X[idx]
            vid_y = y[idx[0]]
            
            # Standardize sequence length
            if len(vid_X) < sequence_length:
                padding = np.zeros((sequence_length - len(vid_X), vid_X.shape[1]))
                vid_X = np.vstack((vid_X, padding))
            else:
                vid_X = vid_X[:sequence_length]
                
            self.sequences.append(vid_X)
            self.labels.append(vid_y)
            
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.labels = torch.LongTensor(np.array(self.labels))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ---------------------------------------------------------
# OPTIMIZED LSTM ARCHITECTURE
# ---------------------------------------------------------
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Reduced complexity: Only 1 layer, smaller hidden size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # FIX: Mean Pooling. Instead of looking at the last frame (which might be zero-padded),
        # we average the LSTM's outputs across the entire 30-frame sequence.
        out = torch.mean(out, dim=1) 
        out = self.fc(out)
        return out

def main():
    print("=" * 60)
    print("SIGNBRIDGE AI - PYTORCH LSTM (MEAN POOLING)")
    print("=" * 60)

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    train_meta = pd.read_csv(PROCESSED_DIR / "train_meta.csv")
    test_meta = pd.read_csv(PROCESSED_DIR / "test_meta.csv")
    
    label_encoder = joblib.load(PROCESSED_DIR / "label_encoder.pkl")
    num_classes = len(label_encoder.classes_)
    input_size = X_train.shape[1]
    
    SEQ_LEN = 30
    train_dataset = SignSequenceDataset(X_train, train_meta, sequence_length=SEQ_LEN)
    test_dataset = SignSequenceDataset(X_test, test_meta, sequence_length=SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) # Smaller batch size
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Using 32 hidden units instead of 64 to prevent overfitting the small dataset
    model = SignLSTM(input_size=input_size, hidden_size=32, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Added weight decay (L2 Regularization) to prevent memorization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    epochs = 150
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            
        model.eval()
        correct_test = 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                correct_test += (predicted == labels).sum().item()
                
        train_acc = (correct_train / len(train_dataset)) * 100
        test_acc = (correct_test / len(test_dataset)) * 100
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")
            
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), PROCESSED_DIR / "lstm_model.pth")
            
    print("=" * 60)
    print(f"Training Complete! Best Test Accuracy: {best_acc:.2f}%")
    print("Model saved to dataset/processed/lstm_model.pth")

if __name__ == "__main__":
    main()