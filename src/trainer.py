# src/trainer.py
import torch
import torch.nn as nn
from tqdm import tqdm

class ClassificationTrainer:
    def __init__(self, model, device, lr=2e-5, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels, _ in tqdm(dataloader, desc="Training", leave=False):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / total, correct / total

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels, _ in dataloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / total
        self.scheduler.step(val_loss)
        return val_loss, correct / total