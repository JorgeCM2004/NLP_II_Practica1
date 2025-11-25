import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset 
from typing import List, Literal, Dict
import numpy as np
from tqdm import tqdm
import torch.optim as optim

class Pretrained:
    
    def __init__(self, model_type: Literal["TOCHO"], id2label: Dict[int, str] = None, label2id: Dict[int, str] = None ):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.id2label = id2label
        self.label2id = label2id
        
        self.num_labels = len(id2label) 
  
        self.model, self.tokenizer = self.select_model()

        self.model.to(self.device)
        
    def select_model(self):
        if self.model_type.lower() == "tocho":
            model_name = "microsoft/deberta-v3-xsmall"
        else:
            raise ValueError("Modelo no encontrado")
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "num_labels": self.num_labels,
            "problem_type": "multi_label_classification",
            "ignore_mismatched_sizes": True
        }

        if self.id2label is not None:
            model_kwargs["id2label"] = self.id2label
            model_kwargs["label2id"] = self.label2id

        model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
        
        return model, tokenizer
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    def fit(self, train_texts: List[str], train_labels: List[List[int]], batch_size=10, epochs=1, learning_rate=1e-5):

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
        
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)

        tokenized_datasets.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask", "labels"]
        )

        train_loader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)
        
        self.model.train()

        scaler = torch.amp.GradScaler()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).float()

                optimizer.zero_grad()
                
                with torch.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Pérdida media Epoch {epoch+1}: {avg_loss:.4f}")
    
    def transform(self, texts: List[str], batch_size=8, threshold=0.5):
        self.model.eval()
        
        dataset = Dataset.from_dict({"text": texts})
            
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        
        tokenized_datasets.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask"]
        )
        
        inference_loader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=False)
        
        all_predictions = []

        with torch.no_grad():
            for batch in tqdm(inference_loader, desc="Inferencia"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                with torch.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).int()
                
                all_predictions.extend(preds.cpu().numpy())
                
        return np.array(all_predictions)
    
    def save_model(self, path="./saved_model"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Modelo guardado en {path}")