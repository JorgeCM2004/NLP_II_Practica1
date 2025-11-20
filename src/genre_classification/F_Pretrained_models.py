import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import Dataset  # <--- La librería clave
from typing import List, Literal
import numpy as np
from tqdm import tqdm

class Pretrained:
    
    def __init__(self, model_type: Literal["TOCHO"]):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, self.tokenizer = self.select_model()
        self.model.to(self.device)
        
    def select_model(self):
        if self.model_type.upper() == "TOCHO":
            model_name = "microsoft/deberta-v3-large"
        else:
            raise ValueError("Modelo no reconocido")
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=10,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        
        return model, tokenizer
    
    def tokenize_function(self, examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    
    def fit(self, train_texts: List[str], train_labels: List[List[int]], 
            batch_size=2, epochs=3, learning_rate=1e-5):

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        hf_dataset = Dataset.from_dict({"text": train_texts,"labels": train_labels})


        tokenized_datasets = hf_dataset.map(self.tokenize_function, batched=True)


        tokenized_datasets.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask", "labels"]
        )


        train_loader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)
        
        self.model.train()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                # Aseguramos que labels sea float para BCEWithLogitsLoss
                labels = batch['labels'].to(self.device).float()

                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Pérdida media Epoch {epoch+1}: {epoch_loss / len(train_loader):.4f}")
    
    def transform(self, texts: List[str], batch_size=8, threshold=0.5):

        self.model.eval()
        
        hf_dataset = Dataset.from_dict({"text": texts})
        
            
        print("Tokenizando para inferencia...")
        tokenized_datasets = hf_dataset.map(self.tokenize_function, batched=True)
        
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
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).int()
                
                all_predictions.extend(preds.cpu().numpy())
                
        return np.array(all_predictions)
    
    def model_information(self):
        if self.model:
            print("¡Modelo cargado exitosamente!")
            print(f"Dispositivo: {self.device}")
            param_size = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"Tamaño del modelo: {param_size:.1f} Millones de parámetros")
