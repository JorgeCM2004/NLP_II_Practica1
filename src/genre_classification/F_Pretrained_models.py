import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset 
from typing import List, Literal, Union
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import os

class Pretrained:
    
    def __init__(self, model_type: Literal["deberta-v3-large"], labels: List[str]):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.num_labels = len(labels)
        
        self.problem_type = "single_label_classification"
  
        self.model, self.tokenizer = self.select_model()
        self.model.to(self.device)
        
    def select_model(self):
        if self.model_type.lower() == "deberta-v3-large":
            model_name = "microsoft/deberta-v3-large" 
        else:
            raise ValueError("Modelo no encontrado")
            
        print(f"Loading model: {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception:
            print("Warning: Falling back to fast tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "num_labels": self.num_labels,
            "problem_type": self.problem_type,
            "id2label": self.id2label,
            "label2id": self.label2id,
            "ignore_mismatched_sizes": True
        }

        model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
        
        # --- CAMBIO IMPORTANTE ---
        # Desactivamos Gradient Checkpointing para evitar el error "backward through graph a second time".
        # Con batch_size=4 y tu GPU, no deberías tener problemas de memoria.
        # model.gradient_checkpointing_enable() 
        # model.config.use_cache = False 
        # -------------------------
        
        return model, tokenizer
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, max_length=512)

    def fit(self, train_texts: List[str], train_labels: Union[List[str], List[int]], batch_size=4, epochs=3, learning_rate=1e-5, weight_decay=0.01):
        
        # Limpieza de memoria GPU
        self.model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Conversión automática STR -> INT
        if len(train_labels) > 0 and isinstance(train_labels[0], str):
            print("Detectadas etiquetas de texto. Convirtiendo a IDs numéricos internamente...")
            try:
                train_labels = [self.label2id[label] for label in train_labels]
            except KeyError as e:
                raise ValueError(f"Error: La etiqueta '{e.args[0]}' no estaba en la lista de labels original pasada al __init__.")

        # Pesos para clases desbalanceadas
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels
        )
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)        
        
        dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        train_loader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        
        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
        )
        
        self.model.train()
        
        # --- CORRECCIÓN DE SINTAXIS PARA PYTORCH 2.X+ ---
        # Usamos 'cuda' como argumento para evitar el Warning y posibles errores en Nightly
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        accumulation_steps = 4 

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")
            
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = self.model(**batch)
                    loss = loss_fct(outputs.logits, batch["labels"]) 
                    loss = loss / accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({"loss": loss.item() * accumulation_steps})
            
            # Limpiar gradientes al final de la época
            if len(train_loader) % accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Average Loss Epoch {epoch+1}: {avg_loss:.4f}")

    def transform(self, texts: List[str], batch_size=16):
        self.model.eval()
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        inference_loader = DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        
        all_predictions = []

        with torch.no_grad():
            for batch in tqdm(inference_loader, desc="Inference"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = self.model(**batch)
                    logits = outputs.logits
                
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_predictions.extend(preds)
                
        return np.array(all_predictions)
    
    def save_model_and_tokenizer(self, path="./Models/Modelos_Transformer"):
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path, self.model_type)
        tokenizer_path = os.path.join(path, self.model_type + "_tokenizer")
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        print(f"Modelo guardado en {model_path}")
        print(f"Tokenizer guardado en {tokenizer_path}")