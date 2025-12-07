import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from lime.lime_text import LimeTextExplainer
import re
from IPython.display import display, HTML

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pd.set_option('display.max_colwidth', None)

'''
Funcion que predice la probabilidad de cada clase para un conjunto de textos. Procedimiento:
    1. Si el texto es una cadena, lo convierte en una lista
    2. Preprocesa los textos
    3. Transforma los textos en features
    4. Devuelve las probabilidades de cada clase
'''
def predict_proba_logreg(model, texts: List[str]):
    clean_texts = model.prep.preprocessing_data(pd.Series(texts))
    features = model.objectPrepro.transform(clean_texts)
    return model.model.predict_proba(features)

'''
Funcion que selecciona ejemplos correctos e incorrectos para mas adelante proporcionar una explicacion. Procedimiento:
    1. Genera predicciones para todos los ejemplos
    2. Selecciona n_correct ejemplos correctos
    3. Selecciona n_incorrect ejemplos incorrectos
    4. Devuelve los indices de los ejemplos seleccionados
'''
def get_analysis_data(model, x_data, y_true, n_correct=10, n_incorrect=10):
    print("Generando predicciones para seleccionar ejemplos...")
    y_pred = model.predict(x_data)
    
    correct_indices = []
    incorrect_indices = []
    
    y_true_list = y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true)
    y_pred_list = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
    
    for idx, (true_label, pred_label) in enumerate(zip(y_true_list, y_pred_list)):
        if true_label == pred_label:
            if len(correct_indices) < n_correct:
                correct_indices.append(idx)
        else:
            if len(incorrect_indices) < n_incorrect:
                incorrect_indices.append(idx)        
        if len(correct_indices) >= n_correct and len(incorrect_indices) >= n_incorrect:
            break        
    return correct_indices, incorrect_indices, y_pred

'''
Funcion que muestra la explicacion de una predicción. Procedimiento:
    1. Obtiene el texto y la etiqueta real
    2. Obtiene la predicción
    3. Obtiene las probabilidades de cada clase
    4. Obtiene la explicacion
    5. Muestra la explicacion
'''
def show_lime_explanation(idx, x_data, y_true, model_name, model_predict_proba, explainer, num_features=6):
    
    text_instance = x_data.iloc[idx]["text"] # me devuelve un texto dado un indice

    true_label = y_true[idx] # me devuelve la etiqueta real dado un indice
    
    lime_predict_fn = lambda texts: model_predict_proba(model_name, texts) # me devuelve las probabilidades de cada clase

    probs = lime_predict_fn([text_instance])[0]
    pred_idx = np.argmax(probs)
    pred_label = model_name.model.classes_[pred_idx]
    
    try:
        true_label_idx = list(model_name.model.classes_).index(true_label)
    except ValueError:
        true_label_idx = pred_idx 

    labels_to_explain = [pred_idx]
    if pred_idx != true_label_idx:
        labels_to_explain.append(true_label_idx)

    exp = explainer.explain_instance(text_instance, lime_predict_fn,num_features=num_features, labels=labels_to_explain)
    
    print(f"\n{'='*50}")
    print(f"EJEMPLO {idx}")
    print(f"{'='*50}")
    print(f"Texto (inicio): {str(text_instance)[:200]}...")
    print(f"Etiqueta Real: {true_label}")
    print(f"Predicción:   {pred_label} (Prob: {probs[pred_idx]:.4f})")
    

    display(HTML(exp.as_html(text=True))) 

    print(f"\nANÁLISIS DE LA PREDICCIÓN ({pred_label}):")
    lime_list = exp.as_list(label=pred_idx)
    
    top_positive = [k for k, v in lime_list if v > 0]
    top_negative = [k for k, v in lime_list if v < 0]
    
    print(f"Palabras que apoyan '{pred_label}':\n   {top_positive}")
    
    if pred_label != true_label:
        print(f"Palabras que confunden (apoyan otra cosa):\n   {top_negative}")