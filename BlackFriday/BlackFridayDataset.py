"""
Black Friday Sales — Multi-target Classification Pipeline
==========================================================
Tarefa: Pipeline com demonstração de inferência, acurácia por classes
e medidas de certeza conforme discutido em aula.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
import itertools

# ─────────────────────────────────────────────
# CONFIGURAÇÃO E CARREGAMENTO
# ─────────────────────────────────────────────
print("=" * 65)
print("BLACK FRIDAY SALES — PIPELINE DE TREINAMENTO")
print("=" * 65)

# AVISO: O arquivo deve estar na mesma pasta deste script .py
df = pd.read_csv("retail_black_friday_sales_100k.csv")

df["purchase_date"] = pd.to_datetime(df["purchase_date"])
df["day_of_week"]   = df["purchase_date"].dt.dayofweek
df["month"]         = df["purchase_date"].dt.month

cat_cols    = ["gender", "city", "customer_segment"]
df_encoded  = pd.get_dummies(df, columns=cat_cols, drop_first=False)

NUM_FEATS = ["original_price", "discount_pct", "final_price", "quantity", 
             "purchase_amount", "purchase_hour", "is_weekend", 
             "is_black_friday", "day_of_week", "month"]
OHE_FEATS = [c for c in df_encoded.columns if any(c.startswith(b + "_") for b in cat_cols)]
BASE_FEATS = NUM_FEATS + OHE_FEATS
TARGETS = ["product_category", "payment_method", "age_group"]

def make_pipelines():
    return {
        "Random Forest": Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))]),
        "Gradient Boosting": Pipeline([("sc", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))]),
        "Logistic Regression": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
    }

def sensitivity_specificity(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sens, spec = [], []
    for i in range(len(classes)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    return np.array(sens), np.array(spec)

# ─────────────────────────────────────────────
# EXECUÇÃO DO PIPELINE
# ─────────────────────────────────────────────
all_results = {}
best_models = {}

for target in TARGETS:
    print(f"\n{'─'*65}")
    print(f"  TARGET: {target.upper()}")
    
    # 1. Demonstração do Fluxo do Pipeline
    print("  [Fluxo do Pipeline]:")
    for step_name, step_obj in make_pipelines()["Random Forest"].steps:
        print(f"   -> {step_name}: {step_obj.__class__.__name__}")
    
    X = df_encoded[BASE_FEATS].values
    le = LabelEncoder()
    y = le.fit_transform(df_encoded[target])
    classes = le.classes_
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    results = {}
    for mname, pipe in make_pipelines().items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1_w = f1_score(y_te, y_pred, average="weighted")
        sens, spec = sensitivity_specificity(y_te, y_pred, list(range(len(classes))))
        
        results[mname] = {"pipe": pipe, "le": le, "acc": acc, "f1_w": f1_w, "sens": sens, "spec": spec}
        
        # 3. Métricas em Português
        print(f"  {mname:<20} | Acurácia Global={acc:.4f} | F1-Score={f1_w:.4f} | "
              f"Sensibilidade={sens.mean():.4f} | Especificidade={spec.mean():.4f}")

    best_name = max(results, key=lambda k: results[k]["acc"])
    best_models[target] = (best_name, results[best_name])
    all_results[target] = results
    
    # 2. Acurácia por classe via Matriz de Confusão
    print(f"\n  --- Acurácia por Classe (Matriz de Confusão) ---")
    r = results[best_name]
    cm_best = confusion_matrix(y_te, r["pipe"].predict(X_te))
    acc_por_classe = cm_best.diagonal() / cm_best.sum(axis=1)
    for cls_nome, acc_cls in zip(classes, acc_por_classe):
        print(f"  Classe {cls_nome:<15}: {acc_cls:.4f}")

# ─────────────────────────────────────────────
# DEMONSTRAÇÃO DE INFERÊNCIA
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("DEMONSTRAÇÃO DO SISTEMA DE INFERÊNCIA")
print("=" * 65)

SAMPLE_IDX = [100, 5000]
for target in TARGETS:
    bname, r = best_models[target]
    pipe, le = r["pipe"], r["le"]
    for idx in SAMPLE_IDX:
        feat = df_encoded[BASE_FEATS].iloc[idx].values.reshape(1, -1)
        proba = pipe.predict_proba(feat)[0]
        pred_idx = np.argmax(proba)
        # 4. Grau de Certeza (Score)
        print(f"\n  [{target.upper()}] Cenário (Linha {idx}):")
        print(f"  Previsão: '{le.inverse_transform([pred_idx])[0]}'")
        print(f"  Grau de Certeza (Score): {proba[pred_idx]:.2%}")

print("\n✓ Pipeline executado com sucesso.")