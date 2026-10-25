# -*- coding: utf-8 -*-
"""Entrena y exporta el modelo de churn del notebook para la app Streamlit.

Reproduce exactamente la tubería del notebook:
  limpieza -> split estratificado 80/20 (semilla 42) -> StandardScaler + OneHotEncoder
  -> SMOTE (solo entrenamiento) -> LogisticRegression(C=0.01)

Genera:
  models/churn_pipeline.joblib   Pipeline de inferencia (preprocesador + clasificador)
  models/metadata.json           Columnas, categorías, límites, métricas, odds ratios y ejemplo
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# En Windows, joblib sondea núcleos físicos con wmic y ensucia stderr.
if os.name == "nt":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEMILLA = 42
RAIZ = Path(__file__).resolve().parent
CARPETA_MODELOS = RAIZ / "models"
URL_DATOS = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
VARS_NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
LIMITES = {
    "tenure": [0, 72],
    "MonthlyCharges": [0.0, 200.0],
    "TotalCharges": [0.0, 20000.0],
}
EJEMPLO_ENUNCIADO = {
    "tenure": 5,
    "MonthlyCharges": 95.0,
    "TotalCharges": 475.0,
    "Contract": "Month-to-month",
    "TechSupport": "No",
    "InternetService": "Fiber optic",
}
PROBABILIDAD_ESPERADA = 0.8768968990904534


def ruta_datos() -> Path:
    """Busca el CSV en rutas locales o lo descarga."""
    candidatos = [
        RAIZ / "data" / "Telco-Customer-Churn.csv",
        RAIZ.parent / "data" / "Telco-Customer-Churn.csv",
        RAIZ / "Telco-Customer-Churn.csv",
    ]
    for ruta in candidatos:
        if ruta.exists():
            return ruta

    destino = RAIZ / "data" / "Telco-Customer-Churn.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset no encontrado. Descargando en {destino} ...")
    urllib.request.urlretrieve(URL_DATOS, destino)
    return destino


def a_nativo(valor):
    """Convierte tipos de NumPy a tipos serializables en JSON."""
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    return valor


def nombre_legible(feature: str) -> str:
    """Convierte 'cat__Contract_One year' en 'Contract = One year'."""
    prefijo, _, resto = feature.partition("__")
    if prefijo == "num":
        return resto
    columna, _, nivel = resto.partition("_")
    return f"{columna} = {nivel}"


def main() -> int:
    print("=" * 72)
    print("ENTRENAMIENTO DEL MODELO DE CHURN PARA STREAMLIT")
    print("=" * 72)

    ruta = ruta_datos()
    df = pd.read_csv(ruta)
    print(f"Dataset: {ruta}")
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # --- Limpieza (idéntica al notebook) -----------------------------------
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nulos = int(df["TotalCharges"].isna().sum())
    if nulos:
        assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all()
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    print(f"TotalCharges vacíos imputados a 0: {nulos}")

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    columnas = list(X.columns)
    vars_cat = [c for c in columnas if c not in VARS_NUM]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEMILLA
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | Churn: {y.mean():.4f}")

    # --- Preprocesamiento ---------------------------------------------------
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), VARS_NUM),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
                vars_cat,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # --- Modelo del notebook: SMOTE solo en entrenamiento ------------------
    tuberia = ImbPipeline([
        ("prep", prep),
        ("smote", SMOTE(random_state=SEMILLA)),
        ("clf", LogisticRegression(C=0.01, max_iter=5000, random_state=SEMILLA)),
    ])
    tuberia.fit(X_train, y_train)

    proba_test = tuberia.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    metricas = {
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "precision": float(precision_score(y_test, pred_test)),
        "recall": float(recall_score(y_test, pred_test)),
        "f1": float(f1_score(y_test, pred_test)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
    }
    matriz = confusion_matrix(y_test, pred_test).astype(int).tolist()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    cv_scores = cross_val_score(tuberia, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

    print("\nMétricas sobre el conjunto de prueba (umbral 0,50):")
    for nombre, valor in metricas.items():
        print(f"  {nombre:10s}: {valor:.4f}")
    print(f"  F1 en CV 5-fold: {cv_scores.mean():.4f} ± {cv_scores.std(ddof=1):.4f}")
    print(f"  Matriz de confusión [[TN, FP], [FN, TP]]: {matriz}")

    # --- Pipeline de inferencia sin SMOTE ----------------------------------
    # En predicción SMOTE se omite; extraer prep + clf produce las mismas
    # probabilidades y elimina la dependencia de imbalanced-learn en la app.
    inferencia = Pipeline([
        ("prep", tuberia.named_steps["prep"]),
        ("clf", tuberia.named_steps["clf"]),
    ])
    assert np.allclose(
        tuberia.predict_proba(X_test), inferencia.predict_proba(X_test)
    ), "El pipeline de inferencia no reproduce al modelo completo"

    # --- Verificación del ejemplo del enunciado ----------------------------
    # Perfil base coherente: moda de las categóricas, mediana de antigüedad y
    # factura mensual, y TotalCharges derivado. Evita combinaciones imposibles
    # (por ejemplo, tenure = 1 con TotalCharges = 20 y fibra óptica).
    perfil_base = {
        c: a_nativo(X_train[c].mode().iloc[0]) for c in columnas if c not in VARS_NUM
    }
    perfil_base["tenure"] = int(X_train["tenure"].median())
    perfil_base["MonthlyCharges"] = round(float(X_train["MonthlyCharges"].median()), 2)
    perfil_base["TotalCharges"] = round(
        perfil_base["tenure"] * perfil_base["MonthlyCharges"], 2
    )
    perfil_base = {c: perfil_base[c] for c in columnas}

    ejemplo = {**perfil_base, **EJEMPLO_ENUNCIADO}
    fila_ejemplo = pd.DataFrame([ejemplo])[columnas]
    prob_ejemplo = float(inferencia.predict_proba(fila_ejemplo)[0, 1])
    print("\nEjemplo del enunciado:")
    print(f"  Probabilidad calculada : {prob_ejemplo:.6f}")
    print(f"  Probabilidad esperada  : {PROBABILIDAD_ESPERADA:.6f}")
    diferencia = abs(prob_ejemplo - PROBABILIDAD_ESPERADA)
    print(f"  Diferencia absoluta    : {diferencia:.2e}")
    if diferencia > 1e-9:
        print("ADVERTENCIA: la probabilidad no coincide con el notebook.", file=sys.stderr)
        return 1

    # --- Metadatos ----------------------------------------------------------
    nombres_features = list(tuberia.named_steps["prep"].get_feature_names_out())
    coeficientes = tuberia.named_steps["clf"].coef_[0]
    odds = np.exp(coeficientes)
    odds_ratios = sorted(
        [
            {
                "feature": f,
                "label": nombre_legible(f),
                "coef": float(c),
                "odds": float(o),
            }
            for f, c, o in zip(nombres_features, coeficientes, odds)
        ],
        key=lambda d: abs(d["coef"]),
        reverse=True,
    )

    metadatos = {
        "model_name": "Regresión Logística (C=0,01) + SMOTE",
        "model_family": "logistic_regression",
        "target": "Churn (1 = abandona, 0 = permanece)",
        "dataset": "Telco Customer Churn (blastchar)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEMILLA,
        "threshold": 0.50,
        "risk_bands": {"bajo": 0.35, "alto": 0.65},
        "columns": columnas,
        "numeric": VARS_NUM,
        "categorical": vars_cat,
        "categories": {
            c: sorted(a_nativo(v) for v in X_train[c].unique()) for c in vars_cat
        },
        "numeric_limits": LIMITES,
        "base_profile": perfil_base,
        "example": ejemplo,
        "example_probability": prob_ejemplo,
        "metrics": metricas,
        "confusion_matrix": matriz,
        "f1_cv_mean": float(cv_scores.mean()),
        "f1_cv_std": float(cv_scores.std(ddof=1)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows": int(len(df)),
        "churn_rate": float(y.mean()),
        "odds_ratios": odds_ratios,
        "versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "notes": [
            "Perfil base de la app: moda de las categóricas y mediana de tenure/MonthlyCharges; TotalCharges = tenure × MonthlyCharges.",
            "El modelo se entrenó con SMOTE solo sobre el conjunto de entrenamiento.",
            "El pipeline exportado no aplica SMOTE en predicción: produce las mismas probabilidades.",
            "La probabilidad es una estimación, no una certeza; el modelo describe asociaciones, no causalidad.",
            "Umbral de decisión 0,50; las bandas de riesgo son una capa de comunicación.",
        ],
    }

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_modelo = CARPETA_MODELOS / "churn_pipeline.joblib"
    ruta_meta = CARPETA_MODELOS / "metadata.json"
    joblib.dump(inferencia, ruta_modelo, compress=3)
    ruta_meta.write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nArtefactos generados:")
    print(f"  {ruta_modelo} ({ruta_modelo.stat().st_size / 1024:.1f} KB)")
    print(f"  {ruta_meta} ({ruta_meta.stat().st_size / 1024:.1f} KB)")
    print("\nVerificación completada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
