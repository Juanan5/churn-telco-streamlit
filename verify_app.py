# -*- coding: utf-8 -*-
"""Verificación automática del artefacto y de la app Streamlit.

Ejecutar:
    python verify_app.py

Comprueba:
  1. Que el modelo carga y reproduce la probabilidad del ejemplo del enunciado.
  2. Que la app se ejecuta sin excepciones y renderiza los controles esperados.
  3. Que el botón «Cargar ejemplo del enunciado» asigna los valores del PDF.
  4. Que cambiar el contrato modifica la probabilidad en la dirección esperada.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "models" / "churn_pipeline.joblib"
RUTA_METADATOS = RAIZ / "models" / "metadata.json"


def main() -> int:
    print("=" * 70)
    print("VERIFICACIÓN DE LA APP STREAMLIT")
    print("=" * 70)

    if not RUTA_MODELO.exists() or not RUTA_METADATOS.exists():
        print("ERROR: faltan los artefactos. Ejecute `python train_model.py`.")
        return 1

    meta = json.loads(RUTA_METADATOS.read_text(encoding="utf-8"))
    modelo = joblib.load(RUTA_MODELO)

    # 1) Artefacto y ejemplo del enunciado
    fila = pd.DataFrame([meta["example"]])[meta["columns"]]
    probabilidad = float(modelo.predict_proba(fila)[0, 1])
    print(f"1. Modelo cargado: {meta['model_name']}")
    print(f"   Probabilidad del ejemplo: {probabilidad:.6f}")
    print(f"   Esperada en metadatos   : {meta['example_probability']:.6f}")
    assert abs(probabilidad - meta["example_probability"]) < 1e-9, (
        "La probabilidad del ejemplo no coincide con los metadatos"
    )
    print("   OK")

    # 2) La app se ejecuta sin excepciones
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]
    print("2. AppTest sin excepciones")
    print(f"   Selectboxes: {len(app.selectbox)} | Sliders: {len(app.slider)} "
          f"| Number inputs: {len(app.number_input)}")
    assert len(app.selectbox) == 16
    assert len(app.slider) == 1
    assert len(app.number_input) == 2
    print("   OK")

    # 3) Botón del ejemplo
    app.sidebar.button[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    probabilidad_app = next(
        m.value for m in app.metric if m.label == "Probabilidad"
    )
    clase_app = next(m.value for m in app.metric if m.label == "Clase")
    print("3. Ejemplo del enunciado")
    print(f"   Probabilidad en la app: {probabilidad_app} | Clase: {clase_app}")
    print(f"   tenure: {app.slider[0].value} | "
          f"MonthlyCharges: {app.number_input[0].value} | "
          f"TotalCharges: {app.number_input[1].value}")
    assert probabilidad_app == "87.7%"
    assert clase_app == "Yes"
    assert app.slider[0].value == 5
    assert abs(app.number_input[0].value - 95.0) < 1e-9
    assert abs(app.number_input[1].value - 475.0) < 1e-9
    print("   OK")

    # 4) Sensibilidad al contrato
    contrato = next(s for s in app.selectbox if s.label == "Tipo de contrato")
    app = contrato.select("Two year").run()
    assert not app.exception, [e.value for e in app.exception]
    probabilidad_2y = next(
        m.value for m in app.metric if m.label == "Probabilidad"
    )
    print("4. Cambio de contrato a Two year")
    print(f"   Probabilidad: {probabilidad_app} -> {probabilidad_2y}")
    assert probabilidad_2y != probabilidad_app
    print("   OK")

    print("\nVERIFICACIÓN COMPLETA: la app y el artefacto son consistentes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
