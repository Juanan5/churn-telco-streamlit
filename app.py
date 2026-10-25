# -*- coding: utf-8 -*-
"""Aplicación Streamlit para predecir abandono de clientes (Telco Churn).

Ejecutar:
    streamlit run app.py

El modelo y los metadatos se generan con `python train_model.py` y se cargan
desde `models/`; la app no necesita el CSV ni vuelve a entrenar.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "models" / "churn_pipeline.joblib"
RUTA_METADATOS = RAIZ / "models" / "metadata.json"

st.set_page_config(
    page_title="Predicción de Churn | Telco",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .risk-card {
        border-left: 8px solid #2E86AB;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 0.4rem 0 1rem 0;
      }
      .risk-label {
        font-size: 0.82rem;
        color: #52606D;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
      }
      .risk-value { font-size: 2.9rem; font-weight: 800; line-height: 1.05; }
      .risk-meta {
        display: flex; gap: 0.8rem; align-items: center; flex-wrap: wrap;
        margin: 0.35rem 0 0.55rem 0; color: #323F4B;
      }
      .risk-badge {
        padding: 0.22rem 0.7rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 700; color: white;
      }
      .risk-action { color: #323F4B; font-size: 0.98rem; }
      .small-note { color: #616E7C; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


@st.cache_data(show_spinner=False)
def cargar_metadatos() -> dict:
    return json.loads(RUTA_METADATOS.read_text(encoding="utf-8"))


if not RUTA_MODELO.exists() or not RUTA_METADATOS.exists():
    st.error(
        "No se encontraron los artefactos del modelo. "
        "Ejecute `python train_model.py` en esta carpeta y vuelva a iniciar la app."
    )
    st.stop()

modelo = cargar_modelo()
meta = cargar_metadatos()

COLUMNAS: list[str] = meta["columns"]
NUMERICAS: list[str] = meta["numeric"]
CATEGORICAS: list[str] = meta["categorical"]
CATEGORIAS: dict = meta["categories"]
LIMITES: dict = meta["numeric_limits"]
PERFIL_BASE: dict = meta["base_profile"]
EJEMPLO: dict = meta["example"]
UMBRAL: float = float(meta["threshold"])
BANDA_BAJA: float = float(meta["risk_bands"]["bajo"])
BANDA_ALTA: float = float(meta["risk_bands"]["alto"])


def inicializar_estado() -> None:
    """Inicializa las claves de los widgets con el perfil base."""
    for columna in COLUMNAS:
        st.session_state.setdefault(f"f_{columna}", PERFIL_BASE[columna])


def cargar_ejemplo() -> None:
    """Callback: carga el cliente del enunciado en el formulario."""
    for columna in COLUMNAS:
        st.session_state[f"f_{columna}"] = EJEMPLO[columna]


def calcular_total() -> None:
    """Callback: TotalCharges coherente para un cliente nuevo."""
    st.session_state["f_TotalCharges"] = round(
        float(st.session_state["f_tenure"])
        * float(st.session_state["f_MonthlyCharges"]),
        2,
    )


def evaluar_riesgo(probabilidad: float) -> tuple[str, str, str]:
    """Devuelve banda, color y acción sugerida."""
    if probabilidad < BANDA_BAJA:
        return (
            "Bajo",
            "#2E86AB",
            "Sin acción inmediata; mantener la comunicación habitual.",
        )
    if probabilidad < BANDA_ALTA:
        return (
            "Medio",
            "#C77700",
            "Monitoreo y contacto preventivo de bajo costo.",
        )
    return (
        "Alto",
        "#D7263D",
        "Contacto proactivo prioritario con oferta de retención.",
    )


def perfil_actual() -> dict:
    return {columna: st.session_state[f"f_{columna}"] for columna in COLUMNAS}


inicializar_estado()

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("📉 Predicción de abandono de clientes")
st.markdown(
    "**Telco Customer Churn** · Regresión Logística · "
    f"{meta['dataset_rows']:,} clientes · {len(COLUMNAS)} variables"
)
st.caption(
    "Aplicación del Ejercicio 2 de Minería de Datos. Estima la probabilidad de que un "
    "cliente abandone y sugiere una acción de retención. La predicción se actualiza al "
    "cambiar cualquier dato."
)

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Panel de control")
    st.button(
        "Cargar ejemplo del enunciado",
        type="primary",
        use_container_width=True,
        on_click=cargar_ejemplo,
        help="tenure=5, MonthlyCharges=95, Month-to-month, TechSupport=No, Fiber optic",
    )
    st.caption(
        "El ejemplo del PDF asigna los cinco valores indicados y conserva el resto del perfil base."
    )

    st.divider()
    st.subheader("Modelo")
    st.markdown(f"**{meta['model_name']}**")
    metricas = meta["metrics"]
    st.metric("F1-Score (prueba)", f"{metricas['f1']:.3f}")
    st.metric("ROC-AUC (prueba)", f"{metricas['roc_auc']:.3f}")
    st.metric(
        "F1 en validación cruzada",
        f"{meta['f1_cv_mean']:.3f} ± {meta['f1_cv_std']:.3f}",
    )
    st.caption(
        f"Entrenado: {meta['trained_at'][:10]} · Semilla {meta['seed']} · "
        f"Train {meta['train_rows']:,} / Test {meta['test_rows']:,}"
    )

    st.divider()
    st.subheader("Umbral y bandas")
    st.markdown(f"Umbral de decisión: **{UMBRAL:.2f}**")
    st.markdown(
        f"Riesgo bajo: < {BANDA_BAJA:.2f}  \n"
        f"Riesgo medio: {BANDA_BAJA:.2f} – {BANDA_ALTA:.2f}  \n"
        f"Riesgo alto: ≥ {BANDA_ALTA:.2f}"
    )
    st.caption(
        f"Tasa base de abandono en el dataset: {meta['churn_rate']:.1%}. "
        "Las bandas son una capa de comunicación; la clase se decide con el umbral."
    )

    st.divider()
    st.caption(
        "Las probabilidades son estimaciones, no certezas. El modelo describe "
        "asociaciones, no causalidad."
    )

# ---------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------
col_izq, col_der = st.columns([0.43, 0.57], gap="large")

with col_izq:
    st.subheader("Datos del cliente")

    tab_perfil, tab_servicios, tab_factura = st.tabs(
        ["Perfil", "Servicios", "Contrato y facturación"]
    )

    with tab_perfil:
        st.selectbox(
            "Género",
            CATEGORIAS["gender"],
            key="f_gender",
            help="Género declarado por el cliente.",
        )
        st.selectbox(
            "Adulto mayor",
            CATEGORIAS["SeniorCitizen"],
            format_func=lambda x: "Sí" if int(x) == 1 else "No",
            key="f_SeniorCitizen",
            help="SeniorCitizen = 1 si el cliente es adulto mayor.",
        )
        st.selectbox(
            "Tiene pareja",
            CATEGORIAS["Partner"],
            key="f_Partner",
        )
        st.selectbox(
            "Tiene dependientes",
            CATEGORIAS["Dependents"],
            key="f_Dependents",
        )
        st.slider(
            "Antigüedad (meses)",
            min_value=int(LIMITES["tenure"][0]),
            max_value=int(LIMITES["tenure"][1]),
            step=1,
            key="f_tenure",
            help="Meses de permanencia del cliente en la empresa.",
        )

    with tab_servicios:
        st.selectbox(
            "Servicio telefónico",
            CATEGORIAS["PhoneService"],
            key="f_PhoneService",
        )
        st.selectbox(
            "Líneas múltiples",
            CATEGORIAS["MultipleLines"],
            key="f_MultipleLines",
        )
        st.selectbox(
            "Servicio de internet",
            CATEGORIAS["InternetService"],
            key="f_InternetService",
        )
        st.selectbox(
            "Seguridad en línea",
            CATEGORIAS["OnlineSecurity"],
            key="f_OnlineSecurity",
        )
        st.selectbox(
            "Respaldo en línea",
            CATEGORIAS["OnlineBackup"],
            key="f_OnlineBackup",
        )
        st.selectbox(
            "Protección de dispositivos",
            CATEGORIAS["DeviceProtection"],
            key="f_DeviceProtection",
        )
        st.selectbox(
            "Soporte técnico",
            CATEGORIAS["TechSupport"],
            key="f_TechSupport",
        )
        st.selectbox(
            "Streaming TV",
            CATEGORIAS["StreamingTV"],
            key="f_StreamingTV",
        )
        st.selectbox(
            "Streaming películas",
            CATEGORIAS["StreamingMovies"],
            key="f_StreamingMovies",
        )

    with tab_factura:
        st.selectbox(
            "Tipo de contrato",
            CATEGORIAS["Contract"],
            key="f_Contract",
        )
        st.selectbox(
            "Facturación electrónica",
            CATEGORIAS["PaperlessBilling"],
            key="f_PaperlessBilling",
        )
        st.selectbox(
            "Método de pago",
            CATEGORIAS["PaymentMethod"],
            key="f_PaymentMethod",
        )
        st.number_input(
            "Factura mensual",
            min_value=float(LIMITES["MonthlyCharges"][0]),
            max_value=float(LIMITES["MonthlyCharges"][1]),
            step=0.01,
            format="%.2f",
            key="f_MonthlyCharges",
            help="Monto facturado por mes (unidades monetarias del dataset).",
        )
        st.number_input(
            "Facturación acumulada",
            min_value=float(LIMITES["TotalCharges"][0]),
            max_value=float(LIMITES["TotalCharges"][1]),
            step=0.01,
            format="%.2f",
            key="f_TotalCharges",
            help="Para un cliente nuevo, use antigüedad × factura mensual.",
        )
        st.button(
            "Calcular TotalCharges = antigüedad × factura",
            on_click=calcular_total,
            use_container_width=True,
            help="Ajusta la facturación acumulada para que sea coherente con el perfil.",
        )

    st.caption(
        "Los valores por defecto son un perfil de referencia: moda de las variables "
        "categóricas y mediana de antigüedad y factura mensual del conjunto de entrenamiento."
    )

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
perfil = perfil_actual()
fila = pd.DataFrame([perfil])[COLUMNAS]
probabilidad = float(modelo.predict_proba(fila)[0, 1])
clase = "Yes" if probabilidad >= UMBRAL else "No"
banda, color, accion = evaluar_riesgo(probabilidad)
esperado_total = float(perfil["tenure"]) * float(perfil["MonthlyCharges"])

with col_der:
    st.subheader("Resultado")

    st.markdown(
        f"""
        <div class="risk-card" style="border-left-color:{color}; background:{color}14;">
          <div class="risk-label">Probabilidad estimada de abandono</div>
          <div class="risk-value" style="color:{color};">{probabilidad:.1%}</div>
          <div class="risk-meta">
            <span class="risk-badge" style="background:{color};">Riesgo {banda}</span>
            <span>Clase predicha: <b>{clase}</b> · umbral {UMBRAL:.2f}</span>
          </div>
          <div class="risk-action"><b>Acción sugerida:</b> {accion}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        min(max(probabilidad, 0.0), 1.0),
        text=f"Probabilidad de abandono: {probabilidad:.1%}",
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidad", f"{probabilidad:.1%}")
    m2.metric("Clase", clase)
    m3.metric("Tasa base", f"{meta['churn_rate']:.1%}")

    diferencia_pp = (probabilidad - float(meta["churn_rate"])) * 100
    if probabilidad >= UMBRAL:
        st.markdown(
            f"El modelo estima que este cliente tiene una probabilidad de abandono de "
            f"**{probabilidad:.1%}**, por encima del umbral de {UMBRAL:.2f}. Se clasifica "
            f"como **Yes** (abandonará) y se ubica en la banda **{banda}**. Está "
            f"{diferencia_pp:+.1f} puntos porcentuales respecto de la tasa base."
        )
    else:
        st.markdown(
            f"El modelo estima que este cliente tiene una probabilidad de abandono de "
            f"**{probabilidad:.1%}**, por debajo del umbral de {UMBRAL:.2f}. Se clasifica "
            f"como **No** (permanecerá) y se ubica en la banda **{banda}**. Está "
            f"{diferencia_pp:+.1f} puntos porcentuales respecto de la tasa base."
        )

    if abs(float(perfil["TotalCharges"]) - esperado_total) > max(1.0, 0.02 * esperado_total):
        st.warning(
            f"La facturación acumulada ({float(perfil['TotalCharges']):,.2f}) no es "
            f"coherente con antigüedad × factura mensual ({esperado_total:,.2f}). "
            "Para un cliente nuevo, use el botón de cálculo en la pestaña "
            "«Contrato y facturación»."
        )

    with st.expander("Ver los 19 valores enviados al modelo"):
        resumen = pd.DataFrame(
            {
                "Variable": COLUMNAS,
                "Valor": [str(perfil[c]) for c in COLUMNAS],
            }
        )
        st.dataframe(resumen, hide_index=True, use_container_width=True)

        ficha = resumen.copy()
        ficha.loc[len(ficha)] = ["probabilidad_abandono", round(probabilidad, 4)]
        ficha.loc[len(ficha)] = ["clase_predicha", clase]
        ficha.loc[len(ficha)] = ["banda_riesgo", banda]
        st.download_button(
            "Descargar ficha del cliente (CSV)",
            ficha.to_csv(index=False).encode("utf-8-sig"),
            file_name="ficha_cliente_churn.csv",
            mime="text/csv",
            use_container_width=True,
        )

    tab_resultado, tab_sensibilidad, tab_modelo = st.tabs(
        ["Cómo leer el resultado", "Sensibilidad", "Sobre el modelo"]
    )

    with tab_resultado:
        st.markdown(
            f"""
            **Clase predicha = `{clase}`.** El clasificador aplica un umbral de
            {UMBRAL:.2f}: si la probabilidad es mayor o igual, marca al cliente como
            posible abandono. La probabilidad no es una certeza: un cliente con 88 %
            todavía tiene aproximadamente un 12 % de quedarse.

            **Banda de riesgo = {banda}.** Es una capa de comunicación para priorizar
            acciones, no una segunda clasificación. La banda media rodea el umbral de
            decisión para señalar casos limítrofes que conviene revisar.

            **Qué hacer.** {accion} La acción debe considerar el valor del cliente y el
            costo de la oferta; el modelo indica *quién* está en riesgo, no *qué* acción
            lo retiene. Eso debe validarse con un experimento A/B.
            """
        )
        st.info(
            "El modelo se entrenó con datos históricos de 7.043 clientes. "
            "No incluye costos, margen ni valor de vida del cliente: esas variables "
            "deben incorporarse en la decisión de negocio."
        )

    with tab_sensibilidad:
        st.markdown(
            "**¿Cómo cambia el riesgo con la antigüedad?** Se mantiene el resto del "
            "perfil y se recalcula la facturación acumulada de forma coherente. "
            "El punto rojo marca la antigüedad actual."
        )

        antiguedades = np.arange(
            int(LIMITES["tenure"][0]), int(LIMITES["tenure"][1]) + 1
        )
        filas_sens = []
        for valor in antiguedades:
            fila_sens = dict(perfil)
            fila_sens["tenure"] = int(valor)
            fila_sens["TotalCharges"] = round(
                float(valor) * float(perfil["MonthlyCharges"]), 2
            )
            filas_sens.append(fila_sens)

        sens = pd.DataFrame(filas_sens)[COLUMNAS]
        sens["probabilidad"] = modelo.predict_proba(sens)[:, 1]
        sens["antiguedad"] = antiguedades

        base = alt.Chart(sens).encode(
            x=alt.X("antiguedad:Q", title="Antigüedad (meses)"),
            y=alt.Y(
                "probabilidad:Q",
                title="Probabilidad de abandono",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
        )
        linea = base.mark_line(color="#2E86AB", strokeWidth=3)
        regla = (
            alt.Chart(pd.DataFrame({"umbral": [UMBRAL]}))
            .mark_rule(color="#D7263D", strokeDash=[6, 4], strokeWidth=1.5)
            .encode(y="umbral:Q")
        )
        actual = sens[
            sens["antiguedad"] == int(float(perfil["tenure"]))
        ]
        punto = (
            alt.Chart(actual)
            .mark_circle(size=150, color="#D7263D")
            .encode(x="antiguedad:Q", y="probabilidad:Q")
        )
        st.altair_chart(
            (linea + regla + punto)
            .properties(height=330, title="Riesgo según antigüedad")
            .configure_view(strokeWidth=0),
            use_container_width=True,
        )

        st.markdown(
            "**¿Cómo cambia el riesgo con el contrato?** Se mantiene el resto del "
            "perfil y se comparan los tres tipos de contrato."
        )
        contratos = CATEGORIAS["Contract"]
        filas_contrato = []
        for contrato in contratos:
            fila_contrato = dict(perfil)
            fila_contrato["Contract"] = contrato
            filas_contrato.append(fila_contrato)

        comp = pd.DataFrame(filas_contrato)[COLUMNAS]
        comp["probabilidad"] = modelo.predict_proba(comp)[:, 1]
        comp["Contrato"] = contratos

        barras = (
            alt.Chart(comp)
            .mark_bar()
            .encode(
                x=alt.X("Contrato:N", sort=contratos, title=""),
                y=alt.Y(
                    "probabilidad:Q",
                    title="Probabilidad de abandono",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color(
                    "Contrato:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=contratos,
                        range=["#D7263D", "#C77700", "#2E86AB"],
                    ),
                ),
            )
        )
        etiquetas = barras.mark_text(
            dy=-8, color="#1F2933", fontWeight="bold"
        ).encode(text=alt.Text("probabilidad:Q", format=".1%"))
        st.altair_chart(
            (barras + etiquetas)
            .properties(height=300, title="Riesgo según tipo de contrato")
            .configure_view(strokeWidth=0),
            use_container_width=True,
        )

        st.caption(
            "La sensibilidad es descriptiva: muestra cómo responde el modelo al variar "
            "una variable, manteniendo el resto constante. No implica causalidad."
        )

    with tab_modelo:
        st.markdown(
            f"**{meta['model_name']}** entrenada sobre {meta['train_rows']:,} clientes "
            f"y evaluada sobre {meta['test_rows']:,}. El conjunto de prueba se usó una "
            "sola vez, después de elegir el modelo por validación cruzada."
        )

        nombres = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-Score",
            "roc_auc": "ROC-AUC",
        }
        tabla_metricas = pd.DataFrame(
            [
                {"Métrica": nombres[k], "Valor": round(v, 4)}
                for k, v in metricas.items()
            ]
        )
        st.dataframe(tabla_metricas, hide_index=True, use_container_width=True)

        st.markdown("**Matriz de confusión (prueba, umbral 0,50)**")
        cm = meta["confusion_matrix"]
        tabla_cm = pd.DataFrame(
            [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
            index=["Real No", "Real Yes"],
            columns=["Predicho No", "Predicho Yes"],
        )
        st.dataframe(tabla_cm, use_container_width=True)
        st.caption(
            f"TN={cm[0][0]} · FP={cm[0][1]} · FN={cm[1][0]} · TP={cm[1][1]}. "
            "Los falsos negativos son el error más costoso: un abandono no detectado."
        )

        st.markdown("**Variables con mayor peso (odds ratios)**")
        odds = pd.DataFrame(meta["odds_ratios"]).head(10).copy()
        odds["tipo"] = np.where(
            odds["coef"] < 0, "Reduce el riesgo", "Aumenta el riesgo"
        )
        grafico_odds = (
            alt.Chart(odds)
            .mark_bar()
            .encode(
                y=alt.Y("label:N", sort="-x", title=""),
                x=alt.X("coef:Q", title="Coeficiente (log-odds)"),
                color=alt.Color(
                    "tipo:N",
                    title="",
                    scale=alt.Scale(
                        domain=["Reduce el riesgo", "Aumenta el riesgo"],
                        range=["#2E86AB", "#D7263D"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Variable"),
                    alt.Tooltip("coef:Q", title="Coeficiente", format=".3f"),
                    alt.Tooltip("odds:Q", title="Odds ratio", format=".3f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(grafico_odds, use_container_width=True)

        with st.expander("Ver la tabla completa de coeficientes y odds ratios"):
            tabla_odds = pd.DataFrame(meta["odds_ratios"])[
                ["label", "coef", "odds"]
            ].rename(
                columns={
                    "label": "Variable",
                    "coef": "Coeficiente",
                    "odds": "Odds ratio",
                }
            )
            st.dataframe(
                tabla_odds.round(4), hide_index=True, use_container_width=True
            )

        st.markdown(
            """
            **Cómo leer los odds ratios.** Las variables numéricas están estandarizadas:
            un odds ratio de 0,47 en `tenure` significa que, por cada desviación estándar
            adicional de antigüedad, las probabilidades de abandono se multiplican por
            0,47. En las categóricas, el odds ratio compara la categoría con la de
            referencia.

            **Limitaciones.** El modelo describe asociaciones, no causalidad; no incluye
            costos ni valor del cliente; se entrenó con una partición aleatoria, no
            temporal; y sus probabilidades no se calibraron formalmente. Úselo para
            priorizar, no para decidir automáticamente.
            """
        )

st.divider()
st.caption(
    f"Modelo: {meta['model_name']} · scikit-learn {meta['versions']['scikit_learn']} · "
    f"Entrenado el {meta['trained_at'][:10]} · Semilla {meta['seed']}. "
    "Aplicación de demostración académica; no sustituye el juicio del área de retención."
)
