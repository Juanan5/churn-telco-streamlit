# Aplicación Streamlit — Predicción de abandono de clientes (Telco Churn)

Aplicación web que usa la **Regresión Logística** seleccionada en
`prediccion_churn_telco.ipynb` para estimar la probabilidad de que un cliente abandone y sugerir una
acción de retención.

![Modelo](https://img.shields.io/badge/modelo-Regresi%C3%B3n%20Log%C3%ADstica-2E86AB)
![F1](https://img.shields.io/badge/F1--Score-0.6121-2E86AB)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.8380-2E86AB)

## Contenido

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Streamlit (formulario de 19 variables, resultado, sensibilidad e información del modelo) |
| `train_model.py` | Reproduce el modelo del notebook y genera `models/churn_pipeline.joblib` + `models/metadata.json` |
| `models/` | Artefacto de inferencia y metadatos versionados (la app no necesita el CSV) |
| `requirements.txt` | Dependencias fijadas al entorno verificado |
| `verify_app.py` | Verificación automática del artefacto y de la app con `AppTest` |
| `.streamlit/config.toml` | Tema visual |

## Ejecución local

```bash
cd streamlit_app
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abrir <http://localhost:8501>. El artefacto ya está incluido; **no hace falta reentrenar**.

Para regenerar el modelo desde el CSV:

```bash
python train_model.py
```

El script busca el dataset en `data/`, en `../data/` o lo descarga desde el espejo público de IBM.
Verifica que la probabilidad del cliente del enunciado sea exactamente **0,876897** antes de guardar.

## Qué muestra la app

- **Formulario completo** con las 19 variables del modelo, agrupadas en perfil, servicios y
  contrato/facturación.
- **Ejemplo del enunciado** en un clic: `tenure=5`, `MonthlyCharges=95`, `Contract=Month-to-month`,
  `TechSupport=No`, `InternetService=Fiber optic`.
- **Resultado**: probabilidad, clase (`Yes`/`No`), banda de riesgo, acción sugerida, comparación con la
  tasa base y descarga de la ficha en CSV.
- **Sensibilidad**: probabilidad según antigüedad y según tipo de contrato, manteniendo el resto del
  perfil.
- **Modelo**: métricas de prueba, matriz de confusión, odds ratios y limitaciones.

## Model card resumida

| Aspecto | Valor |
|---|---|
| Algoritmo | Regresión Logística, `C=0,01`, `max_iter=5000` |
| Preprocesamiento | `StandardScaler` + `OneHotEncoder(drop="first")` |
| Balanceo | SMOTE solo en entrenamiento (no se aplica en inferencia) |
| Datos | 7.043 clientes; 80/20 estratificado; semilla 42 |
| Accuracy / Precision | 0,7410 / 0,5079 |
| Recall / F1-Score | 0,7701 / 0,6121 |
| ROC-AUC | 0,8380 |
| Umbral | 0,50 |
| Bandas | Bajo < 0,35 · Medio 0,35–0,65 · Alto ≥ 0,65 |
| Ejemplo del PDF | 87,69 % · clase `Yes` · riesgo alto |

**Uso previsto:** priorizar clientes para campañas de retención y explorar el efecto de variables.

**No usar para:** decidir automáticamente sobre un cliente, negar servicios, fijar precios
individuales ni afirmar causalidad. El modelo describe asociaciones en datos históricos.

**Limitaciones:** sin costos ni valor de vida del cliente; partición aleatoria (no temporal); sin
calibración formal de probabilidades; entrenado con un único conjunto de datos.

## Despliegue en Streamlit Community Cloud

**Repositorio publicado:** <https://github.com/Juanan5/churn-telco-streamlit> (público).

**Enlace directo al asistente de despliegue:**
<https://share.streamlit.io/deploy?repository=Juanan5/churn-telco-streamlit&branch=main&mainModule=app.py>

La publicación final de la app requiere tu inicio de sesión en Streamlit Cloud; el repositorio ya
contiene todo lo necesario.

1. Entrar a <https://share.streamlit.io> y autorizar GitHub.
2. **New app → Deploy a public app from GitHub**.
3. Seleccionar el repositorio `Juanan5/churn-telco-streamlit`, rama `main` y archivo principal
   `app.py`.
4. En **Advanced settings**, elegir Python 3.12 (o 3.11). No se necesitan secretos ni variables de
   entorno.
5. **Deploy**. La primera compilación instala `requirements.txt` en 1–2 minutos.
6. Streamlit asigna una URL del tipo `https://churn-telco-streamlit.streamlit.app`; se puede cambiar
   el subdominio desde la configuración de la app.

### Si necesitas recrear el repositorio

```bash
cd streamlit_app
git init -b main
git add .
git commit -m "feat: app Streamlit de predicción de churn"
gh repo create churn-telco-streamlit --public --source=. --remote=origin --push
```

Es importante que `app.py`, `requirements.txt` y `models/` queden en la raíz del repositorio.

### Opción alternativa: repositorio completo del trabajo

Si se publica todo `Mineria_De_Datos`, usar como *main file path* `streamlit_app/app.py`. En ese caso
el `requirements.txt` que Streamlit lee es el de la raíz del repositorio, que corresponde a otro
ejercicio; hay que añadir allí `streamlit` y las dependencias de la app. Por eso se recomienda el
repositorio propio.

## Verificación

```bash
python verify_app.py
```

Comprueba que el artefacto carga, que la probabilidad del ejemplo coincide con el notebook y que la
app se ejecuta sin excepciones, incluyendo el clic en «Cargar ejemplo del enunciado».

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `No se encontraron los artefactos del modelo` | Falta `models/` | Ejecutar `python train_model.py` |
| Error al deserializar el modelo | Versión de scikit-learn distinta | Usar `requirements.txt` o reentrenar con `train_model.py` |
| La app no encuentra el CSV al reentrenar | Ruta distinta | Copiar el CSV a `streamlit_app/data/` o dejar que el script lo descargue |
| Streamlit Cloud no instala dependencias | `requirements.txt` fuera de la raíz | Publicar esta carpeta como raíz del repositorio |

## Reproducibilidad

El artefacto se generó con Python 3.13.5, scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.1.3 y joblib
1.4.2. `requirements.txt` fija esas versiones. Si se cambia de versión de scikit-learn, conviene
regenerar el artefacto con `train_model.py`.
