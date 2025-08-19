# 📶 Telecom X – Predicción de Cancelación (Churn)

---

## 1️⃣ Resumen Ejecutivo
**Objetivo:** Identificar clientes con alto riesgo de cancelación (**churn**) para focalizar estrategias de retención y maximizar ROI.

**Enfoque:** Se implementó un pipeline de ML con:
- Preprocesamiento de datos
- Selección de variables **dentro del pipeline** (sin fugas)
- Entrenamiento de **Regresión Logística (LR)** y **Random Forest (RF)**

**Resultados clave (Holdout 30%):**

| Modelo | ROC-AUC | PR-AUC | Accuracy | Recall (churn) | Precisión (churn) | F1 |
|--------|---------|--------|---------|----------------|-----------------|----|
| **LR_L1_select** | **0.845** | **0.659** | 0.75 | **0.81** | 0.52 | **0.63** |
| **RF_select** | 0.824 | 0.609 | **0.78** | 0.60 | **0.60** | 0.59 |

**Interpretación ejecutiva:**
- **LR**: detecta más churners (alto recall) → indicado si **perder clientes es costoso**.  
- **RF**: reduce falsas alarmas (mayor precisión y accuracy) → útil si el **presupuesto de retención es limitado**.  
- **Umbral operativo LR sugerido:** ~0.505 (max F1), ajustable según estrategia comercial.

---

## 2️⃣ Organización de Archivos


---

## 3️⃣ Preparación de Datos (resumen)
- **Categorías → dummies** (`drop_first=True`)  
- **Numéricas:** tenure, cargos totales/mensuales  
- **Normalización:** `StandardScaler` para LR dentro del pipeline  
- **Balanceo:** SMOTE solo en datos de entrenamiento  
- **Split:** 70% entrenamiento / 30% prueba, estratificado + CV 5-fold  
- **Selección de variables:** `SelectFromModel` con **LR-L1** y **RF** (`threshold='median'`)  

**Razonamiento:**  
- Escalado + dummies mejoran modelos lineales  
- SMOTE y selección dentro de CV evitan **fugas de información**  
- Métricas clave: **PR-AUC**, **recall**, **ROC-AUC** (ranking global)  
- Umbral de predicción ajustable según **costo-beneficio de retención**

---

## 4️⃣ Insights del EDA
- **Riesgo alto**: contrato *month-to-month*, tenure bajo, cargos altos, Electronic check, sin OnlineSecurity/TechSupport  
- **Riesgo bajo / protector**: contratos 1/2 años, tenure alto, OnlineSecurity/TechSupport=Yes  
- Visualizaciones útiles: heatmap de correlaciones, barras por categoría, distribuciones por churn, coeficientes LR e importancias RF

---

## 5️⃣ Factores Predictivos Principales
*(según coeficientes LR e importancias RF)*

| Factor | Impacto sobre churn |
|--------|------------------|
| Contrato *month-to-month* | ↑ churn |
| Contrato 1/2 años | ↓ churn |
| Tenure bajo | ↑ churn |
| Tenure alto | ↓ churn |
| Cargos mensuales / totales altos | ↑ churn |
| OnlineSecurity / TechSupport ausentes | ↑ churn |
| Electronic check | ↑ churn |
| Fiber optic | relevante, dirección según coeficiente |

---

## 6️⃣ Estrategia de Retención
**Segmentación según probabilidad de churn (p):**

| Riesgo | Probabilidad | Acciones sugeridas |
|--------|-------------|-----------------|
| **Alto** | p ≥ 0.60 | Month-to-month, tenure < 6–12m, cargos altos, sin seguridad/soporte, Electronic check → Ofertas permanencia 12/24m + bundle seguridad/soporte 3–6m |
| **Medio** | 0.40 ≤ p < 0.60 | Migración a contrato anual, add-ons, educación de beneficios |
| **Bajo** | p < 0.40 | Comunicaciones de mantenimiento, referidos |

**Experimentos A/B recomendados:**  
- Comparar incentivos: **precio vs bundle**  
- Métrica: retención 60–90 días, ROI vs grupo control  

---

## 7️⃣ Flujo de Implementación
**Instalación de librerías**
import pandas as pd
df = pd.read_csv('data/processed/df_clean.csv')  # Colab: '/content/df_clean.csv'

```bash
pip install -U pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib statsmodels
