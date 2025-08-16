# streamlit app generated from your notebook (enhanced)
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from contextlib import redirect_stdout
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# New: render Markdown cells from the original notebook
import nbformat

st.set_page_config(page_title="HDHI Admission — Notebook App", layout="wide")

SKIPPED = []  # track skipped steps (with reasons)

# ---------- Helpers ----------
@st.cache_data
def load_csv_robust():
    """Try to load the dataset from common filenames in repo root."""
    candidates = [
        "HDHI Admission data.csv",
        "HDHI_Admission_data.csv",
        "HDHI Admission data.CSV",
        "HDHI_Admission_data.CSV",
    ]
    for c in candidates:
        if os.path.exists(c):
            return pd.read_csv(c)
    files = "\n".join(os.listdir("."))
    raise FileNotFoundError(
        "No se encontró el archivo de datos. Sube 'HDHI Admission data.csv' a la raíz del repo.\n"
        "Archivos en el directorio:\n" + files
    )

def show_matplotlib(figure=None):
    if figure is None:
        fig = plt.gcf()
    else:
        fig = figure
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

def capture_text(fn, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()

# ---------- Readme of notebook (Markdown cells) ----------
st.sidebar.header("Opciones de visualización")
show_md = st.sidebar.checkbox("Mostrar texto original del Notebook (.ipynb)", value=True)
md_container = st.expander("Texto original del notebook", expanded=show_md)

def render_notebook_markdown(nb_path="HDHI_Admission_Notebook.ipynb"):
    try:
        if os.path.exists(nb_path):
            nb = nbformat.read(nb_path, as_version=4)
            for cell in nb.cells:
                if cell.get("cell_type") == "markdown":
                    md_container.markdown(cell.get("source", ""))
        else:
            md_container.info("No se encontró el archivo del notebook para renderizar Markdown.")
    except Exception as e:
        md_container.error(f"No se pudo renderizar el markdown del notebook: {e}")

render_notebook_markdown()

# ---------- App UI ----------
st.title("Base de datos")

# === 1. Cargar base de datos ===
st.header("1. Cargar base de datos")
try:
    bd = load_csv_robust()
    st.write("Vista previa:")
    st.dataframe(bd.head())
    st.subheader("Información del DataFrame")
    info_text = capture_text(bd.info)
    st.code(info_text, language="text")
except Exception as e:
    st.error(str(e))
    st.stop()

# Intentar eliminar columna BNP si existe (como en tu notebook)
if "BNP" in bd.columns:
    bd.drop("BNP", axis=1, inplace=True)

# === Descripción de la base de datos (markdown del notebook) ===
st.header("Descripción de la base de datos")
st.markdown(
"""
Este conjunto de datos corresponde a los registros de 14.845 admisiones hospitalarias (12.238 pacientes, incluyendo 1.921 con múltiples ingresos) recogidos 
durante un período de dos años (1 de abril de 2017 a 31 de marzo de 2019) en el Hero DMC Heart Institute, unidad del Dayanand Medical College and Hospital 
en Ludhiana, Punjab, India.
La información incluye:
Datos demográficos: edad, género y procedencia (rural o urbana).
Detalles de admisión: tipo de admisión (emergencia u OPD), fechas de ingreso y alta, duración total de la estancia y duración en unidad de cuidados intensivos
(columna objetivo en este proyecto).
Antecedentes médicos: tabaquismo, consumo de alcohol, diabetes mellitus (DM), hipertensión (HTN), enfermedad arterial coronaria (CAD), cardiomiopatía previa (CMP),
y enfermedad renal crónica (CKD).
Parámetros de laboratorio: hemoglobina (HB), conteo total de leucocitos (TLC), plaquetas, glucosa, urea, creatinina, péptido natriurético cerebral (BNP), 
enzimas cardíacas elevadas (RCE) y fracción de eyección (EF).
Condiciones clínicas y comorbilidades: más de 28 variables como insuficiencia cardíaca, infarto con elevación del ST (STEMI), embolia pulmonar, shock, 
infecciones respiratorias, entre otras.
Resultado hospitalario: estado al alta (alta médica o fallecimiento).
"""
)

# === 2. Tratamiento de la base de datos ===
st.header("2. Tratamiento de la base de datos")

df = bd.copy()
for col in ["SNO", "MRD No."]:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)
if "month year" in df.columns:
    df.drop("month year", axis=1, inplace=True)
"""
Para iniciar el tratamiento de la base de datos, se realiza una revisión de las variables con el fin de identificar inconsistencias en su tipo de dato.
En esta etapa, se detecta que algunas variables están clasificadas como categóricas (tipo texto), pero en realidad representan valores numéricos. 
Por lo tanto, se procede a transformar dichas variables a formato numérico, asegurando su correcta interpretación en los análisis posteriores. 
Este proceso es fundamental para evitar errores en cálculos estadísticos, garantizar la adecuada aplicación de modelos predictivos y facilitar la exploración 
de relaciones entre variables.
"""
# 2.2 Fechas a datetime
for date_col in ["D.O.A", "D.O.D"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y", errors="coerce")

# 2.3 Variables numéricas que vienen como texto
cols_to_clean = ['HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'EF', 'CHEST INFECTION']
for col in cols_to_clean:
    if col in df.columns:
        df[col] = (
            df[col].astype(str).str.strip()
            .replace(['EMPTY', 'nan', 'NaN', 'None', ''], np.nan)
            .str.replace(r'[<>]', '', regex=True)
            .str.replace(',', '.', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 2.4 Mapear categóricas a numéricas/dummies
mapping_applied = False
if "GENDER" in df.columns:
    mapping_applied = True
    df["GENDER"] = df["GENDER"].map({'M': 1, 'F': 0})
if "RURAL" in df.columns:
    mapping_applied = True
    df["RURAL"] = df["RURAL"].map({'R': 1, 'U': 0})
if "TYPE OF ADMISSION-EMERGENCY/OPD" in df.columns:
    mapping_applied = True
    df["TYPE OF ADMISSION-EMERGENCY/OPD"] = df["TYPE OF ADMISSION-EMERGENCY/OPD"].map({'E': 1, 'O': 0})
if "OUTCOME" in df.columns:
    mapping_applied = True
    df = pd.get_dummies(df, columns=["OUTCOME"], drop_first=False)
if not mapping_applied:
    SKIPPED.append("Mapeos de categóricas → Algunas columnas esperadas (GENDER/RURAL/TYPE.../OUTCOME) no están en la base.")

# 2.5 Cols booleanas a int
bool_cols = df.select_dtypes(include=bool).columns
if len(bool_cols) > 0:
    df[bool_cols] = df[bool_cols].astype(int)

# 2.6 CHEST INFECTION a entero (si existe)
if "CHEST INFECTION" in df.columns:
    df["CHEST INFECTION"] = df["CHEST INFECTION"].fillna(0).astype("Int64").astype(np.int64)

# 2.7 Strip a nombres
df.columns = df.columns.str.strip()

# === Separación en variables ===
st.subheader("2.1 Separación en variables")
cat_features = [
    'GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD',
    'OUTCOME_DAMA', 'OUTCOME_DISCHARGE', 'OUTCOME_EXPIRY',
    'SMOKING', 'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD',
    'RAISED CARDIAC ENZYMES', 'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA',
    'ACS', 'STEMI', 'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF',
    'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
    'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
    'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
    'PULMONARY EMBOLISM'
]
cat_features = [c for c in cat_features if c in df.columns]
num_features = [col for col in df.columns if col not in cat_features and col not in ["D.O.A","D.O.D"]]

if len(cat_features) == 0:
    SKIPPED.append("Categóricas → Ninguna de las columnas esperadas está disponible.")
if len(num_features) == 0:
    SKIPPED.append("Numéricas → No hay columnas numéricas disponibles tras limpieza.")

st.write("**Numéricas (preview):**")
if len(num_features) > 0:
    st.dataframe(df[num_features].head())
else:
    st.info("No hay columnas numéricas para mostrar.")

st.write("**Categóricas (preview):**")
if len(cat_features) > 0:
    st.dataframe(df[cat_features].head())
else:
    st.info("No hay columnas categóricas para mostrar.")

# === 3. División y preprocesamiento ===
st.header("3. División de datos y preprocesamiento")
"""
La variable elegida como objetivo es de tipo numérico continuo y representa el número de días, o fracción de días, que un paciente permanecerá en el hospital. 
Su predicción tiene un alto valor clínico y operativo, ya que permite planificar con mayor precisión los recursos, la disponibilidad de camas y la asignación de 
personal. Además, esta duración está influenciada por múltiples factores presentes en el conjunto de datos, como diagnósticos, comorbilidades y resultados 
de laboratorio.
"""
if "DURATION OF STAY" not in df.columns:
    st.error("No se encontró la columna objetivo 'DURATION OF STAY'. Revisa el nombre exacto en tu CSV.")
    st.stop()

X = df[num_features + cat_features].copy()
y = df["DURATION OF STAY"]

test_size = st.sidebar.slider("Tamaño de test", 0.1, 0.4, 0.3, 0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, num_features), ("cat", categorical_transformer, cat_features)]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# === 4. Spearman ===

st.header("4. Selección por filtrado: Spearman (numéricas)")
"""
Se realiza un análisis de correlación de Spearman entre las variables numéricas y la variable objetivo, con el fin de identificar el grado y la dirección 
de la relación monotónica existente. Este procedimiento permite seleccionar aquellas variables que presentan una mayor asociación con la variable objetivo, 
lo cual es útil para orientar el proceso de selección de características y mejorar el rendimiento de los modelos predictivos.
"""
if len(num_features) >= 2:
    X_train_num = X_train[num_features].copy()
    train_num_with_target = X_train_num.copy()
    train_num_with_target["DURATION OF STAY"] = y_train
    correlaciones = train_num_with_target.corr(method="spearman")["DURATION OF STAY"]
    correlaciones_ordenadas = correlaciones.reindex(correlaciones.abs().sort_values(ascending=False).index)

    st.write("**Correlaciones (Spearman) ordenadas:**")
    st.code("\n".join([f"{col}: {val:.4f}" for col, val in correlaciones_ordenadas.items()]), language="text")

    df_corr = correlaciones_ordenadas.drop("DURATION OF STAY").to_frame(name="correlation")
    df_corr["abs_corr"] = df_corr["correlation"].abs()
    if df_corr["abs_corr"].sum() == 0:
        SKIPPED.append("Spearman → Correlaciones cero o NaN; no se puede calcular porcentajes.")
        numericas_significativas = []
    else:
        df_corr["percentage"] = df_corr["abs_corr"] / df_corr["abs_corr"].sum() * 100
        df_corr["cum_percentage"] = df_corr["percentage"].cumsum()
        umbral_acumulado = 90
        numericas_significativas = df_corr[df_corr["cum_percentage"] <= umbral_acumulado].index.tolist()

        plt.figure(figsize=(10,6))
        plt.bar(df_corr.index, df_corr["percentage"], label="Porcentaje individual")
        plt.plot(df_corr.index, df_corr["cum_percentage"], marker="o", label="Porcentaje acumulado")
        plt.axhline(umbral_acumulado, linestyle="--", label=f"Umbral {umbral_acumulado}%")
        plt.xticks(rotation=90)
        plt.ylabel("Porcentaje (%)")
        plt.title("Importancia por Spearman y acumulado")
        plt.legend()
        plt.tight_layout()
        show_matplotlib()

        st.success(f"Variables numéricas seleccionadas por 90% acumulado: {numericas_significativas}")
else:
    SKIPPED.append("Spearman → Se necesitan al menos 2 variables numéricas.")

# === 5. ANOVA (categóricas) ===

st.header("5. Selección por filtrado: ANOVA (categóricas)")
"""
Luego, para las variables categóricas, se aplica un análisis de varianza (ANOVA) con el objetivo de evaluar si existen diferencias estadísticamente 
significativas en la variable objetivo según los niveles de cada categoría.
"""
significativas = []
if len(cat_features) > 0:
    for col in cat_features:
        grupos = [df[df[col] == categoria]["DURATION OF STAY"].dropna() for categoria in df[col].dropna().unique()]
        grupos = [g for g in grupos if len(g) > 1]
        if len(grupos) >= 2:
            f_stat, p_val = stats.f_oneway(*grupos)
            if p_val < 0.05:
                significativas.append(col)
    st.write("**Variables categóricas significativas (p < 0.05):**")
    st.write(significativas if len(significativas) > 0 else "Ninguna con p<0.05")
else:
    SKIPPED.append("ANOVA → No hay variables categóricas disponibles.")
"""
Una vez realizados estos análisis, se seleccionan las mejores variables de cada modelo (numéricas y categóricas). Con base en estas variables seleccionadas, 
se aplican métodos automáticos de selección de características, tales como:

- KBest Selector.
- RFE (Recursive Feature Elimination).
- Random Forest.

El objetivo final es reducir el conjunto de variables a aquellas que aporten mayor relevancia al modelo predictivo, mejorando así su desempeño y evitando 
el sobreajuste.
"""
# === 6. SelectKBest ===
st.header("6. SelectKBest (sobre variables filtradas)")
var_seleccionadas = (numericas_significativas if 'numericas_significativas' in locals() else []) + significativas
if len(var_seleccionadas) > 0:
    all_cols = num_features + cat_features
    X_train_df = pd.DataFrame(X_train_processed, columns=all_cols)
    X_test_df  = pd.DataFrame(X_test_processed,  columns=all_cols)
    # filtrar solo las columnas disponibles
    var_seleccionadas = [v for v in var_seleccionadas if v in X_train_df.columns]
    if len(var_seleccionadas) == 0:
        SKIPPED.append("SelectKBest → Ninguna de las variables filtradas está en el dataset procesado.")
    else:
        X_train_filtrado = X_train_df[var_seleccionadas]
        X_test_filtrado  = X_test_df[var_seleccionadas]

        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X_train_filtrado, y_train)
        scores = selector.scores_
        ranking = sorted(zip(var_seleccionadas, scores), key=lambda x: (0 if x[1] is None or np.isnan(x[1]) else x[1]), reverse=True)
        df_scores = pd.DataFrame(ranking, columns=["Variable", "Score"])
        df_scores["Score"] = df_scores["Score"].fillna(0)
        if df_scores["Score"].sum() == 0:
            SKIPPED.append("SelectKBest → Todos los scores son 0/NaN.")
        df_scores["Perc"] = (df_scores["Score"] / (df_scores["Score"].sum() if df_scores["Score"].sum()!=0 else 1)) * 100
        df_scores["CumPerc"] = df_scores["Perc"].cumsum()

        st.dataframe(df_scores)

        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.bar(df_scores["Variable"], df_scores["Score"])
        ax1.set_xlabel("Variables")
        ax1.set_ylabel("Score F-test")
        ax1.set_xticklabels(df_scores["Variable"], rotation=45, ha="right")

        ax2 = ax1.twinx()
        ax2.plot(df_scores["Variable"], df_scores["CumPerc"], marker="o")
        ax2.set_ylabel("Cumulative %")
        ax2.set_ylim(0, 110)
        ax2.axhline(y=90, linestyle="--", linewidth=1)
        plt.title("SelectKBest - Importance & Cumulative %")
        plt.tight_layout()
        show_matplotlib(fig)

        vars_90 = df_scores[df_scores["CumPerc"] <= 90]["Variable"].tolist()
        st.info(f"Variables que acumulan el 90%: {vars_90} (Cantidad: {len(vars_90)})")
else:
    SKIPPED.append("SelectKBest → No hay variables seleccionadas por Spearman/ANOVA.")

# === 7. RFE ===
st.header("7. RFE (Linear Regression)")
if 'var_seleccionadas' in locals() and len(var_seleccionadas) > 0:
    modelo_rfe = LinearRegression()
    try:
        selector_rfe = RFE(modelo_rfe, n_features_to_select=min(max(1, len(var_seleccionadas)//2), len(var_seleccionadas)))
        selector_rfe.fit(pd.DataFrame(X_train_processed, columns=num_features + cat_features)[var_seleccionadas], y_train)
        selected_features_rfe = [c for c, keep in zip(var_seleccionadas, selector_rfe.support_) if keep]
        st.success(f"Variables seleccionadas por RFE: {selected_features_rfe}")
    except Exception as e:
        SKIPPED.append(f"RFE → Error: {e}")
        st.warning("RFE no se pudo ejecutar. Revisa el apartado 'Diagnóstico' al final.")
else:
    SKIPPED.append("RFE → No hay variables de entrada (var_seleccionadas).")

# === 8. Random Forest Importances ===
st.header("8. Random Forest — Importancia de variables")
if 'var_seleccionadas' in locals() and len(var_seleccionadas) > 0:
    try:
        X_train_filtrado = pd.DataFrame(X_train_processed, columns=num_features + cat_features)[var_seleccionadas]
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train_filtrado, y_train)
        importances = rf_model.feature_importances_
        importances_df = pd.DataFrame({"Variable": X_train_filtrado.columns, "Importancia": importances}).sort_values(
            by="Importancia", ascending=False
        )
        importances_df["Importancia_Acum"] = importances_df["Importancia"].cumsum()
        selected_vars_rf = importances_df[importances_df["Importancia_Acum"] <= 0.90]["Variable"].tolist()

        st.dataframe(importances_df)

        plt.figure(figsize=(10,6))
        plt.bar(importances_df["Variable"], importances_df["Importancia"], alpha=0.7, label="Importancia individual")
        plt.plot(importances_df["Variable"], importances_df["Importancia_Acum"], marker="o", label="Importancia acumulada")
        plt.axhline(0.90, linestyle="--", label="90% acumulado")
        plt.xticks(rotation=90)
        plt.ylabel("Importancia")
        plt.title("Ranking de importancia de variables - RandomForest")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        show_matplotlib()

        comunes = list(set([v for v in locals().get('vars_90', [])]) & set(locals().get('selected_features_rfe', [])))
        st.write("**Intersección (SelectKBest 90% ∩ RFE):**", comunes if len(comunes)>0 else "Sin intersección o pasos previos no disponibles.")
        if len(comunes) > 0:
            df_filtrado = df[comunes]
            st.write("Preview df filtrado (solo comunes):")
            st.dataframe(df_filtrado.head())
    except Exception as e:
        SKIPPED.append(f"RandomForest → Error: {e}")
        st.warning("RandomForest no se pudo ejecutar. Revisa el apartado 'Diagnóstico' al final.")
else:
    SKIPPED.append("RandomForest → No hay variables de entrada (var_seleccionadas).")

"""
En primera instancia, se aplicaron métodos de filtrado utilizando la medida de correlación de Spearman y la prueba ANOVA, seleccionando en cada caso
las variables con mayor relevancia estadística. Posteriormente, con el conjunto reducido obtenido de esta etapa, se implementaron tres métodos de 
selección de características: SelectKBest, RFE y Random Forest. Finalmente, se identificaron las variables comunes entre SelectKBest y RFE, las cuales 
fueron seleccionadas para conformar un nuevo dataframe con menor dimensionalidad, optimizando así la eficiencia del modelo sin comprometer su capacidad 
predictiva.
"""

# === 9. PCA y MCA ===
st.header("9. PCA y MCA")
# a) PCA
if len(num_features) > 0:
    try:
        num_indices = [i for i, col in enumerate(X_train.columns) if col in num_features]
        X_train_numericas = pd.DataFrame(preprocessor.transformers_[0][1].fit_transform(X_train[num_features]), columns=num_features)
        X_test_numericas  = pd.DataFrame(preprocessor.transformers_[0][1].transform(X_test[num_features]), columns=num_features)

        pca = PCA(n_components=0.70, random_state=42)
        Xn_train_pca = pca.fit_transform(X_train_numericas)
        Xn_test_pca  = pca.transform(X_test_numericas)
        pca_names = [f"PCA{i+1}" for i in range(Xn_train_pca.shape[1])]

        st.info(f"PCA: {len(pca_names)} componentes, var. explicada acumulada = {pca.explained_variance_ratio_.sum():.3f}")
        var_exp = pca.explained_variance_ratio_
        cum_var_exp = np.cumsum(var_exp)
        plt.figure(figsize=(8,5))
        plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.6, label="Varianza explicada por componente")
        plt.step(range(1, len(cum_var_exp)+1), cum_var_exp, where="mid", label="Varianza acumulada")
        plt.axhline(y=0.70, linestyle="--", label="70%")
        plt.xlabel("Componentes principales")
        plt.ylabel("Proporción de varianza explicada")
        plt.title("PCA - Varianza explicada y acumulada")
        plt.xticks(range(1, len(var_exp)+1))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        show_matplotlib()
    except Exception as e:
        SKIPPED.append(f"PCA → Error: {e}")
else:
    SKIPPED.append("PCA → No hay variables numéricas para calcular.")

# b) MCA (opcional, si 'prince' está instalado)
try:
    import prince
    if len(cat_features) > 0:
        cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_categoricas = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_features]), columns=cat_features)
        X_test_categoricas  = pd.DataFrame(cat_imputer.transform(X_test[cat_features]), columns=cat_features)

        mca = prince.MCA(n_components=5, n_iter=5, random_state=42)
        mca.fit(X_train_categoricas)
        Xc_train_mca = mca.transform(X_train_categoricas)

        ev = mca.eigenvalues_summary
        perc = ev['% of variance']
        if perc.dtype == "object":
            perc = perc.str.replace('%','', regex=False).astype(float)
        perc_values = perc.values
        cum_perc = np.cumsum(perc_values)

        st.write("**MCA inercia por eje (%):**", perc_values)

        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(perc_values)+1), perc_values, marker="o", label="% por eje")
        plt.plot(range(1, len(cum_perc)+1), cum_perc, marker="o", label="Acumulada")
        plt.xlabel("Componentes")
        plt.ylabel("% de varianza")
        plt.title("MCA — % de varianza por eje y acumulada")
        plt.legend()
        plt.tight_layout()
        show_matplotlib()
    else:
        SKIPPED.append("MCA → No hay variables categóricas para calcular.")
except ImportError:
    SKIPPED.append("MCA → Paquete 'prince' no instalado (ver requirements.txt).")
except Exception as e:
    SKIPPED.append(f"MCA → Error: {e}")



"""

Posteriormente, se realizaron análisis de componentes principales (PCA) y de correspondencias múltiples (MCA) con el fin de identificar patrones y 
reducir la dimensionalidad de los datos. El MCA presentó una inercia por eje de 8.21%, 5.03%, 4.94%, 4.05% y 3.64%, mientras que el PCA retuvo 6 componentes
que explicaron de forma acumulada el 74.8% de la varianza total. Los resultados obtenidos de ambos análisis se integraron en un nuevo *dataframe* para su 
posterior uso en la modelación.
"""
