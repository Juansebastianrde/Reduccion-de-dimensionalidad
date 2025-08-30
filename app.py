# ============================================
# Paso 1 — Librerías y configuración Streamlit
# ============================================
import streamlit as st

st.set_page_config(page_title="Proyecto ML - Librerías y Setup", layout="wide")

# --- Librerías base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Estadística / ML
from scipy.stats import spearmanr, stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# --- prince (MCA) con manejo de ausencia
try:
    import prince
    HAS_PRINCE = True
except Exception:
    HAS_PRINCE = False

st.title("Proyecto ML — Setup de Librerías")
st.markdown("Este módulo adapta la celda de **carga de librerías** a un entorno Streamlit.")

# Estado de dependencias
col1, col2 = st.columns(2)
with col1:
    st.success("✅ pandas, numpy, matplotlib, scipy, scikit-learn disponibles")
with col2:
    if HAS_PRINCE:
        st.success("✅ `prince` instalado (MCA habilitado)")
    else:
        st.warning(
            "ℹ️ `prince` no está instalado. "
            "Para habilitar MCA, agrega `prince` a tu `requirements.txt` "
            "o instala localmente con `pip install prince`."
        )

with st.expander("Descripción de la base de datos", expanded=True):
    st.markdown("""
Este conjunto de datos corresponde a los registros de **14.845 admisiones hospitalarias** 
(**12.238** pacientes, incluyendo **1.921** con múltiples ingresos) recogidos durante un período de dos años 
(**1 de abril de 2017** a **31 de marzo de 2019**) en el **Hero DMC Heart Institute**, unidad del 
**Dayanand Medical College and Hospital** en **Ludhiana, Punjab, India**.

**La información incluye:**

- **Datos demográficos:** edad, género y procedencia (rural o urbana).
- **Detalles de admisión:** tipo de admisión (emergencia u OPD), fechas de ingreso y alta, 
  duración total de la estancia y **duración en UCI** *(columna objetivo en este proyecto)*.
- **Antecedentes médicos:** tabaquismo, consumo de alcohol, diabetes mellitus (DM), hipertensión (HTN),
  enfermedad arterial coronaria (CAD), cardiomiopatía previa (CMP) y enfermedad renal crónica (CKD).
- **Parámetros de laboratorio:** hemoglobina (HB), conteo total de leucocitos (TLC), plaquetas, glucosa, 
  urea, creatinina, péptido natriurético cerebral (BNP), enzimas cardíacas elevadas (RCE) y fracción de eyección (EF).
- **Condiciones clínicas y comorbilidades:** más de 28 variables como insuficiencia cardíaca, infarto con elevación del ST (STEMI),
  embolia pulmonar, shock, infecciones respiratorias, entre otras.
- **Resultado hospitalario:** estado al alta (alta médica o fallecimiento).
    """)

# ================================
# Diccionario de variables (UI)
# ================================
with st.expander("Diccionario de variables", expanded=True):
    data = [
        {"Nombre de la variable":"SNO","Nombre completo":"Serial Number","Explicación breve":"Número único de registro"},
        {"Nombre de la variable":"MRD No.","Nombre completo":"Admission Number","Explicación breve":"Número asignado al ingreso"},
        {"Nombre de la variable":"D.O.A","Nombre completo":"Date of Admission","Explicación breve":"Fecha en que el paciente fue admitido"},
        {"Nombre de la variable":"D.O.D","Nombre completo":"Date of Discharge","Explicación breve":"Fecha en que el paciente fue dado de alta"},
        {"Nombre de la variable":"AGE","Nombre completo":"AGE","Explicación breve":"Edad del paciente"},
        {"Nombre de la variable":"GENDER","Nombre completo":"GENDER","Explicación breve":"Sexo del paciente"},
        {"Nombre de la variable":"RURAL","Nombre completo":"RURAL(R) /Urban(U)","Explicación breve":"Zona de residencia (rural/urbana)"},
        {"Nombre de la variable":"TYPE OF ADMISSION-EMERGENCY/OPD","Nombre completo":"TYPE OF ADMISSION-EMERGENCY/OPD","Explicación breve":"Si el ingreso fue por urgencias o consulta externa"},
        {"Nombre de la variable":"month year","Nombre completo":"month year","Explicación breve":"Mes y año del ingreso"},
        {"Nombre de la variable":"DURATION OF STAY","Nombre completo":"DURATION OF STAY","Explicación breve":"Días totales de hospitalización"},
        {"Nombre de la variable":"duration of intensive unit stay","Nombre completo":"duration of intensive unit stay","Explicación breve":"Duración de la estancia en UCI"},
        {"Nombre de la variable":"OUTCOME","Nombre completo":"OUTCOME","Explicación breve":"Resultado del paciente (alta, fallecimiento, etc.)"},
        {"Nombre de la variable":"SMOKING","Nombre completo":"SMOKING","Explicación breve":"Historial de consumo de tabaco"},
        {"Nombre de la variable":"ALCOHOL","Nombre completo":"ALCOHOL","Explicación breve":"Historial de consumo de alcohol"},
        {"Nombre de la variable":"DM","Nombre completo":"Diabetes Mellitus","Explicación breve":"Diagnóstico de diabetes mellitus"},
        {"Nombre de la variable":"HTN","Nombre completo":"Hypertension","Explicación breve":"Diagnóstico de hipertensión arterial"},
        {"Nombre de la variable":"CAD","Nombre completo":"Coronary Artery Disease","Explicación breve":"Diagnóstico de enfermedad coronaria"},
        {"Nombre de la variable":"PRIOR CMP","Nombre completo":"CARDIOMYOPATHY","Explicación breve":"Historial de miocardiopatía"},
        {"Nombre de la variable":"CKD","Nombre completo":"CHRONIC KIDNEY DISEASE","Explicación breve":"Diagnóstico de enfermedad renal crónica"},
        {"Nombre de la variable":"HB","Nombre completo":"Haemoglobin","Explicación breve":"Nivel de hemoglobina en sangre"},
        {"Nombre de la variable":"TLC","Nombre completo":"TOTAL LEUKOCYTES COUNT","Explicación breve":"Conteo total de leucocitos"},
        {"Nombre de la variable":"PLATELETS","Nombre completo":"PLATELETS","Explicación breve":"Conteo de plaquetas"},
        {"Nombre de la variable":"GLUCOSE","Nombre completo":"GLUCOSE","Explicación breve":"Nivel de glucosa en sangre"},
        {"Nombre de la variable":"UREA","Nombre completo":"UREA","Explicación breve":"Nivel de urea en sangre"},
        {"Nombre de la variable":"CREATININE","Nombre completo":"CREATININE","Explicación breve":"Nivel de creatinina en sangre"},
        {"Nombre de la variable":"BNP","Nombre completo":"B-TYPE NATRIURETIC PEPTIDE","Explicación breve":"Péptido relacionado con función cardíaca"},
        {"Nombre de la variable":"RAISED CARDIAC ENZYMES","Nombre completo":"RAISED CARDIAC ENZYMES","Explicación breve":"Presencia de enzimas cardíacas elevadas"},
        {"Nombre de la variable":"EF","Nombre completo":"Ejection Fraction","Explicación breve":"Fracción de eyección cardíaca"},
        {"Nombre de la variable":"SEVERE ANAEMIA","Nombre completo":"SEVERE ANAEMIA","Explicación breve":"Presencia de anemia grave"},
        {"Nombre de la variable":"ANAEMIA","Nombre completo":"ANAEMIA","Explicación breve":"Presencia de anemia"},
        {"Nombre de la variable":"STABLE ANGINA","Nombre completo":"STABLE ANGINA","Explicación breve":"Dolor torácico estable por angina"},
        {"Nombre de la variable":"ACS","Nombre completo":"Acute coronary Syndrome","Explicación breve":"Síndrome coronario agudo"},
        {"Nombre de la variable":"STEMI","Nombre completo":"ST ELEVATION MYOCARDIAL INFARCTION","Explicación breve":"Infarto agudo de miocardio con elevación del ST"},
        {"Nombre de la variable":"ATYPICAL CHEST PAIN","Nombre completo":"ATYPICAL CHEST PAIN","Explicación breve":"Dolor torácico no típico"},
        {"Nombre de la variable":"HEART FAILURE","Nombre completo":"HEART FAILURE","Explicación breve":"Diagnóstico de insuficiencia cardíaca"},
        {"Nombre de la variable":"HFREF","Nombre completo":"HEART FAILURE WITH REDUCED EJECTION FRACTION","Explicación breve":"Insuficiencia cardíaca con fracción de eyección reducida"},
        {"Nombre de la variable":"HFNEF","Nombre completo":"HEART FAILURE WITH NORMAL EJECTION FRACTION","Explicación breve":"Insuficiencia cardíaca con fracción de eyección conservada"},
        {"Nombre de la variable":"VALVULAR","Nombre completo":"Valvular Heart Disease","Explicación breve":"Enfermedad de válvulas cardíacas"},
        {"Nombre de la variable":"CHB","Nombre completo":"Complete Heart Block","Explicación breve":"Bloqueo cardíaco completo"},
        {"Nombre de la variable":"SSS","Nombre completo":"Sick sinus syndrome","Explicación breve":"Síndrome de disfunción sinusal"},
        {"Nombre de la variable":"AKI","Nombre completo":"ACUTE KIDNEY INJURY","Explicación breve":"Lesión renal aguda"},
        {"Nombre de la variable":"CVA INFRACT","Nombre completo":"Cerebrovascular Accident INFRACT","Explicación breve":"Accidente cerebrovascular isquémico"},
        {"Nombre de la variable":"CVA BLEED","Nombre completo":"Cerebrovascular Accident BLEED","Explicación breve":"Accidente cerebrovascular hemorrágico"},
        {"Nombre de la variable":"AF","Nombre completo":"Atrial Fibrilation","Explicación breve":"Fibrilación auricular"},
        {"Nombre de la variable":"VT","Nombre completo":"Ventricular Tachycardia","Explicación breve":"Taquicardia ventricular"},
        {"Nombre de la variable":"PSVT","Nombre completo":"PAROXYSMAL SUPRA VENTRICULAR TACHYCARDIA","Explicación breve":"Taquicardia supraventricular paroxística"},
        {"Nombre de la variable":"CONGENITAL","Nombre completo":"Congenital Heart Disease","Explicación breve":"Enfermedad cardíaca congénita"},
        {"Nombre de la variable":"UTI","Nombre completo":"Urinary tract infection","Explicación breve":"Infección de vías urinarias"},
        {"Nombre de la variable":"NEURO CARDIOGENIC SYNCOPE","Nombre completo":"NEURO CARDIOGENIC SYNCOPE","Explicación breve":"Síncope de origen cardiogénico"},
        {"Nombre de la variable":"ORTHOSTATIC","Nombre completo":"ORTHOSTATIC","Explicación breve":"Hipotensión postural"},
        {"Nombre de la variable":"INFECTIVE ENDOCARDITIS","Nombre completo":"INFECTIVE ENDOCARDITIS","Explicación breve":"Inflamación de las válvulas cardíacas por infección"},
        {"Nombre de la variable":"DVT","Nombre completo":"Deep venous thrombosis","Explicación breve":"Trombosis venosa profunda"},
        {"Nombre de la variable":"CARDIOGENIC SHOCK","Nombre completo":"CARDIOGENIC SHOCK","Explicación breve":"Shock de origen cardíaco"},
        {"Nombre de la variable":"SHOCK","Nombre completo":"SHOCK","Explicación breve":"Shock por otras causas"},
        {"Nombre de la variable":"PULMONARY EMBOLISM","Nombre completo":"PULMONARY EMBOLISM","Explicación breve":"Bloqueo de arterias pulmonares por coágulo"},
        {"Nombre de la variable":"CHEST INFECTION","Nombre completo":"CHEST INFECTION","Explicación breve":"Infección pulmonar"},
        {"Nombre de la variable":"DAMA","Nombre completo":"Discharged Against Medical Advice","Explicación breve":"Alta médica solicitada por el paciente en contra de la recomendación"},
    ]

    dicc_df = pd.DataFrame(data, columns=["Nombre de la variable","Nombre completo","Explicación breve"])

    # Buscador simple
    q = st.text_input("Buscar en el diccionario…")
    if q:
        mask = dicc_df.apply(lambda r: r.astype(str).str.contains(q, case=False, na=False).any(), axis=1)
        st.dataframe(dicc_df[mask], use_container_width=True)
    else:
        st.dataframe(dicc_df, use_container_width=True)

    # Descargar CSV
    csv_bytes = dicc_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Descargar diccionario (CSV)", data=csv_bytes,
                       file_name="diccionario_variables.csv", mime="text/csv")
import pandas as pd

url = "https://raw.githubusercontent.com/Juansebastianrde/Reduccion-de-dimensionalidad/main/HDHI%20Admission%20data.csv"
df = pd.read_csv(url, sep=None, engine="python")  # infiere el separador

st.header("1. Cargar base de datos")

# Lee el CSV desde el archivo local (misma carpeta que la app)
bd = pd.read_csv("HDHI Admission data.csv", sep=None, engine="python")  # infiere el separador
st.success(f"Datos cargados: {bd.shape}")

# Equivalente a bd.head()
st.dataframe(bd.head(5), use_container_width=True)


st.subheader("Resumen de columnas (nulos y dtypes)")

# bd: tu DataFrame ya cargado
summary = pd.DataFrame({
    "dtype": bd.dtypes.astype(str),
    "non_null": bd.notna().sum(),
    "nulls": bd.isna().sum(),
    "unique": bd.nunique(dropna=True)
})
summary["%nulls"] = (summary["nulls"] / len(bd) * 100).round(2)

# Orden sugerido de columnas
summary = summary[["dtype", "non_null", "nulls", "%nulls", "unique"]]

st.caption(f"Filas: {len(bd):,}")
st.dataframe(summary, use_container_width=True)

st.subheader("Eliminar columna: BNP")
st.markdown("**Se decide eliminar la variabla BNP dado que tiene más del 50% de valores faltantes.**")

if "BNP" in bd.columns:
    bd.drop("BNP", axis=1, inplace=True)
    st.success("Columna 'BNP' eliminada.")
else:
    st.info("La columna 'BNP' no existe en el DataFrame.")


st.header("2. Eliminar variables y transformar datos")

# Partimos del DataFrame cargado previamente: `bd`
st.write("Shape inicial:", bd.shape)

# ==============================
# Eliminar variables innecesarias
# ==============================
st.markdown("**Eliminar variables innecesarias**")
cols_drop = ["SNO", "MRD No.", "month year"]
present = [c for c in cols_drop if c in bd.columns]
df = bd.drop(columns=present, errors="ignore").copy()
st.success(f"Columnas eliminadas: {', '.join(present) if present else 'ninguna (no se encontraron)'}")

# =======================================
# Transformar variables de fecha a datetime
# =======================================
st.markdown("**Transformar variables de fecha a formato datetime**")
if "D.O.A" in df.columns:
    df["D.O.A"] = pd.to_datetime(df["D.O.A"], format="%m/%d/%Y", errors="coerce")
if "D.O.D" in df.columns:
    df["D.O.D"] = pd.to_datetime(df["D.O.D"], format="%m/%d/%Y", errors="coerce")

# ===========================================================
# Limpiar numéricas que vienen como texto y convertir a número
# ===========================================================
st.markdown("**Tratamiento de variables numéricas mal tipadas**")
cols_to_clean = ["HB", "TLC", "PLATELETS", "GLUCOSE", "UREA", "CREATININE", "EF"]
cols_found = [c for c in cols_to_clean if c in df.columns]

for col in cols_found:
    df[col] = (
        df[col]
        .astype(str)  # asegurar string
        .str.strip()
        .replace(["EMPTY", "nan", "NaN", "None", ""], np.nan)  # a NaN
        .str.replace(r"[<>]", "", regex=True)  # quitar > y <
        .str.replace(",", ".", regex=False)    # coma decimal -> punto
    )
for col in cols_found:
    df[col] = pd.to_numeric(df[col], errors="coerce")

st.success(f"Columnas limpiadas y convertidas a numérico: {', '.join(cols_found) if cols_found else 'ninguna'}")

import streamlit as st
import pandas as pd

st.subheader("Mapeo a 0/1 y dummies")

# Toma el df ya trabajado (o usa df local si lo tienes en el scope)
df = st.session_state.get("df", df).copy()

col_adm = "TYPE OF ADMISSION-EMERGENCY/OPD"
modificadas = []

# GENDER: M/F -> 1/0
if "GENDER" in df.columns:
    df["GENDER"] = (
        df["GENDER"].astype(str).str.strip().str.upper().map({"M": 1, "F": 0})
    )
    modificadas.append("GENDER")

# RURAL: R/U -> 1/0
if "RURAL" in df.columns:
    df["RURAL"] = (
        df["RURAL"].astype(str).str.strip().str.upper().map({"R": 1, "U": 0})
    )
    modificadas.append("RURAL")

# TYPE OF ADMISSION-EMERGENCY/OPD: E/O -> 1/0
if col_adm in df.columns:
    df[col_adm] = (
        df[col_adm].astype(str).str.strip().str.upper().map({"E": 1, "O": 0})
    )
    modificadas.append(col_adm)

# CHEST INFECTION: '1'/'0' o 1/0 -> 1/0
if "CHEST INFECTION" in df.columns:
    # convierte cualquier cosa a numérico; strings inválidos -> NaN
    df["CHEST INFECTION"] = pd.to_numeric(df["CHEST INFECTION"], errors="coerce").astype("Int64")
    modificadas.append("CHEST INFECTION")

# OUTCOME -> dummies (mantén todas las categorías)
if "OUTCOME" in df.columns:
    df = pd.get_dummies(df, columns=["OUTCOME"], drop_first=False, dtype=int)
    modificadas.append("OUTCOME (dummies)")

# Booleans -> 0/1
bool_cols = df.select_dtypes(include=bool).columns
if len(bool_cols) > 0:
    df[bool_cols] = df[bool_cols].astype(int)

# Guarda y muestra
st.session_state["df"] = df
st.success(f"Columnas mapeadas: {', '.join(modificadas) if modificadas else '—'}")

cols_preview = [c for c in ["GENDER", "RURAL", col_adm, "CHEST INFECTION"] if c in df.columns]
cols_preview += [c for c in df.columns if c.startswith("OUTCOME_")]
if cols_preview:
    st.dataframe(df[cols_preview].head(), use_container_width=True)
else:
    st.info("No se encontraron columnas para mostrar en el preview.")


# ==========================================
# Convertir columnas booleanas a 0/1 (int)
# ==========================================
bool_cols = df.select_dtypes(include=bool).columns
if len(bool_cols) > 0:
    df[bool_cols] = df[bool_cols].astype(int)
    st.success(f"Booleans convertidos a 0/1: {', '.join(bool_cols)}")
else:
    st.info("No se encontraron columnas booleanas para convertir.")

st.subheader("Decisión sobre variable de UCI")
st.markdown("""
**Teniendo en cuenta que la variable que se refiere a duración en la unidad de cuidados intensivos contiene información que no se tiene cuando un paciente es ingresado al hospital, se decide eliminar con el objetivo de hacer un análisis más realista.**
""")

st.subheader("Eliminar variable de UCI")
col_uci = "duration of intensive unit stay"

if col_uci in df.columns:
    df.drop(col_uci, axis=1, inplace=True)
    st.success(f"Columna '{col_uci}' eliminada. Shape actual: {df.shape}")
else:
    st.info(f"La columna '{col_uci}' no existe en el DataFrame.")

st.subheader("Normalizar nombres de columnas (strip)")

# Ver columnas antes
cols_before = list(df.columns)
st.markdown("**Antes:**")
st.code("\n".join([str(c) for c in cols_before]))

# Aplicar strip
df.columns = df.columns.str.strip()

# Ver columnas después
cols_after = list(df.columns)
st.markdown("**Después:**")
st.code("\n".join([str(c) for c in cols_after]))

# Mostrar lista final en tabla simple
st.markdown("**Lista de columnas actual:**")
st.dataframe(pd.DataFrame({"columnas": cols_after}), use_container_width=True)

# Opcional: mostrar cuáles cambiaron
changed = [i for i, (a, b) in enumerate(zip(cols_before, cols_after)) if a != b]
if changed:
    st.success(f"Columnas modificadas (índices): {changed}")
else:
    st.info("No hubo cambios en los nombres de columnas.")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# FIX #1: Guardar el DF limpio para que TODO el resto lo use
st.session_state["df"] = df
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

st.subheader("2.1 Separación en variables categóricas y variables numéricas")

# Lista base (tal como la definiste)
raw_cat_features = [
    'GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD',
    'OUTCOME_DAMA', 'OUTCOME_DISCHARGE', 'OUTCOME_EXPIRY',
    'SMOKING', 'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD',
    'RAISED CARDIAC ENZYMES', 'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA',
    'ACS', 'STEMI', 'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF',
    'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',
    'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',
    'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',
    'PULMONARY EMBOLISM', 'CHEST INFECTION'
]

# Intersección con columnas reales del DF para evitar errores
cat_features = [c for c in raw_cat_features if c in df.columns]

# Columnas a excluir del set numérico (fechas y target si existe)
exclude = [c for c in ['D.O.A', 'D.O.D', 'DURATION OF STAY'] if c in df.columns]

# Numéricas = todo lo que no sea categórica ni excluido
num_features = [c for c in df.columns if c not in cat_features + exclude]

# Feedback visual
st.success(f"Categóricas detectadas: {len(cat_features)} | Numéricas detectadas: {len(num_features)}")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Categóricas**")
    st.write(cat_features)
with c2:
    st.markdown("**Numéricas**")
    st.write(num_features)

# Guardar en sesión para reutilizar después
st.session_state["cat_features"] = cat_features
st.session_state["num_features"] = num_features

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

st.subheader("Boxplots de variables numéricas")

# Usa df procesado si lo guardaste en la sesión; si no, usa df
df_plot = st.session_state.get("df", df)

# Detecta columnas numéricas (o usa las que guardaste antes)
num_cols_default = st.session_state.get("num_features") or df_plot.select_dtypes(include=[np.number]).columns.tolist()

# Selección de columnas a graficar (máximo 16 por rejilla como en tu ejemplo 4x4)
cols_sel = st.multiselect(
    "Selecciona variables numéricas",
    options=num_cols_default,
    default=num_cols_default[:16]
)

if len(cols_sel) == 0:
    st.info("Selecciona al menos una columna para graficar.")
else:
    n = len(cols_sel)
    cols_per_row = 4
    rows = math.ceil(n / cols_per_row)

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 4.5 * rows))
    # Asegurar vector 1D de ejes aunque rows/cols cambien
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    for i, col in enumerate(cols_sel):
        sns.boxplot(x=df_plot[col], ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", labelrotation=45)

    # Eliminar ejes vacíos si sobran
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    st.pyplot(fig)


st.subheader("2.2 Porcentaje de datos atípicos (método IQR)")
st.markdown("""
En las gráficas anteriores se identificó que varias variables numéricas presentan muchos atípicos.
A continuación se calcula el **porcentaje de outliers** por variable usando el criterio **1.5 · IQR**.
""")

st.subheader("2.3 Resumen de outliers por IQR (1.5·IQR)")
# --- OUTLIERS POR IQR (1.5·IQR) ---

df_use = st.session_state.get("df", df)
num_feats = st.session_state.get("num_features", num_features)

outliers_list = []
for c in num_feats:
    # Fuerza la columna a numérico (si hay strings -> NaN)
    col_num = pd.to_numeric(df_use[c], errors="coerce")
    s = col_num.dropna()
    if s.empty:
        continue

    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Máscara sobre la columna ya convertida a numérico
    mask = (col_num < lower) | (col_num > upper)

    temp = (
        df_use.loc[mask, [c]]
        .rename(columns={c: "value"})
        .assign(
            variable=c,
            lower_bound=lower,
            upper_bound=upper,
            row_index=lambda x: x.index
        )
    )
    outliers_list.append(temp)

if len(outliers_list) == 0:
    st.info("No se encontraron outliers con el criterio 1.5·IQR.")
else:
    outliers = pd.concat(outliers_list, ignore_index=True)
    resumen = outliers.groupby("variable").size().reset_index(name="n_outliers")
    st.dataframe(resumen.sort_values("n_outliers", ascending=False), use_container_width=True)

    # % de outliers (usa df_use por si cambió el df)
    st.subheader("Porcentaje de outliers por variable")
    resumen["pct_outliers"] = (resumen["n_outliers"] / len(df_use) * 100).round(2)
    resumen_show = resumen.sort_values("pct_outliers", ascending=False).copy()
    resumen_show["pct_outliers"] = resumen_show["pct_outliers"].map(lambda x: f"{x:.2f}%")
    st.dataframe(resumen_show, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np

st.subheader("Asimetría (skewness) de variables numéricas")

# Usa el DF procesado si está en sesión; si no, usa df
df_use = st.session_state.get("df", df)

# Toma las numéricas conocidas o detecta automáticamente
num_cols = st.session_state.get("num_features") or df_use.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    st.info("No se detectaron variables numéricas.")
else:
    df_num = df_use[num_cols]

    # 1) Asimetría con pandas
    skew_series = df_num.skew(numeric_only=True).sort_values(ascending=False)
    skew_df = skew_series.to_frame(name="skew")
    skew_df["abs_skew"] = skew_df["skew"].abs()
    skew_df = skew_df.reset_index().rename(columns={"index": "variable"})

    st.markdown("**Asimetría con pandas:**")
    st.dataframe(skew_df, use_container_width=True)

    # 2) Variables con fuerte asimetría (|skew| > 2)
    st.markdown("**Variables con |asimetría| > 2:**")
    highly_skewed = skew_df[skew_df["abs_skew"] > 2].sort_values("abs_skew", ascending=False)

    if highly_skewed.empty:
        st.success("No hay variables con |asimetría| > 2.")
    else:
        st.dataframe(highly_skewed, use_container_width=True)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.subheader("Histogramas de variables numéricas")

df_use = st.session_state.get("df", df)
num_feats = st.session_state.get("num_features") or df_use.select_dtypes(np.number).columns.tolist()

var_hist = st.multiselect("Elige variables", options=num_feats, default=num_feats)
bins = st.slider("Bins", 5, 100, 50, 5)

if var_hist:
    # NO pre-crear fig: deja que .hist cree la suya
    df_use[var_hist].hist(bins=bins, figsize=(12, 8))
    fig = plt.gcf()                 # captura la figura actual creada por .hist
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Selecciona al menos una variable.")

st.subheader("Análisis univariado — distribución de variables")

with st.expander("Resumen interpretativo", expanded=True):
    st.markdown("""
**AGE (edad)**  
- La variable AGE (edad) presenta una distribución aproximadamente normal con ligera asimetría hacia la derecha. La mayor parte de los registros se concentra entre los 50 y 70 años, lo que refleja que la población del dataset corresponde principalmente a adultos de mediana y mayor edad.

**HB (hemoglobina)**  
- La variable HB (hemoglobina) muestra una distribución bastante simétrica, con la mayor densidad de valores entre 12 y 14 g/dL. Los valores extremos por debajo de 8 g/dL o por encima de 18 g/dL son poco frecuentes, lo que indica que la mayoría de los registros se ubica en un rango considerado habitual.

**TLC (total leucocyte count)**  
- La variable TLC presenta una distribución altamente asimétrica a la derecha. La mayoría de los valores se concentra en rangos bajos, mientras que existe un número reducido de observaciones con valores muy elevados, que generan una cola larga en la distribución.

**PLATELETS (plaquetas)**  
- La variable PLATELETS tiene una distribución sesgada positivamente. La mayor concentración se encuentra entre 200,000 y 300,000, aunque se observan registros con valores más altos que extienden la cola de la distribución.

**GLUCOSE (glucosa)**  
- La variable GLUCOSE muestra una distribución asimétrica hacia la derecha, con un pico en los valores bajos y una dispersión amplia que incluye observaciones por encima de 400. Esto evidencia la presencia de valores extremos elevados en el dataset.

**UREA**  
- La variable UREA presenta una fuerte asimetría positiva. La mayoría de los valores se concentra por debajo de 100, aunque se registran observaciones con valores mucho más altos, que extienden la distribución hacia la derecha.

**CREATININE (creatinina)**  
- La variable CREATININE también exhibe una asimetría positiva pronunciada. La mayor parte de los registros se concentra en valores bajos, mientras que existen observaciones dispersas con valores más altos que alargan la cola de la distribución.

**EF (ejection fraction)**  
- La variable EF (fracción de eyección) muestra un patrón particular: existe una concentración importante de registros en el valor 60, mientras que el resto de la distribución se reparte entre valores de 20 a 40. Esto genera una distribución no simétrica con un pico muy marcado en el límite superior.
""")

st.subheader("Pairplot por género")

df_use = st.session_state.get("df", df)
num_feats_all = st.session_state.get("num_features") or df_use.select_dtypes(include=[np.number]).columns.tolist()

if "GENDER" not in df_use.columns:
    st.warning("No existe la columna 'GENDER' en el DataFrame.")
else:
    # Convertir a etiqueta si está en 0/1 (opcional)
    if set(df_use["GENDER"].dropna().unique()).issubset({0, 1}):
        hue_series = df_use["GENDER"].map({1: "M", 0: "F"})
        df_plot = df_use.copy()
        df_plot["GENDER"] = hue_series
    else:
        df_plot = df_use

    # Selección de variables numéricas para el pairplot
    sel = st.multiselect(
        "Selecciona variables numéricas (máx. 6 recomendado)",
        options=list(num_feats_all),
        default=list(num_feats_all)[:4]
    )

    if len(sel) < 2:
        st.info("Elige al menos 2 variables para el pairplot.")
    else:
        cols = sel + ["GENDER"]
        grid = sns.pairplot(df_plot[cols].dropna(), hue="GENDER", diag_kind="hist", height=2.5)
        st.pyplot(grid.fig)
        plt.close(grid.fig)

st.header("Relaciones bivariadas")

with st.expander("Hallazgos principales", expanded=True):
    st.markdown("""
### Edad vs otras variables
- No se observan tendencias lineales marcadas entre **AGE** y las demás variables.
- Los puntos están bastante dispersos en ambos géneros.

### HB vs otras variables
- Ligera correlación negativa con **Urea** y **Creatinine** (a medida que aumentan, la hemoglobina tiende a ser más baja).
- Diferencia por género: los hombres concentran valores algo más altos de **HB** en todos los rangos.

### TLC vs otras variables
- **TLC** presenta gran dispersión, con muchos valores extremos, pero no muestra relación clara con otras variables.
- La distribución por género es muy similar.

### Plaquetas (PLATELETS)
- No se aprecian correlaciones fuertes con otras variables.
- La dispersión es amplia y comparable entre hombres y mujeres.

### Glucose vs Urea/Creatinine
- No hay una correlación directa clara, aunque algunos casos con **glucosa** muy alta también muestran valores elevados de **urea** o **creatinina**.
- Ambos géneros siguen el mismo patrón.

### Urea y Creatinine
- Relación positiva clara: a mayor **creatinina**, mayor **urea**.
- Ambos géneros siguen exactamente la misma tendencia.

### EF (fracción de eyección)
- Se nota la concentración en el valor **60**.
- No hay una diferencia visible entre géneros en este patrón.
- Relación inversa tenue con **urea/creatinina**: pacientes con valores altos de estos parámetros tienden a mostrar **EF** más baja.
""")

st.subheader("¿Cuál sexo presenta mayor cantidad de hospitalizaciones?")

# Usa el DF procesado si está en sesión; si no, usa df
df_use = st.session_state.get("df", df)

if "GENDER" not in df_use.columns:
    st.warning("No existe la columna 'GENDER' en el DataFrame.")
else:
    # Normaliza posibles codificaciones de género
    g = df_use["GENDER"].copy()

    # Si viene como números 0/1
    if set(pd.Series(g.dropna().unique())).issubset({0, 1}):
        g = g.map({1: "Masculino", 0: "Femenino"})
    # Si viene como letras M/F
    elif set(pd.Series(g.dropna().astype(str).str.upper().unique())).issubset({"M", "F"}):
        g = g.astype(str).str.upper().map({"M": "Masculino", "F": "Femenino"})
    else:
        # Dejar tal cual, pero convertir a string para evitar problemas
        g = g.astype(str)

    # Conteos
    gender_counts = g.value_counts().rename_axis("Género").reset_index(name="Cantidad")
    st.dataframe(gender_counts, use_container_width=True)

    # Gráfico
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Género", y="Cantidad", data=gender_counts, ax=ax)
    ax.set_title("Distribución por sexo", fontsize=14)
    ax.set_xlabel("Sexo", fontsize=12)
    ax.set_ylabel("Cantidad de personas", fontsize=12)

    # Etiquetas encima de las barras
    for i, v in enumerate(gender_counts["Cantidad"]):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=10)

    st.pyplot(fig)
    plt.close(fig)

st.subheader("¿Cómo se ve afectada la cantidad de hospitalizaciones por la edad?")

# Usa el DataFrame procesado si está en sesión; si no, usa df
df_use = st.session_state.get("df", df)

if "AGE" not in df_use.columns:
    st.warning("No existe la columna 'AGE' en el DataFrame.")
else:
    # Controles
    bins = st.slider("Número de bins", 5, 80, 20, 1)

    # Estilo y figura
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_use, x="AGE", bins=bins, kde=False, color="blue", ax=ax)

    # Personalización
    ax.set_title("Distribución de Edades", fontsize=16)
    ax.set_xlabel("Edad", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)

    st.pyplot(fig)
    plt.close(fig)
import streamlit as st

st.subheader("Interpretación de la distribución de hospitalizaciones")

with st.expander("Resumen interpretativo", expanded=True):
    st.markdown("""
La distribución de hospitalizaciones presenta un patrón claro y esperado.

**Pico de hospitalizaciones.**  
El rango de edad con mayor número de hospitalizaciones se encuentra entre **55 y 63 años**, seguido de cerca por **63 a 68 años**. Esto es coherente con el aumento de la prevalencia de enfermedades crónicas (hipertensión, diabetes, cardiovasculares) y la acumulación de factores de riesgo a medida que las personas envejecen.

**Asimetría negativa.**  
Aunque hay hospitalizaciones en todas las edades, la mayor concentración ocurre en los grupos de mayor edad. Las personas mayores suelen tener sistemas inmunológicos más débiles y múltiples comorbilidades, lo que incrementa su vulnerabilidad a infecciones y complicaciones.

**Menor frecuencia en 0–20 años.**  
Este grupo presenta menos hospitalizaciones porque, en general, niños y adultos jóvenes tienen un sistema inmune más robusto y menor incidencia de enfermedades crónicas graves. En ellos, las hospitalizaciones suelen asociarse a accidentes, infecciones agudas o condiciones congénitas.

**Conclusión.**  
El gráfico refleja una **relación positiva entre envejecimiento y probabilidad de hospitalización**, explicada por la acumulación de riesgos y el deterioro natural del cuerpo.
""")

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.subheader("¿Existe relación entre la edad y los días de hospitalización?")

df_use = st.session_state.get("df", df)

xcol = "AGE"
ycol = "DURATION OF STAY"
if xcol not in df_use.columns or ycol not in df_use.columns:
    st.warning(f"Faltan columnas: '{xcol}' o '{ycol}'.")
else:
    # Controles
    alpha = st.slider("Transparencia de puntos (alpha)", 0.05, 1.0, 0.6, 0.05)
    add_trend = st.checkbox("Añadir línea de tendencia (regresión lineal)", value=True)
    corr_method = st.selectbox("Tipo de correlación", ["pearson", "spearman"], index=0)

    # Datos sin nulos
    data = df_use[[xcol, ycol]].dropna()

    # Gráfico
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x=xcol, y=ycol, alpha=alpha, ax=ax)
    if add_trend:
        sns.regplot(data=data, x=xcol, y=ycol, scatter=False, color="red", ax=ax)

    ax.set_title("Relación entre Edad y Días de Hospitalización")
    ax.set_xlabel("Edad del paciente")
    ax.set_ylabel("Días de hospitalización")
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)

    # Correlación
    corr = data[xcol].corr(data[ycol], method=corr_method)
    st.markdown(f"**Correlación ({corr_method}):** `{corr:.4f}`")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.header("3. Dividir conjunto de entrenamiento y prueba")

with st.expander("¿Por qué esta variable objetivo?", expanded=True):
    st.markdown("""
La variable elegida como objetivo es de tipo **numérico continuo** y representa el número de días
que un paciente permanece en el hospital. Su predicción tiene valor clínico y operativo
(planificación de recursos, disponibilidad de camas y asignación de personal).
La duración está influenciada por múltiples factores del conjunto de datos
(diagnósticos, comorbilidades y resultados de laboratorio).
""")

# DataFrame base
df_use = st.session_state.get("df", df)

# Target y features
target_default = "DURATION OF STAY" if "DURATION OF STAY" in df_use.columns else None
target = st.selectbox("Variable objetivo (y)", options=[c for c in df_use.columns], index=(list(df_use.columns).index(target_default) if target_default else 0))

# Usamos listas de features guardadas o las detectamos
num_features = st.session_state.get("num_features") or df_use.select_dtypes(include="number").columns.tolist()
cat_features = st.session_state.get("cat_features") or [c for c in df_use.columns if c not in num_features]

# Excluir fechas/target si estuvieran
exclude = [c for c in ["D.O.A", "D.O.D", target, "SNO", "MRD No."] if c in df_use.columns]
feat_list = [c for c in (num_features + cat_features) if c in df_use.columns and c not in exclude]

import re
import streamlit as st

df_use = st.session_state.get("df", df).copy()
df_use.columns = df_use.columns.str.strip()  # quita espacios

# --------- lista de columnas a excluir (con variantes) ----------
import re

df_use = st.session_state.get("df", df).copy()
df_use.columns = df_use.columns.str.strip()  # quita espacios

# columnas a eliminar (NO incluir la variable objetivo)
ban_raw = ["D.O.A", "D.O.D", "SNO", "MRD No.", "MRD No"]
def canon(name: str) -> str:
    return re.sub(r"[\W_]+", "", str(name)).lower()

ban_canon = {canon(c) for c in ban_raw}
cols_to_drop = [c for c in df_use.columns if canon(c) in ban_canon]
if cols_to_drop:
    df_use.drop(columns=cols_to_drop, inplace=True)

# >>> Corrección: excluir el target al recalcular las listas
target = st.session_state.get("target", "DURATION OF STAY")

num_features = df_use.select_dtypes(include="number").columns.tolist()
# quita el target si quedó como numérica
num_features = [c for c in num_features if c != target]

# categóricas = resto sin contar el target
cat_features = [c for c in df_use.columns if c not in num_features + [target]]

# guarda en sesión
st.session_state["df"] = df_use
st.session_state["num_features"] = num_features
st.session_state["cat_features"] = cat_features

# si después armas feat_list, filtra otra vez por seguridad:
target = "DURATION OF STAY"
exclude = [c for c in ["D.O.A", "D.O.D", target, "SNO", "MRD No.", "MRD No"] if c in df_use.columns]
feat_list = [c for c in (num_features + cat_features) if c not in exclude]



st.markdown(f"**# de features seleccionadas:** {len(feat_list)}")
st.caption(f"Excluyendo: {', '.join(exclude) if exclude else '—'}")

# Parámetros del split
col_a, col_b = st.columns(2)
with col_a:
    test_size = st.slider("Proporción de test", 0.1, 0.5, 0.3, 0.05)
with col_b:
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

# Ejecutar split
if target and feat_list:
    X = df_use[feat_list].copy()
    y = df_use[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    st.success(f"Split realizado ✅  |  X_train: {X_train.shape}  |  X_test: {X_test.shape}  |  y_train: {y_train.shape}  |  y_test: {y_test.shape}")

    # Guardar en sesión para los siguientes pasos
    st.session_state["X_train"] = X_train
    st.session_state["X_test"]  = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"]  = y_test
    st.session_state["target"]  = target
else:
    st.warning("Selecciona la variable objetivo y verifica que existan features disponibles.")


# --- aquí termina el split y se guardan en session_state ---

# =====================
# Preprocesamiento (fix)
# =====================
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd  # para reconstruir DataFrames

X_train = st.session_state.get("X_train")
X_test  = st.session_state.get("X_test")
num_features_raw = list(st.session_state.get("num_features", []))
cat_features_raw = list(st.session_state.get("cat_features", []))

if X_train is None or X_test is None:
    st.error("Primero realiza el split de entrenamiento/prueba.")
else:
    cols_train = set(X_train.columns)
    num_features = [c for c in num_features_raw if c in cols_train]
    cat_features = [c for c in cat_features_raw if c in cols_train]

    missing = sorted(set(num_features_raw + cat_features_raw) - set(num_features + cat_features))
    if missing:
        st.warning(f"Estas columnas estaban en tus listas pero NO en X_train y se ignoraron: {missing}")

    overlap = sorted(set(num_features) & set(cat_features))
    if overlap:
        st.warning(f"Columnas estaban en num y cat a la vez (se quitan de cat): {overlap}")
        cat_features = [c for c in cat_features if c not in overlap]

    transformers = []
    if num_features:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler()),
        ]), num_features))
    if cat_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]), cat_features))

    if not transformers:
        st.error("No hay columnas válidas para transformar (verifica tus listas).")
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed  = preprocessor.transform(X_test)

        out_cols = num_features + cat_features
        X_train_proc_df = pd.DataFrame(X_train_processed, columns=out_cols, index=X_train.index)
        X_test_proc_df  = pd.DataFrame(X_test_processed,  columns=out_cols, index=X_test.index)

        st.success(f"Preprocesamiento OK · X_train_proc: {X_train_proc_df.shape} · X_test_proc: {X_test_proc_df.shape}")

        st.session_state["preprocessor"] = preprocessor
        st.session_state["X_train_processed"] = X_train_proc_df
        st.session_state["X_test_processed"]  = X_test_proc_df

import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

st.subheader("📦 Preprocesamiento (imputación + RobustScaler)")

# Recupera desde el estado (deben haberse creado en el split)
X_train = st.session_state.get("X_train")
X_test  = st.session_state.get("X_test")
num_features_raw = list(st.session_state.get("num_features", []))
cat_features_raw = list(st.session_state.get("cat_features", []))

if X_train is None or X_test is None:
    st.error("Primero realiza el split de entrenamiento/prueba.")
else:
    # Asegura que las listas solo incluyan columnas presentes en X_train
    cols_train = set(X_train.columns)
    num_features = [c for c in num_features_raw if c in cols_train]
    cat_features = [c for c in cat_features_raw if c in cols_train]

    # Evita solapamientos
    overlap = sorted(set(num_features) & set(cat_features))
    if overlap:
        st.warning(f"Columnas en num y cat a la vez (se quitan de cat): {overlap}")
        cat_features = [c for c in cat_features if c not in overlap]

    # Transformadores
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", RobustScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    transformers = []
    if num_features:
        transformers.append(("num", numeric_transformer, num_features))
    if cat_features:
        transformers.append(("cat", categorical_transformer, cat_features))

    if not transformers:
        st.error("No hay columnas válidas para transformar.")
    else:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        # Aplicar el preprocesamiento
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed  = preprocessor.transform(X_test)

        # Reconstruir DataFrames (no hay OHE aquí, por eso usamos los nombres originales)
        out_cols = num_features + cat_features
        X_train_proc_df = pd.DataFrame(X_train_processed, columns=out_cols, index=X_train.index)
        X_test_proc_df  = pd.DataFrame(X_test_processed,  columns=out_cols, index=X_test.index)

        st.success(f"Preprocesamiento OK · X_train_proc: {X_train_proc_df.shape} · X_test_proc: {X_test_proc_df.shape}")

        # Guardar en sesión
        st.session_state["preprocessor"] = preprocessor
        st.session_state["X_train_processed"] = X_train_proc_df
        st.session_state["X_test_processed"]  = X_test_proc_df

        # (Opcional) muestra un vistazo
        st.dataframe(X_train_proc_df.head(), use_container_width=True)


st.header("PCA y MCA")

import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA

st.subheader("PCA sobre variables numéricas (70% varianza por defecto)")

# Recupera objetos del estado
X_train = st.session_state.get("X_train")
X_test  = st.session_state.get("X_test")
X_train_processed = st.session_state.get("X_train_processed")
X_test_processed  = st.session_state.get("X_test_processed")
num_features = list(st.session_state.get("num_features", []))

# === nombres de salida del ColumnTransformer ===
try:
    feat_out = preprocessor.get_feature_names_out()
except AttributeError:
    # fallback por si la versión de sklearn no tiene el método
    feat_out = [f"feat_{i}" for i in range(X_train_processed.shape[1])]

# limpia prefijos 'num__' / 'cat__'
feat_out = [f.split("__", 1)[1] if "__" in f else f for f in feat_out]

# === reconstruye DataFrames con el número correcto de columnas ===
X_train_proc_df = pd.DataFrame(X_train_processed, columns=feat_out, index=X_train.index)
X_test_proc_df  = pd.DataFrame(X_test_processed,  columns=feat_out, index=X_test.index)

st.success(f"Preprocesamiento OK · X_train_proc: {X_train_proc_df.shape} · X_test_proc: {X_test_proc_df.shape}")

# guarda en sesión si lo usas después
st.session_state["X_train_processed"] = X_train_proc_df
st.session_state["X_test_processed"]  = X_test_proc_df
st.session_state["preprocessor"] = preprocessor

