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

