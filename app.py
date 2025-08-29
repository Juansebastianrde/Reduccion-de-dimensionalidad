
# streamlit run app_from_notebook.py
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
import plotly.express as px

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

def ensure_df_loaded():
    if "df" not in st.session_state or st.session_state.df is None:
        st.info("Primero carga datos en la sección '1) Cargar base de datos'.")
        st.stop()

# ---------- Readme of notebook (Markdown cells) ----------
st.sidebar.header("Opciones de visualización")
show_md = st.sidebar.checkbox("Mostrar texto original del Notebook (.ipynb)", value=True)
md_container = st.expander("Texto original del notebook", expanded=show_md)
MARKDOWN_BLOCKS = [
  "# Base de datos",
  "## Descripción de la base de datos",
  "Este conjunto de datos corresponde a los registros de 14.845 admisiones hospitalarias (12.238 pacientes, incluyendo 1.921 con múltiples ingresos) recogidos durante un período de dos años (1 de abril de 2017 a 31 de marzo de 2019) en el Hero DMC Heart Institute, unidad del Dayanand Medical College and Hospital en Ludhiana, Punjab, India.\n\nLa información incluye:\n\nDatos demográficos: edad, género y procedencia (rural o urbana).\n\nDetalles de admisión: tipo de admisión (emergencia u OPD), fechas de ingreso y alta, duración total de la estancia y duración en unidad de cuidados intensivos (columna objetivo en este proyecto).\n\nAntecedentes médicos: tabaquismo, consumo de alcohol, diabetes mellitus (DM), hipertensión (HTN), enfermedad arterial coronaria (CAD), cardiomiopatía previa (CMP), y enfermedad renal crónica (CKD).\n\nParámetros de laboratorio: hemoglobina (HB), conteo total de leucocitos (TLC), plaquetas, glucosa, urea, creatinina, péptido natriurético cerebral (BNP), enzimas cardíacas elevadas (RCE) y fracción de eyección (EF).\n\nCondiciones clínicas y comorbilidades: más de 28 variables como insuficiencia cardíaca, infarto con elevación del ST (STEMI), embolia pulmonar, shock, infecciones respiratorias, entre otras.\n\nResultado hospitalario: estado al alta (alta médica o fallecimiento).",
  "| Nombre de la variable | Nombre completo                          | Explicacion breve |\n|---------------|-------------------------------------------|-------------------|\n| SNO           | Serial Number                             | Número único de registro |\n| MRD No.       | Admission Number                          | Número asignado al ingreso |\n| D.O.A         | Date of Admission                         | Fecha en que el paciente fue admitido |\n| D.O.D         | Date of Discharge                         | Fecha en que el paciente fue dado de alta |\n| AGE           | AGE                                       | Edad del paciente |\n| GENDER        | GENDER                                    | Sexo del paciente |\n| RURAL         | RURAL(R) /Urban(U)                        | Zona de residencia (rural/urbana) |\n| TYPE OF ADMISSION-EMERGENCY/OPD | TYPE OF ADMISSION-EMERGENCY/OPD | Si el ingreso fue por urgencias o consulta externa |\n| month year    | month year                                | Mes y año del ingreso |\n| DURATION OF STAY | DURATION OF STAY                       | Días totales de hospitalización |\n| duration of intensive unit stay | duration of intensive unit stay | Duración de la estancia en UCI |\n| OUTCOME       | OUTCOME                                   | Resultado del paciente (alta, fallecimiento, etc.) |\n| SMOKING       | SMOKING                                   | Historial de consumo de tabaco |\n| ALCOHOL       | ALCOHOL                                   | Historial de consumo de alcohol |\n| DM            | Diabetes Mellitus                         | Diagnóstico de diabetes mellitus |\n| HTN           | Hypertension                              | Diagnóstico de hipertensión arterial |\n| CAD           | Coronary Artery Disease                   | Diagnóstico de enfermedad coronaria |\n| PRIOR CMP     | CARDIOMYOPATHY                            | Historial de miocardiopatía |\n| CKD           | CHRONIC KIDNEY DISEASE                    | Diagnóstico de enfermedad renal crónica |\n| HB            | Haemoglobin                               | Nivel de hemoglobina en sangre |\n| TLC           | TOTAL LEUKOCYTES COUNT                    | Conteo total de leucocitos |\n| PLATELETS     | PLATELETS                                 | Conteo de plaquetas |\n| GLUCOSE       | GLUCOSE                                   | Nivel de glucosa en sangre |\n| UREA          | UREA                                      | Nivel de urea en sangre |\n| CREATININE    | CREATININE                                | Nivel de creatinina en sangre |\n| BNP           | B-TYPE NATRIURETIC PEPTIDE                | Péptido relacionado con función cardíaca |\n| RAISED CARDIAC ENZYMES | RAISED CARDIAC ENZYMES           | Presencia de enzimas cardíacas elevadas |\n| EF            | Ejection Fraction                         | Fracción de eyección cardíaca |\n| SEVERE ANAEMIA| SEVERE ANAEMIA                            | Presencia de anemia grave |\n| ANAEMIA       | ANAEMIA                                   | Presencia de anemia |\n| STABLE ANGINA | STABLE ANGINA                             | Dolor torácico estable por angina |\n| ACS           | Acute coronary Syndrome                   | Síndrome coronario agudo |\n| STEMI         | ST ELEVATION MYOCARDIAL INFARCTION        | Infarto agudo de miocardio con elevación del ST |\n| ATYPICAL CHEST PAIN | ATYPICAL CHEST PAIN                 | Dolor torácico no típico |\n| HEART FAILURE | HEART FAILURE                             | Diagnóstico de insuficiencia cardíaca |\n| HFREF         | HEART FAILURE WITH REDUCED EJECTION FRACTION | Insuficiencia cardíaca con fracción de eyección reducida |\n| HFNEF         | HEART FAILURE WITH NORMAL EJECTION FRACTION | Insuficiencia cardíaca con fracción de eyección conservada |\n| VALVULAR      | Valvular Heart Disease                    | Enfermedad de válvulas cardíacas |\n| CHB           | Complete Heart Block                      | Bloqueo cardíaco completo |\n| SSS           | Sick sinus syndrome                       | Síndrome de disfunción sinusal |\n| AKI           | ACUTE KIDNEY INJURY                       | Lesión renal aguda |\n| CVA INFRACT   | Cerebrovascular Accident INFRACT          | Accidente cerebrovascular isquémico |\n| CVA BLEED     | Cerebrovascular Accident BLEED            | Accidente cerebrovascular hemorrágico |\n| AF            | Atrial Fibrilation                        | Fibrilación auricular |\n| VT            | Ventricular Tachycardia                   | Taquicardia ventricular |\n| PSVT          | PAROXYSMAL SUPRA VENTRICULAR TACHYCARDIA  | Taquicardia supraventricular paroxística |\n| CONGENITAL    | Congenital Heart Disease                  | Enfermedad cardíaca congénita |\n| UTI           | Urinary tract infection                   | Infección de vías urinarias |\n| NEURO CARDIOGENIC SYNCOPE | NEURO CARDIOGENIC SYNCOPE     | Síncope de origen cardiogénico |\n| ORTHOSTATIC   | ORTHOSTATIC                               | Hipotensión postural |\n| INFECTIVE ENDOCARDITIS | INFECTIVE ENDOCARDITIS           | Inflamación de las válvulas cardíacas por infección |\n| DVT           | Deep venous thrombosis                    | Trombosis venosa profunda |\n| CARDIOGENIC SHOCK | CARDIOGENIC SHOCK                     | Shock de origen cardíaco |\n| SHOCK         | SHOCK                                     | Shock por otras causas |\n| PULMONARY EMBOLISM | PULMONARY EMBOLISM                   | Bloqueo de arterias pulmonares por coágulo |\n| CHEST INFECTION | CHEST INFECTION                         | Infección pulmonar |\n| DAMA          | Discharged Against Medical Advice         | Alta médica solicitada por el paciente en contra de la recomendación |",
  "## 1 Cargar base de datos",
  "Se decide eliminar la variabla BNP dado que tiene más del 50% de valores faltantes.",
  "## 2. Tratamiento de la base de datos",
  "Teniendo en cuenta que la variable que se refiere a duración en la unidad de cuidados intensivos contiene información que no se tiene cuando un paciente es ingresado al hospital, se decide eliminar con el objetivo de hacer un análisis más realista.",
  "### 2.1 Separación en variables categóricas y variables numéricas",
  "En las graficas anteriores se logra identificar que las variables numéricas representan una gran cantidad de valores atípicos. Teniendo en cuento esto, se decide calcular el porcentaje de datos atípicos que representa cada variable.",
  "# Análisis exploratorio de datos",
  "AGE (edad)\n\n-  La variable AGE (edad) presenta una distribución aproximadamente normal con ligera asimetría hacia la derecha. La mayor parte de los registros se concentra entre los 50 y 70 años, lo que refleja que la población del dataset corresponde principalmente a adultos de mediana y mayor edad.\n\nHB (hemoglobina)\n\n-  La variable HB (hemoglobina) muestra una distribución bastante simétrica, con la mayor densidad de valores entre 12 y 14 g/dL. Los valores extremos por debajo de 8 g/dL o por encima de 18 g/dL son poco frecuentes, lo que indica que la mayoría de los registros se ubica en un rango considerado habitual.\n\nTLC (total leucocyte count)\n\n-  La variable TLC presenta una distribución altamente asimétrica a la derecha. La mayoría de los valores se concentra en rangos bajos, mientras que existe un número reducido de observaciones con valores muy elevados, que generan una cola larga en la distribución.\n\nPLATELETS (plaquetas)\n\n-  La variable PLATELETS tiene una distribución sesgada positivamente. La mayor concentración se encuentra entre 200,000 y 300,000, aunque se observan registros con valores más altos que extienden la cola de la distribución.\n\nGLUCOSE (glucosa)\n\n-  La variable GLUCOSE muestra una distribución asimétrica hacia la derecha, con un pico en los valores bajos y una dispersión amplia que incluye observaciones por encima de 400. Esto evidencia la presencia de valores extremos elevados en el dataset.\n\nUREA\n\n-  La variable UREA presenta una fuerte asimetría positiva. La mayoría de los valores se concentra por debajo de 100, aunque se registran observaciones con valores mucho más altos, que extienden la distribución hacia la derecha.\n\nCREATININE (creatinina)\n\n-  La variable CREATININE también exhibe una asimetría positiva pronunciada. La mayor parte de los registros se concentra en valores bajos, mientras que existen observaciones dispersas con valores más altos que alargan la cola de la distribución.\n\nEF (ejection fraction)\n\n-  La variable EF (fracción de eyección) muestra un patrón particular: existe una concentración importante de registros en el valor 60, mientras que el resto de la distribución se reparte entre valores de 20 a 40. Esto genera una distribución no simétrica con un pico muy marcado en el límite superior.",
  "## Relaciones bivariadas\n\n### Edad vs otras variables\n\n-  No se observan tendencias lineales marcadas entre AGE y las demás variables.\n\n-  Los puntos están bastante dispersos en ambos géneros.\n\n### HB vs otras variables\n\n-  Ligera correlación negativa con Urea y Creatinine (a medida que aumentan, la hemoglobina tiende a ser más baja).\n\n-  Diferencia por género: los hombres concentran valores algo más altos de HB en todos los rangos.\n\n### TLC vs otras variables\n\n-  TLC presenta gran dispersión, con muchos valores extremos, pero no muestra relación clara con otras variables.\n\n-  La distribución por género es muy similar.\n\n### Plaquetas (PLATELETS)\n\n-  No se aprecian correlaciones fuertes con otras variables.\n\n-  La dispersión es amplia y comparable entre hombres y mujeres.\n\n### Glucose vs Urea/Creatinine\n\n-  No hay una correlación directa clara, aunque algunos casos con glucosa muy alta también muestran valores elevados de urea o creatinina.\n\n-  Ambos géneros siguen el mismo patrón.\n\n### Urea y Creatinine\n\n-  Relación positiva clara: a mayor creatinina, mayor urea.\n\n-  Ambos géneros siguen exactamente la misma tendencia.\n\n### EF (fracción de eyección)\n\n-  Se nota la concentración en el valor 60.\n\n-  No hay una diferencia visible entre géneros en este patrón.\n\n-  Relación inversa tenue con urea/creatinina: pacientes con valores altos de estos parámetros tienden a mostrar EF más baja.",
  "### ¿Cuál sexo presenta mayor cantidad de hospitalizaciones?",
  "Con base en el gráfico anterior, es posible observar que la mayor cantidad de pacientes son de género masculino.",
  "### ¿Cómo se ve afectada la cantidad de hospitalizaciones por la edad de los pacientes?",
  "La distribución de hospitalizaciones presenta un patrón claro y esperado.\n\nPico de hospitalizaciones: El rango de edad con mayor número de hospitalizaciones se encuentra entre los 55 y 63 años, seguido de cerca por el grupo de 63 a 68 años. Este hallazgo es coherente con el aumento de la prevalencia de enfermedades crónicas, como la hipertensión, la diabetes y las enfermedades cardiovasculares, a medida que las personas envejecen. Además, la acumulación de factores de riesgo a lo largo de la vida contribuye a una mayor necesidad de atención médica y hospitalización en esta etapa.\n\nAsimetría negativa: El gráfico presenta asimetría negativa, lo que indica que, aunque hay hospitalizaciones en todas las edades, la mayor concentración de eventos ocurre en los grupos de mayor edad. La razón es que las personas mayores suelen tener sistemas inmunológicos más débiles y múltiples comorbilidades (varias enfermedades al mismo tiempo), lo que aumenta su vulnerabilidad a infecciones y complicaciones de salud.\n\nMenor frecuencia de hospitalizaciones: El rango de edad con la menor cantidad de hospitalizaciones es el de 0 a 20 años. Esto se debe a que, en general, los niños y adultos jóvenes tienen un sistema inmune robusto y una menor incidencia de enfermedades crónicas graves. Las hospitalizaciones en este grupo suelen estar asociadas a accidentes, infecciones agudas o condiciones congénitas, que son menos frecuentes en comparación con las enfermedades degenerativas que afectan a la población de mayor edad.\n\nEn resumen, el gráfico refleja una clara correlación entre el envejecimiento y la probabilidad de ser hospitalizado debido a la acumulación de riesgos y el deterioro natural del cuerpo.",
  "## ¿Existe relación entre la edad del paciente y la cantidad de días de hospitalización?",
  "El gráfico muestra una correlación de 0.1064 entre la edad y los días de permanencia en el hospital. Esto se interpreta como una correlación positiva, pero muy débil. Esto significa que, aunque a medida que la edad aumenta, los días de hospitalización también tienden a aumentar ligeramente, la relación no es lo suficientemente fuerte como para ser considerada significativa. La gran dispersión de los puntos en el gráfico, que no forman una línea clara, confirma esta débil relación. Los días de hospitalización de una persona no pueden predecirse con fiabilidad basándose únicamente en su edad, ya que otros factores —como la severidad de la enfermedad, la presencia de comorbilidades y el tipo de tratamiento— pueden tener un impacto mayor.",
  "## 3. Dividir conjunto de entrenamiento y prueba",
  "La variable elegida como objetivo es de tipo numérico continuo y representa el número de días, o fracción de días, que un paciente permanecerá en el hospital. Su predicción tiene un alto valor clínico y operativo, ya que permite planificar con mayor precisión los recursos, la disponibilidad de camas y la asignación de personal. Además, esta duración está influenciada por múltiples factores presentes en el conjunto de datos, como diagnósticos, comorbilidades y resultados de laboratorio.",
  "### 2.2 Preprocesamiento",
  "## 4. Seleccion de características",
  "### PCA Y MCA",
  "En este gráfico se observa que para las variables numéricas solamente se logran construir tres componentes principales, las cuales alcanzan a explicar un 74% de la varianza total de las variables, siendo la primera de ellas la que más porcentaje de varianza explica con un 47% de varianza aproximadamente.",
  "Es posible observar que la primera componente explica un 43.31% de la varianza, mientras que la segunda componente explica un 16.28% de esta. También, la agrupación de puntos cerca del punto (0,0) indica que la mayoría de datos tienen valores bajos en ambas componentes, por otro lado, aquellos puntos que se alejan del origen, pueden representar valores atípicos y que son muy diferentes al resto de las observaciones. Finalmente, se evidencia que hay una mayor dispersión de los puntos en la componente 1 que en la componente 2, lo que podría indicar que hay una gran variación en las variables que contribuyen a la componente 1, mientras que la variación en la segunda componente es menor.",
  "PCA1: Esta componente está fuertemente influenciada por CREATININE (0.81) y UREA (0.54). Ambas son marcadores importantes de la función renal. Por lo tanto, podemos interpretar a la PCA1 como una componente relacionada con la función renal.\n\nPCA2: La variable que domina esta componente es TLC (0.96), que es el recuento total de leucocitos. Por lo tanto, la PCA2 probablemente representa información relacionada con la respuesta inmune o el estado infeccioso del paciente.\n\nPCA3: Esta componente tiene una fuerte correlación positiva con GLUCOSE (0.92). Esto sugiere que la PCA3 está relacionada con los niveles de glucosa del paciente. También hay una contribución moderada de AGE (0.23).",
  "Dimensión 1 (Componente 0): Esta dimensión está fuertemente definida por OUTCOME_DAMA_1.0 (valor alto, rojo) y SHOCK_0 (valor alto, rojo), lo que sugiere que esta dimensión podría estar relacionada con el desenlace del paciente. Por otro lado, OUTCOME_DISCHARGE_1.0, DVT_1.0 y PULMONARY EMBOLISM_1.0 tienen contribuciones negativas (azules).\n\nDimensión 2 (Componente 1): Esta componente está fuertemente influenciada por ALCOHOL_1.0, HTN_1.0, CAD_1.0, y CKD_0.0, lo que podría indicar una dimensión relacionada con factores de riesgo y comorbilidades preexistentes.\n\nDimensión 3 (Componente 2): Esta dimensión parece estar relacionada con afecciones más graves o agudas, ya que STEMI_1.0 y INFECTION_ENDOCARDITIS_0.0 tienen contribuciones positivas (rojas), mientras que CVA_INFARCT_1.0 contribuye de manera negativa (azul).\n\nDimensión 4 y 5 (Componentes 3 y 4): A partir de la Dimensión 4, las contribuciones se vuelven más difusas. Sin embargo, en la Dimensión 5, se observa una fuerte contribución de INFECTIVE_ENDOCARDITIS_1.0 y PULMONARY_EMBOLISM_1.0.",
  "En contraste, el Análisis de Correspondencias Múltiples (MCA), aplicado a las variables categóricas, no logró una reducción de dimensionalidad significativa, ya que las cinco primeras componentes solo explicaron el 25% de la varianza. Esto sugiere que las relaciones entre las categorías son complejas y difusas, lo cual es una información valiosa por sí misma sobre la estructura de los datos.",
  "\nEl scatterplot de las dos primeras componentes principales del Análisis de Correspondencia Múltiple (MCA) muestra que las variables categóricas se han transformado en un nuevo espacio de dos dimensiones. Sin embargo, la varianza explicada por estas dos componentes es bastante baja, con la Componente 1 explicando el 8.06% de la varianza total y la Componente 2 el 4.89%. Esta baja inercia total explicada indica que estas dos dimensiones por sí solas no capturan una parte significativa de la información contenida en las variables categóricas originales. Además, la distribución de los puntos en el gráfico no revela una estructura de grupos o clústeres distintiva, lo que sugiere que no hay divisiones claras en el conjunto de datos a lo largo de estas dos primeras dimensiones.",
  "## Conclusiones\n\nEl proceso de pre-pruning se aplicó con el objetivo de optimizar el modelo de árbol de decisión antes de su entrenamiento completo, evitando la creación de un árbol excesivamente complejo. La búsqueda de los mejores hiperparámetros identificó una configuración óptima que utiliza una profundidad máxima de 3 y requiere que cada hoja contenga al menos 7 muestras.\n\nSin embargo, el análisis de la curva de aprendizaje reveló que el modelo tiene un claro sobreajuste.  El error de entrenamiento (cercano a 0) es significativamente más bajo que el error de validación (alrededor de 4.75), una brecha que no se reduce a medida que aumenta el tamaño del conjunto de entrenamiento.\n\nA pesar de que el RMSE en el conjunto de prueba (4.6892) es consistente con el RMSE de validación cruzada (4.7832), la gran diferencia con el error de entrenamiento indica que el modelo no está generalizando bien. Las restricciones aplicadas por el pre-pruning no fueron suficientes para mitigar por completo el sobreajuste. Para mejorar el rendimiento, sería necesario explorar otras estrategias de regularización o hiperparámetros, o considerar un modelo más robusto, como Random Forest.",
  "El modelo logró encontrar el ccp_alpha óptimo de 0.034431, lo que resultó en un árbol con 70 hojas y 139 nodos, un tamaño significativamente menor que un árbol sin podar. El RMSE de validación cruzada fue de 4.821. Aunque este valor es similar al de otros modelos, el gráfico de la curva de aprendizaje revela la verdadera naturaleza de su bajo rendimiento.\nLa curva de aprendizaje para el árbol podado muestra una gran diferencia entre el error de entrenamiento y el error de validación. El error de entrenamiento comienza bajo pero aumenta a medida que se agregan más datos, mientras que el error de validación comienza alto y se estabiliza. Las dos curvas nunca convergen, lo que es un claro signo de subajuste, ya que el modelo no puede aprender adecuadamente las relaciones en los datos, incluso con un conjunto de entrenamiento más grande.",
  "## Conclusiones\n\nEl mejor modelo KNN, aunque ha sido optimizado con RandomizedSearchCV, sigue mostrando un comportamiento de sobreajuste significativo. La capacidad del modelo para memorizar los datos de entrenamiento contrasta fuertemente con su rendimiento en datos no vistos. Esto sugiere que:\n\n*   Aún hay complejidad excesiva: A pesar de tener 49 vecinos, el modelo sigue siendo demasiado \"detallista\" para los datos disponibles.\n\n*   Los datos pueden ser ruidosos: Podría haber mucho ruido o variabilidad en los datos que el modelo está intentando aprender.\n\nAl explorar con los diferentes algoritmos para KNN no se evidencia un cambio en las medidas de calidad.",
  "## Conclusiones\n\nEl modelo de regresión lineal presenta un rendimiento modesto y se encuentra en una situación de subajuste (underfitting). Aunque el error de entrenamiento y el de validación están muy cerca el uno del otro, un signo de consistencia, su valor absoluto (alrededor de 4.8 RMSE) es bastante alto. Esto indica que el modelo es demasiado simple para capturar la complejidad de los datos. La baja puntuación de $R^2$ de 0.103 confirma que el modelo explica solo el 10.3% de la varianza en los datos, lo que significa que la gran mayoría de la variabilidad en los datos no se ha capturado. Por lo tanto, el modelo tiene un poder predictivo muy limitado.",
  "## Conclusiones\n\nEl RMSE de 4.690 y el $R^2$ de 0.104 en el conjunto de prueba indican que el modelo tiene un poder predictivo muy limitado. El #R^2# en particular sugiere que el modelo solo es capaz de explicar el 10.4% de la varianza en la variable objetivo, lo cual es un indicador de que las predicciones no son muy fiables.\n\nLa curva de aprendizaje del modelo Lasso confirma este diagnóstico. Aunque las curvas de error de entrenamiento y de validación están muy cerca, lo que sugiere que no hay sobreajuste, ambas se mantienen en un valor alto (alrededor de 4.8 RMSE). Esto es un signo de subajuste, donde el modelo es demasiado simple para capturar los patrones subyacentes de los datos.\n\nEn este caso, la regularización de Lasso no está aportando un beneficio significativo. Para mejorar el rendimiento, sería recomendable explorar modelos más complejos y no lineales, como el Random Forest, que podrían ser más adecuados para capturar las relaciones complejas de los datos.",
  "## Conclusiones\n\nEl modelo Ridge muestra un comportamiento de subajuste (underfitting), con un rendimiento limitado. El valor de RMSE de 4.690 y un $R^2$ de 0.104 en el conjunto de prueba indican que el modelo es débil, ya que solo explica un 10.4% de la varianza en la variable objetivo.",
  "## Conclusiones\n\nEl modelo logra un RMSE de 4.632 en el conjunto de prueba, una ligera mejora respecto a los modelos lineales anteriores. El valor de $R^2$ de 0.125 también indica que, a diferencia de los modelos anteriores, este modelo es capaz de explicar un poco más de la varianza en los datos (12.5% para ser exactos). Sin embargo, la gran mayoría de la varianza sigue sin ser explicada.",
  "## Conclusiones\n\nEl modelo logra un RMSE de 4.631 y un $R^2$ de 0.126 en el conjunto de prueba. Si bien estos valores son ligeramente mejores que los de los modelos lineales y polinomiales sin regularización, siguen indicando que el modelo tiene un poder predictivo limitado. Un $R^2$ de 0.126 significa que el modelo solo puede explicar el 12.6% de la varianza en la variable objetivo, lo cual no es suficiente para hacer predicciones confiables.",
  "## Conclusiones\n\nEl modelo logra un RMSE de 4.713 en el conjunto de prueba, un valor similar al de los modelos lineales y que no representa una mejora significativa. El $R^2$ de 0.095 confirma este bajo rendimiento, indicando que el modelo solo es capaz de explicar el 9.5% de la varianza en la variable objetivo. El RMSE relativo de 0.741 sugiere que el error de predicción es muy alto en relación con los valores reales, lo que hace que las predicciones sean poco fiables.\n\nLa curva de aprendizaje del SVR  muestra una brecha constante y relativamente grande entre el error de entrenamiento (la línea azul) y el de validación (la línea naranja). Aunque el error de validación se mantiene relativamente estable, el error de entrenamiento sigue disminuyendo a medida que se añaden más datos. Esto, junto con el alto valor de ambos errores, es un signo de que el modelo no tiene la capacidad para capturar la complejidad de los datos, lo cual es la definición de subajuste.\n\n",
  "## Conclusiones\n\nEl modelo Random Forest, aun con hiperparámetros optimizados, muestra un poder explicativo bajo (R² ≈ 0.205) y un error relativamente alto (RMSE relativo ≈ 0.694). Esto indica que los predictores disponibles no capturan suficientemente bien la variabilidad de la variable objetivo.",
  "## Conclusiones\nEl modelo logra un RMSE de 4.5371 en el conjunto de prueba, un valor que es una mejora sobre los modelos lineales y un poco mejor que Random Forest. El Mejor RMSE CV de 4.6201 es muy cercano al RMSE de prueba, lo que indica que el modelo es consistente y su rendimiento es una estimación confiable.\n\nLa curva de aprendizaje  es la evidencia principal del subajuste. La línea de error de entrenamiento y la de validación no convergen. El error de entrenamiento sigue aumentando a medida que aumenta el tamaño de la muestra, y el error de validación se mantiene relativamente plano. Esto, junto con el hecho de que ambos errores están en un nivel alto, es una señal de que el modelo no tiene la capacidad para capturar la complejidad de los datos, lo cual es la definición de subajuste.",
  "## Conclusiones finales (PCA y MCA)\n\n| Modelo                               | RMSE Test | RMSE Relativo | R² (Test) |\n| ------------------------------------ | --------- | ------------- | --------- |\n| **Árbol de Decisión (Pre-pruning)**  | 4.689     | 0.737         | 0.104     |\n| **Árbol de Decisión (Post-pruning)** | 4.821     | 0.758         | 0.053     |\n| **Regresión Lineal**                 | 4.690     | 0.738         | 0.103     |\n| **Lasso**                            | 4.690     | 0.739         | 0.104     |\n| **Ridge**                            | 4.690     | 0.738         | 0.104     |\n| **Ridge + Polinomial**               | 4.632     | 0.728         | 0.125     |\n| **ElasticNet + Polinomial**          | 4.631     | 0.728         | 0.126     |\n| **SVR**                              | 4.731     | 0.741         | 0.095     |\n| **KNN**                              | 4.415     | 0.694         | 0.205     |\n| **Random Forest**                    | 4.415     | 0.694         | 0.205     |\n| **Gradient Boosting**                | 4.537     | 0.714         | 0.160     |\n\nEl KNN y el Random Forest obtuvieron el mismo RMSE en el conjunto de prueba (4.415), el valor más bajo de todos los modelos. Además, ambos lograron un $R^2$ de 0.205, lo que significa que explican el 20.5% de la varianza de los datos. Esta es la mejor puntuación entre todos los modelos. Por el contrario, los modelos lineales, como la Regresión Lineal, Lasso y Ridge, tienen un rendimiento pobre, con un RMSE de 4.690 y un $R^2$ de apenas 0.104.\n",
  "## Selección por filtrado",
  "#### Correlación de Spearman",
  "Edad (0.1289): El valor de la correlación para la edad es 0.1289, lo que se considera una correlación muy débil o nula. Aunque es una correlación positiva, esto sugiere que las personas mayores tienden a durar más tiempo en el hospital, aunque el efecto no es significativo. Esto deja en evidencia que la edad no es un predictor fuerte del tiempo de estancia en el hospital.\n\nUrea (0.2288) y Creatinina (0.1854): Ambos valores indican la función renal. En estos se muestra que hay una correlación débil entre el tiempo de permanencia de hospitalización y estas variables. Esto sugiere que, si bien la insuficiencia renal puede ser un factor que prolongue la estancia, no es el factor principal.\n\nHb (Hemoglobina) (-0.1747): La correlación negativa y muy débil sugiere que a niveles ligeramente más bajos de hemoglobina, la estancia puede ser un poco más larga, pero la relación es muy baja.\n\nInfección de pecho (0.1969) y Plaquetas (0.0103): Estos valores extremadamente cercanos a cero indican que no existe una correlación significativa entre estas variables y los días de hospitalización.",
  "Es posible observar que una de las variables que más aporta a la predicción del tiempo de estancia en el hospital es UREA, que alcanza una importancia del 18% aproximadamente. Posteriormente, variables como TLC, CREATININE, HB, EF y la edad alcanzan un 90% del porcentaje acumulado.",
  "#### ANOVA",
  "Para las variables categóricas se realizó una prueba ANOVA, con el objetivo de encontrar aquellas variables que están significativamente relacionadas con las variable objetivo.\nHo: La variable no está relacionada con el Target.\nH1: La variable está relacionada con el Target.\nCon base en las hipótesis anteriores, las variables que se seleccionan son aquellas que rechazan la Hipótesis Nula, es decir, las que tienen un p-valor menor a 0.05. Con base en esto, las variables que se seleccionan son las siguientes:",
  "Una vez realizada la selección de las variables con los métodos manuales, se decide utilizar solamente el conjunto de variables que representan mayor importancia y/o correlación con el Target para aplicar los métodos automáticos.",
  "## Selección automática",
  "### SelectKBest",
  "Primero, se utilizó la correlación de Spearman para asignar un \"score\" a cada variable. Este score mide la fuerza de la relación de cada variable independiente con la variable objetivo (en este caso, los días de hospitalización).\n\nLas variables se ordenan de mayor a menor score. Luego, se calcula el porcentaje de importancia individual de cada una, dividiendo su score entre la suma total de los scores de todas las variables. Luego, se suman los porcentajes individuales de forma secuencial, empezando por la variable más importante.\n\nFinalmente, se seleccionan las variables hasta que la suma acumulada alcanza el umbral del 90%.",
  "### RFECV",
  "### Random Forest",
  "En primera instancia, se aplicaron métodos de filtrado utilizando la medida de correlación de Spearman y la prueba ANOVA, seleccionando en cada caso las variables con mayor relevancia estadística. Posteriormente, con el conjunto reducido obtenido de esta etapa, se implementaron tres métodos de selección de características: SelectKBest, RFE y Random Forest. Finalmente, se identificaron las variables comunes entre SelectKBest y RFE, las cuales fueron seleccionadas para conformar un nuevo dataframe con menor dimensionalidad, optimizando así la eficiencia del modelo sin comprometer su capacidad predictiva.",
  "## Arbol de decision con PRE-PRUNING",
  "## Conclusiones\nEl modelo de árbol de decisión con pre-pruning alcanzó un RMSE en test de 4.68, lo que equivale a un RMSE relativo de 0.737 respecto a la media del target. El coeficiente de determinación fue R² = 0.106, lo que indica que el modelo explica alrededor del 10% de la variabilidad de los datos. Si bien el error de validación muestra cierta estabilidad y mejora a medida que aumenta el tamaño del conjunto de entrenamiento, la capacidad predictiva global sigue siendo limitada. Esto sugiere que, aunque la poda previa evita el sobreajuste y produce un modelo más generalizable, la complejidad del árbol no es suficiente para capturar patrones más profundos en los datos.",
  "## Arbol de decisión con POST-PRUNING",
  "## Conclusiones\nEl modelo de árbol de decisión con post-pruning seleccionó un valor óptimo de ccp_alpha = 0.0387, con el cual redujo la complejidad del árbol a 50 hojas y 99 nodos. En validación cruzada alcanzó un RMSE promedio de 4.79, mientras que en el conjunto de prueba obtuvo un RMSE de 4.83, equivalente a un RMSE relativo de 0.760 respecto a la media de la variable objetivo. El coeficiente de determinación fue R² = 0.048, lo que indica que el modelo explica apenas un 4.8% de la variabilidad de los datos.\n\nLa curva de aprendizaje muestra que el error de validación se mantiene estable al aumentar el tamaño del conjunto de entrenamiento, lo que refleja un nivel limitado de sobreajuste, pero también evidencia que el modelo carece de capacidad para capturar patrones más profundos en los datos.",
  "## K NeigborsRegressor",
  "## Conclusiones\n\nEl modelo KNN (k=44, métrica Minkowski con p=1, weights='distance') alcanzó un RMSE en validación cruzada de 4.428, mientras que en el conjunto de prueba obtuvo un RMSE de 4.346, equivalente a un R² = 0.230, lo que significa que explica aproximadamente el 23% de la variabilidad de los datos.\n\nLa curva de aprendizaje muestra que el error de entrenamiento se mantiene muy bajo (casi nulo), mientras que el error de validación es significativamente más alto y estable. Esta gran brecha entre ambas curvas refleja un claro sobreajuste, donde el modelo se adapta demasiado a los datos de entrenamiento pero no logra generalizar con la misma eficacia en el conjunto de validación.",
  "## Regresion lineal",
  "## Conclusiones\nEl modelo de regresión lineal alcanzó un RMSE en test de 4.591, equivalente a un RMSE relativo de 0.722 respecto a la media de la variable objetivo. El coeficiente de determinación fue R² = 0.141, lo que significa que el modelo explica aproximadamente el 14% de la variabilidad de los datos.\n\nLa curva de aprendizaje muestra que los errores de entrenamiento y validación son cercanos y se mantienen relativamente estables conforme aumenta el tamaño del conjunto de entrenamiento. Esto indica que el modelo no sufre de sobreajuste, pero sí de subajuste, ya que su capacidad explicativa es limitada y no logra capturar de manera adecuada la complejidad de la relación entre las variables.",
  "## Regresión Lasso",
  "## Conclusiones\n\nEl modelo Lasso con α=0.0001 mostró un desempeño muy similar al de la regresión lineal, con un RMSE en test de 4.591, equivalente a un RMSE relativo de 0.722, y un R² de 0.141, lo que significa que apenas explica un 14% de la variabilidad de los datos. La curva de aprendizaje refleja que los errores de entrenamiento y validación son cercanos y estables, lo que indica que el modelo no sufre de sobreajuste, pero sí de subajuste, ya que su capacidad explicativa es limitada. En conclusión, aunque Lasso controla la complejidad y ofrece estabilidad, no logra capturar patrones relevantes en los datos",
  "## Regresión Ridge",
  "## Conclusiones\nEl modelo Ridge, con un α óptimo de aproximadamente 2.78 según validación cruzada, alcanzó un RMSE en test de 4.591, equivalente a un RMSE relativo de 0.722, y un R² de 0.141, lo que indica que explica solo el 14% de la variabilidad de los datos. La curva de aprendizaje muestra que los errores de entrenamiento y validación son cercanos y tienden a estabilizarse con el aumento del tamaño de los datos, lo cual refleja que el modelo no sufre de sobreajuste, pero sí de subajuste, ya que su capacidad predictiva es limitada. En conclusión, aunque la regularización Ridge aporta estabilidad al modelo y evita la complejidad excesiva, su poder explicativo sigue siendo bajo",
  "## Regresión Ridge + Polinomica",
  "## Conclusiones\nEl modelo Ridge con características polinomiales alcanzó un RMSE en test de 4.580, con un RMSE relativo de 0.720, y un R² de 0.145, lo que indica que explica aproximadamente el 14.5% de la variabilidad de la variable objetivo. La curva de aprendizaje muestra que tanto el error de entrenamiento como el de validación tienden a estabilizarse a medida que aumenta el tamaño del conjunto de entrenamiento, con valores relativamente cercanos entre sí, lo que sugiere que el modelo generaliza de manera consistente y no presenta sobreajuste. Sin embargo, el nivel de error se mantiene elevado y el poder explicativo continúa siendo bajo, lo que evidencia subajuste: aun incorporando términos polinomiales, la capacidad predictiva del modelo sigue siendo limitada. En conclusión, aunque Ridge con expansión polinomial aporta una ligera mejora respecto a la regresión lineal simple, su desempeño aún es modesto",
  "## ElasticNet",
  "## Conclusiones\nEl modelo ElasticNet con características polinomiales, utilizando un alpha óptimo de 0.0046 según validación cruzada, obtuvo un RMSE en test de 4.576, equivalente a un RMSE relativo de 0.720, y un R² de 0.147, lo que significa que explica alrededor del 14.7% de la variabilidad de la variable objetivo. La curva de aprendizaje muestra una clara reducción del error de validación al aumentar el tamaño del conjunto de entrenamiento, hasta estabilizarse en valores similares al error de entrenamiento, lo que indica que el modelo generaliza de manera adecuada y no presenta sobreajuste. Sin embargo, los errores se mantienen relativamente altos y el poder explicativo es limitado, reflejando cierto subajuste, ya que aun con la regularización mixta de ElasticNet y la expansión polinomial, el modelo no logra capturar toda la complejidad de los datos",
  "## SVR",
  "## Conclusiones\nEl modelo SVR, con un parámetro óptimo de C = 1.93 según validación cruzada, alcanzó un RMSE en test de 4.602, equivalente a un RMSE relativo de 0.724, y un R² de 0.137, lo que indica que explica únicamente el 13.7% de la variabilidad de la variable objetivo. La curva de aprendizaje muestra que los errores de entrenamiento y validación se mantienen relativamente cercanos y estables conforme aumenta el tamaño del conjunto de entrenamiento, lo que refleja que el modelo no incurre en sobreajuste; sin embargo, los valores de error siguen siendo elevados y la capacidad explicativa es limitada, señal clara de subajuste. En conclusión, aunque SVR con el mejor C ofrece estabilidad y generalización aceptable, su poder predictivo es bajo y no logra capturar de forma adecuada la complejidad de los datos\n",
  "## Random Forest",
  "## Conclusiones\nEl modelo Random Forest con los hiperparámetros óptimos (max_depth=20, max_features='sqrt', n_estimators=200) alcanzó su mejor desempeño con un RMSE en test de 4.312, equivalente a un RMSE relativo de 0.680, y un R² de 0.137, lo que indica que, aunque reduce el error absoluto respecto a otros modelos y logra el valor más bajo de RMSE hasta ahora, su capacidad explicativa sigue siendo limitada al explicar apenas un 13.7% de la variabilidad de los datos. La curva de aprendizaje se muestra más irregular, lo cual es normal en Random Forest por la aleatoriedad de sus submuestras y árboles, pero refleja una tendencia estable en validación, confirmando que el modelo generaliza bien aunque no logra capturar toda la complejidad de los datos.",
  "## Gradient Boosting",
  "## Conclusiones\nEl modelo de Gradient Boosting con hiperparámetros óptimos (subsample=0.8, n_estimators=400, max_features='sqrt', max_depth=6, learning_rate=0.01) alcanzó un RMSE en test de 4.454, equivalente a un RMSE relativo de 0.700, y un R² de 0.137. La curva de aprendizaje muestra un error de entrenamiento claramente menor que el de validación, aunque ambos tienden a estabilizarse conforme aumenta el tamaño del conjunto de entrenamiento, lo cual refleja que el modelo logra generalizar de manera razonable y no está sobreajustado. Sin embargo, la brecha entre ambos errores y el bajo R² indican que, aunque el modelo reduce el RMSE respecto a otros enfoques más simples, la capacidad explicativa sigue siendo limitada, probablemente por la naturaleza y calidad de los datos disponibles más que por el algoritmo en sí.",
  "## Conclusiones finales (selector features)\n\n| Modelo                               | RMSE Test | RMSE Relativo | R² (Test) |\n| ------------------------------------ | --------- | ------------- | --------- |\n| **Árbol de Decisión (Pre-pruning)**  | 4.684     | 0.737         | 0.106     |\n| **Árbol de Decisión (Post-pruning)** | 4.834     | 0.760         | 0.048     |\n| **Regresión Lineal**                 | 4.591     | 0.722         | 0.141     |\n| **Lasso**                            | 4.591     | 0.722         | 0.141     |\n| **Ridge**                            | 4.591     | 0.722         | 0.141     |\n| **Ridge + Polinomial**               | 4.580     | 0.720         | 0.145     |\n| **ElasticNet + Polinomial**          | 4.576     | 0.720         | 0.147     |\n| **SVR**                              | 4.602     | 0.724         | 0.137     |\n| **KNN**                              | 4.346     | 0.683         | 0.230     |\n| **Random Forest**                    | **4.312** | **0.680**     | 0.137     |\n| **Gradient Boosting**                | 4.454     | 0.700         | 0.137     |\n",
  "Al revisar todos los modelos probados —incluyendo lineales (Lasso, Ridge), variantes polinomiales, SVR, KNN, árboles de decisión con pre y post-pruning, Random Forest y Gradient Boosting— se observa que los lineales y sus extensiones se mantuvieron en un rendimiento estable con RMSE cercano a 4.58–4.60, R² entre 0.14–0.15 y RMSE relativo de 0.72, mientras que los árboles de decisión obtuvieron un desempeño más bajo (RMSE entre 4.68–4.83 y R² de 0.05–0.10). El modelo KNN destacó con un RMSE de 4.35, R² de 0.23 y RMSE relativo de 0.68, mostrando mejor capacidad explicativa, mientras que SVR y ElasticNet polinomial ofrecieron métricas similares a las lineales. Por su parte, el Random Forest alcanzó el mejor resultado absoluto en error con un RMSE de 4.31, RMSE relativo de 0.68 y R² de 0.137, y el Gradient Boosting, con hiperparámetros ajustados, logró un RMSE CV de 4.52, RMSE Test de 4.45, R² de 0.137 y RMSE relativo de 0.70, consolidándose ambos como los más consistentes, aunque el Random Forest fue el que presentó el menor error global."
]

def render_notebook_markdown():
    if MARKDOWN_BLOCKS:
        for md in MARKDOWN_BLOCKS:
            md_container.markdown(md)
    else:
        md_container.info("No hay celdas Markdown en el notebook.")
render_notebook_markdown()

# ---------- App UI ----------
st.title("Base de datos")

# Persistencia
if "df" not in st.session_state:
    st.session_state.df = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None

# Barra lateral de navegación
st.sidebar.header("⚙️ Controles")
app_mode = st.sidebar.radio(
    "Secciones",
    ["1) Cargar base de datos", "2) Filtros (GENDER / RURAL)", "3) Ejecutar Notebook completo"],
    index=0
)

# === 1) Cargar base de datos ===
if app_mode == "1) Cargar base de datos":
    st.header("1. Cargar base de datos")
    try:
        bd = load_csv_robust()
        st.session_state.df = bd.copy()
        st.success("Datos cargados desde archivo en la raíz del repo.")
        st.write("Vista previa:")
        st.dataframe(bd.head(), use_container_width=True)

        st.subheader("Información del DataFrame")
        info_text = capture_text(bd.info)
        st.code(info_text, language="text")
    except Exception as e:
        st.error(str(e))
        st.stop()

# === 2) Filtros (GENDER / RURAL) ===
elif app_mode == "2) Filtros (GENDER / RURAL)":
    st.header("2. Filtros por GENDER y RURAL")
    ensure_df_loaded()
    df = st.session_state.df.copy()

    # Validación de columnas requeridas
    missing = [c for c in ["GENDER", "RURAL"] if c not in df.columns]
    if missing:
        st.error(f"Faltan las columnas requeridas en el dataset: {missing}. "
                 "Asegúrate de que existan con esos nombres exactos.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        gender_vals = sorted(df["GENDER"].dropna().unique().tolist(), key=lambda x: str(x))
        # Map values to user-friendly labels if they look like M/F or 0/1
        def pretty_gender(v):
            s = str(v).strip().lower()
            if s in ["m","male","1"]:
                return "Hombre"
            if s in ["f","female","0"]:
                return "Mujer"
            return str(v)
        pretty_map_g = {v: pretty_gender(v) for v in gender_vals}
        sel_gender_labels = st.multiselect(
            "Filtrar por GENDER",
            options=[pretty_map_g[v] for v in gender_vals],
            default=[pretty_map_g[v] for v in gender_vals]
        )
        # inverse map
        sel_gender = [k for k,v in pretty_map_g.items() if v in sel_gender_labels]

    with col2:
        rural_vals = sorted(df["RURAL"].dropna().unique().tolist(), key=lambda x: str(x))
        def pretty_area(v):
            s = str(v).strip().lower()
            if s in ["1","urban","urbano","u"]:
                return "Urbano"
            if s in ["0","rural","r"]:
                return "Rural"
            return str(v)
        pretty_map_r = {v: pretty_area(v) for v in rural_vals}
        sel_rural_labels = st.multiselect(
            "Filtrar por RURAL (Urbano/Rural)",
            options=[pretty_map_r[v] for v in rural_vals],
            default=[pretty_map_r[v] for v in rural_vals]
        )
        sel_rural = [k for k,v in pretty_map_r.items() if v in sel_rural_labels]

    filtered = df[df["GENDER"].isin(sel_gender) & df["RURAL"].isin(sel_rural)].copy()
    st.session_state.filtered_df = filtered

    st.markdown("### Resultado del filtrado")
    m1, m2, m3 = st.columns(3)
    m1.metric("Filas totales", len(df))
    m2.metric("Filas filtradas", len(filtered))
    m3.metric("Columnas", filtered.shape[1])
    st.dataframe(filtered.head(50), use_container_width=True)

    st.subheader("Distribución por GENDER y RURAL (filtrado)")
    grp = filtered.groupby(["GENDER", "RURAL"]).size().reset_index(name="count")
    if len(grp) > 0:
        fig = px.bar(grp, x="GENDER", y="count", color="RURAL", barmode="group",
                     title="Conteo por GENDER y RURAL")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay filas bajo los filtros actuales. Ajusta las selecciones.")

# === 3) Ejecutar Notebook completo ===
elif app_mode == "3) Ejecutar Notebook completo":
    st.header("3. Ejecución completa del Notebook (celdas de código)")
    ensure_df_loaded()
    # Base de datos a usar: filtrada si existe, si no, original
    df_base = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.df
    st.caption("Las celdas del notebook se ejecutan sobre el DataFrame actual (filtrado si aplicaste filtros).")
    show_code = st.checkbox("Mostrar código de cada celda", value=False)

    # Namespace compartido para todas las celdas
    ns = {
        "np": np, "pd": pd, "plt": plt, "st": st,
        "stats": stats,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline, "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer, "StandardScaler": StandardScaler,
        "RFE": RFE, "SelectKBest": SelectKBest, "f_regression": f_regression,
        "LinearRegression": LinearRegression, "PCA": PCA,
        "RandomForestRegressor": RandomForestRegressor,
        # DataFrame base disponible bajo nombres comunes
        "df": df_base.copy(),
        "bd": df_base.copy(),
        "data": df_base.copy(),
        "dataset": df_base.copy(),
        "px": px,
        "io": io,
    }

    prev_figs = set(plt.get_fignums())

    st.markdown("#### Celda 1")
    if show_code:
        st.code("# Cargar librer\u00edas\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nfrom scipy.stats import spearmanr, stats\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.feature_selection import RFE\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.decomposition import PCA\nfrom sklearn.ensemble import RandomForestRegressor # Import RandomForestRegressor\n\nimport prince", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Cargar librerías
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt

            from scipy.stats import spearmanr, stats
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LinearRegression
            from sklearn.decomposition import PCA
            from sklearn.ensemble import RandomForestRegressor # Import RandomForestRegressor

            import prince
    except Exception as e:
        st.error(f"Error en la celda 1: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 2")
    if show_code:
        st.code("# Subir el token\nfrom google.colab import files\nfiles.upload()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Subir el token
            from google.colab import files
            files.upload()
    except Exception as e:
        st.error(f"Error en la celda 2: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 3")
    if show_code:
        st.code("", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            pass
    except Exception as e:
        st.error(f"Error en la celda 3: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 4")
    if show_code:
        st.code("", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            pass
    except Exception as e:
        st.error(f"Error en la celda 4: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 5")
    if show_code:
        st.code("", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            pass
    except Exception as e:
        st.error(f"Error en la celda 5: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 6")
    if show_code:
        st.code("bd = pd.read_csv('HDHI Admission data.csv')\nbd.head()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            bd = pd.read_csv('HDHI Admission data.csv')
            bd.head()
    except Exception as e:
        st.error(f"Error en la celda 6: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 7")
    if show_code:
        st.code("bd.info()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            bd.info()
    except Exception as e:
        st.error(f"Error en la celda 7: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 8")
    if show_code:
        st.code("bd.drop('BNP', axis=1, inplace=True)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            bd.drop('BNP', axis=1, inplace=True)
    except Exception as e:
        st.error(f"Error en la celda 8: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 9")
    if show_code:
        st.code("## Eliminar variables innecesarias\n\ndf = bd.drop(['SNO', 'MRD No.'], axis=1)\ndf.drop('month year', axis=1, inplace=True)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ## Eliminar variables innecesarias

            df = bd.drop(['SNO', 'MRD No.'], axis=1)
            df.drop('month year', axis=1, inplace=True)
    except Exception as e:
        st.error(f"Error en la celda 9: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 10")
    if show_code:
        st.code("## Transformar las variables de fecha a formate datetime\ndf['D.O.A'] = pd.to_datetime(df['D.O.A'], format='%m/%d/%Y', errors='coerce')\ndf['D.O.D'] = pd.to_datetime(df['D.O.D'], format='%m/%d/%Y', errors='coerce')", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ## Transformar las variables de fecha a formate datetime
            df['D.O.A'] = pd.to_datetime(df['D.O.A'], format='%m/%d/%Y', errors='coerce')
            df['D.O.D'] = pd.to_datetime(df['D.O.D'], format='%m/%d/%Y', errors='coerce')
    except Exception as e:
        st.error(f"Error en la celda 10: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 11")
    if show_code:
        st.code("# Tratamiento de aquellas variables que son num\u00e9ricas pero est\u00e1n como categ\u00f3ricas\n\ncols_to_clean = ['HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'EF']\n\nfor col in cols_to_clean:\n    df[col] = (\n        df[col]\n        .astype(str)                      # aseguramos que todo sea string\n        .str.strip()                       # quitamos espacios\n        .replace(['EMPTY', 'nan', 'NaN', 'None', ''], np.nan)  # reemplazamos valores no v\u00e1lidos\n        .str.replace(r'[<>]', '', regex=True)  # quitamos > y <\n        .str.replace(',', '.', regex=False)    # cambiamos coma decimal a punto\n    )\n# Convierte las variables anteriores a num\u00e9ricas\nfor col in cols_to_clean:\n    df[col] = pd.to_numeric(df[col], errors='coerce')", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Tratamiento de aquellas variables que son numéricas pero están como categóricas

            cols_to_clean = ['HB', 'TLC', 'PLATELETS', 'GLUCOSE', 'UREA', 'CREATININE', 'EF']

            for col in cols_to_clean:
                df[col] = (
                    df[col]
                    .astype(str)                      # aseguramos que todo sea string
                    .str.strip()                       # quitamos espacios
                    .replace(['EMPTY', 'nan', 'NaN', 'None', ''], np.nan)  # reemplazamos valores no válidos
                    .str.replace(r'[<>]', '', regex=True)  # quitamos > y <
                    .str.replace(',', '.', regex=False)    # cambiamos coma decimal a punto
                )
            # Convierte las variables anteriores a numéricas
            for col in cols_to_clean:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        st.error(f"Error en la celda 11: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 12")
    if show_code:
        st.code("# Transforma las variables categ\u00f3ricas a dummies\n\ndf['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})\ndf['RURAL'] = df['RURAL'].map({'R': 1, 'U': 0})\ndf['TYPE OF ADMISSION-EMERGENCY/OPD'] = df['TYPE OF ADMISSION-EMERGENCY/OPD'].map({'E': 1, 'O': 0})\ndf['CHEST INFECTION'] = df['CHEST INFECTION'].map({'1': 1, '0': 0})\ndf = pd.get_dummies(df, columns=['OUTCOME'], drop_first=False)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Transforma las variables categóricas a dummies

            df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
            df['RURAL'] = df['RURAL'].map({'R': 1, 'U': 0})
            df['TYPE OF ADMISSION-EMERGENCY/OPD'] = df['TYPE OF ADMISSION-EMERGENCY/OPD'].map({'E': 1, 'O': 0})
            df['CHEST INFECTION'] = df['CHEST INFECTION'].map({'1': 1, '0': 0})
            df = pd.get_dummies(df, columns=['OUTCOME'], drop_first=False)
    except Exception as e:
        st.error(f"Error en la celda 12: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 13")
    if show_code:
        st.code("# Convierte cualquier columna booleana a int (0 y 1)\n\nbool_cols = df.select_dtypes(include=bool).columns\ndf[bool_cols] = df[bool_cols].astype(int)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Convierte cualquier columna booleana a int (0 y 1)

            bool_cols = df.select_dtypes(include=bool).columns
            df[bool_cols] = df[bool_cols].astype(int)
    except Exception as e:
        st.error(f"Error en la celda 13: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 14")
    if show_code:
        st.code("df = df.drop('duration of intensive unit stay', axis=1)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            df = df.drop('duration of intensive unit stay', axis=1)
    except Exception as e:
        st.error(f"Error en la celda 14: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 15")
    if show_code:
        st.code("df.columns = df.columns.str.strip()\nlist(df.columns)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            df.columns = df.columns.str.strip()
            list(df.columns)
    except Exception as e:
        st.error(f"Error en la celda 15: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 16")
    if show_code:
        st.code("# Separar categ\u00f3ricas y num\u00e9ricas\ncat_features = binary_cats = [\n    'GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD',\n    'OUTCOME_DAMA', 'OUTCOME_DISCHARGE', 'OUTCOME_EXPIRY',\n    'SMOKING', 'ALCOHOL', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD',\n    'RAISED CARDIAC ENZYMES', 'SEVERE ANAEMIA', 'ANAEMIA', 'STABLE ANGINA',\n    'ACS', 'STEMI', 'ATYPICAL CHEST PAIN', 'HEART FAILURE', 'HFREF', 'HFNEF',\n    'VALVULAR', 'CHB', 'SSS', 'AKI', 'CVA INFRACT', 'CVA BLEED', 'AF', 'VT', 'PSVT',\n    'CONGENITAL', 'UTI', 'NEURO CARDIOGENIC SYNCOPE', 'ORTHOSTATIC',\n    'INFECTIVE ENDOCARDITIS', 'DVT', 'CARDIOGENIC SHOCK', 'SHOCK',\n    'PULMONARY EMBOLISM', 'CHEST INFECTION'\n]\nnum_features = [col for col in df.columns if col not in cat_features and col not in ['D.O.A', 'D.O.D', 'DURATION OF STAY']]", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Separar categóricas y numéricas
            cat_features = binary_cats = [
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
            num_features = [col for col in df.columns if col not in cat_features and col not in ['D.O.A', 'D.O.D', 'DURATION OF STAY']]
    except Exception as e:
        st.error(f"Error en la celda 16: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 17")
    if show_code:
        st.code("num_features", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            num_features
    except Exception as e:
        st.error(f"Error en la celda 17: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 18")
    if show_code:
        st.code("df_numericas = df[num_features]\ndf_numericas.head()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            df_numericas = df[num_features]
            df_numericas.head()
    except Exception as e:
        st.error(f"Error en la celda 18: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 19")
    if show_code:
        st.code("import seaborn as sns\n\n# Crear figura con 3 filas y 3 columnas\nfig, axes = plt.subplots(4, 4, figsize=(20, 15))\naxes = axes.flatten()\n\nfor i, col in enumerate(df_numericas):\n    sns.boxplot(x=df[col], ax=axes[i])\n    axes[i].set_title(col)\n    axes[i].tick_params(axis='x', rotation=45)\n\n# Eliminar ejes vac\u00edos si hay menos gr\u00e1ficos que subplots\nfor j in range(i + 1, len(axes)):\n    fig.delaxes(axes[j])\n\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            import seaborn as sns

            # Crear figura con 3 filas y 3 columnas
            fig, axes = plt.subplots(4, 4, figsize=(20, 15))
            axes = axes.flatten()

            for i, col in enumerate(df_numericas):
                sns.boxplot(x=df[col], ax=axes[i])
                axes[i].set_title(col)
                axes[i].tick_params(axis='x', rotation=45)

            # Eliminar ejes vacíos si hay menos gráficos que subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 19: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 20")
    if show_code:
        st.code("outliers_list = []\nfor c in num_features:\n\n    Q1 = df[c].quantile(0.25)\n    Q3 = df[c].quantile(0.75)\n    IQR = Q3 - Q1\n    lower = Q1 - 1.5 * IQR\n    upper = Q3 + 1.5 * IQR\n\n    mask = (df[c] < lower) | (df[c] > upper)\n\n    temp = (df.loc[mask, [c]]\n              .rename(columns={c: 'value'})\n              .assign(variable=c,\n                      lower_bound=lower,\n                      upper_bound=upper,\n                      row_index=lambda x: x.index))\n\n    outliers_list.append(temp)\n\noutliers = pd.concat(outliers_list, ignore_index=True)\nresumen = outliers.groupby(\"variable\").size().reset_index(name=\"n_outliers\")\nprint(resumen)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            outliers_list = []
            for c in num_features:

                Q1 = df[c].quantile(0.25)
                Q3 = df[c].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                mask = (df[c] < lower) | (df[c] > upper)

                temp = (df.loc[mask, [c]]
                          .rename(columns={c: 'value'})
                          .assign(variable=c,
                                  lower_bound=lower,
                                  upper_bound=upper,
                                  row_index=lambda x: x.index))

                outliers_list.append(temp)

            outliers = pd.concat(outliers_list, ignore_index=True)
            resumen = outliers.groupby("variable").size().reset_index(name="n_outliers")
            print(resumen)
    except Exception as e:
        st.error(f"Error en la celda 20: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 21")
    if show_code:
        st.code("resumen[\"pct_outliers\"] = (resumen[\"n_outliers\"] / len(df)) * 100\nprint(resumen)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            resumen["pct_outliers"] = (resumen["n_outliers"] / len(df)) * 100
            print(resumen)
    except Exception as e:
        st.error(f"Error en la celda 21: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 22")
    if show_code:
        st.code("from scipy.stats import skew\nasimetria_pandas = df_numericas.skew().sort_values(ascending=False)\nprint(\"Asimetr\u00eda con pandas:\")\nprint(asimetria_pandas)\n\n# 3. Identificar variables con fuerte asimetr\u00eda\naltamente_asimetricas = asimetria_pandas[abs(asimetria_pandas) > 2]\nprint(\"\\nVariables con |asimetr\u00eda| > 2:\")\nprint(altamente_asimetricas)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from scipy.stats import skew
            asimetria_pandas = df_numericas.skew().sort_values(ascending=False)
            print("Asimetría con pandas:")
            print(asimetria_pandas)

            # 3. Identificar variables con fuerte asimetría
            altamente_asimetricas = asimetria_pandas[abs(asimetria_pandas) > 2]
            print("\nVariables con |asimetría| > 2:")
            print(altamente_asimetricas)
    except Exception as e:
        st.error(f"Error en la celda 22: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 23")
    if show_code:
        st.code("df_categoricas = df[cat_features]\ndf_categoricas.head()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            df_categoricas = df[cat_features]
            df_categoricas.head()
    except Exception as e:
        st.error(f"Error en la celda 23: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 24")
    if show_code:
        st.code("# Histograma de variables numericas\nvar_hist = num_features\ndf[var_hist].hist(bins=50, figsize=(12, 8))\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Histograma de variables numericas
            var_hist = num_features
            df[var_hist].hist(bins=50, figsize=(12, 8))
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 24: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 25")
    if show_code:
        st.code("sns.pairplot(df[num_features + [\"GENDER\"]], hue=\"GENDER\", diag_kind=\"hist\", height=2.5)\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            sns.pairplot(df[num_features + ["GENDER"]], hue="GENDER", diag_kind="hist", height=2.5)
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 25: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 26")
    if show_code:
        st.code("import seaborn as sns\n\n# Establece un estilo para el gr\u00e1fico\nsns.set_style(\"whitegrid\")\n\n# Obtiene los conteos y renombra el \u00edndice\ngender_counts = df['GENDER'].value_counts().rename(index={1: 'Masculino', 0: 'Femenino'})\n\n# Crea el gr\u00e1fico de barras usando Seaborn\nplt.figure(figsize=(8, 6))\nsns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis')\n\n# Personaliza el gr\u00e1fico\nplt.title('Distribuci\u00f3n de G\u00e9nero', fontsize=16)\nplt.xlabel('G\u00e9nero', fontsize=12)\nplt.ylabel('Cantidad de Personas', fontsize=12)\nplt.xticks(rotation=0)\n\n# Opcional: a\u00f1ade los valores encima de las barras\nfor i, value in enumerate(gender_counts.values):\n    plt.text(i, value, str(value), ha='center', va='bottom', fontsize=10)\n\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            import seaborn as sns

            # Establece un estilo para el gráfico
            sns.set_style("whitegrid")

            # Obtiene los conteos y renombra el índice
            gender_counts = df['GENDER'].value_counts().rename(index={1: 'Masculino', 0: 'Femenino'})

            # Crea el gráfico de barras usando Seaborn
            plt.figure(figsize=(8, 6))
            sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis')

            # Personaliza el gráfico
            plt.title('Distribución de Género', fontsize=16)
            plt.xlabel('Género', fontsize=12)
            plt.ylabel('Cantidad de Personas', fontsize=12)
            plt.xticks(rotation=0)

            # Opcional: añade los valores encima de las barras
            for i, value in enumerate(gender_counts.values):
                plt.text(i, value, str(value), ha='center', va='bottom', fontsize=10)

            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 26: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 27")
    if show_code:
        st.code("# Establece el estilo para una mejor visualizaci\u00f3n\nsns.set_style(\"whitegrid\")\n\n# Crea el histograma usando Seaborn\nplt.figure(figsize=(10, 6))\nsns.histplot(data=df, x='AGE', bins=20, kde=False, color='blue')\n\n# Personaliza el gr\u00e1fico\nplt.title('Distribuci\u00f3n de Edades', fontsize=16)\nplt.xlabel('Edad', fontsize=12)\nplt.ylabel('Frecuencia', fontsize=12)\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Establece el estilo para una mejor visualización
            sns.set_style("whitegrid")

            # Crea el histograma usando Seaborn
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x='AGE', bins=20, kde=False, color='blue')

            # Personaliza el gráfico
            plt.title('Distribución de Edades', fontsize=16)
            plt.xlabel('Edad', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 27: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 28")
    if show_code:
        st.code("# Gr\u00e1fico de dispersi\u00f3n con l\u00ednea de tendencia\nplt.figure(figsize=(8,6))\nsns.scatterplot(data=df, x=\"AGE\", y=\"DURATION OF STAY\", alpha=0.6)\nsns.regplot(data=df, x=\"AGE\", y=\"DURATION OF STAY\", scatter=False, color=\"red\")\n\nplt.title(\"Relaci\u00f3n entre Edad y D\u00edas de Hospitalizaci\u00f3n\")\nplt.xlabel(\"Edad del paciente\")\nplt.ylabel(\"D\u00edas de hospitalizaci\u00f3n\")\nplt.grid(True, linestyle=\"--\", alpha=0.5)\nplt.show()\n\ncorr = df[\"AGE\"].corr(df[\"DURATION OF STAY\"])\nprint(\"Correlaci\u00f3n entre edad y d\u00edas de hospitalizaci\u00f3n:\", corr)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Gráfico de dispersión con línea de tendencia
            plt.figure(figsize=(8,6))
            sns.scatterplot(data=df, x="AGE", y="DURATION OF STAY", alpha=0.6)
            sns.regplot(data=df, x="AGE", y="DURATION OF STAY", scatter=False, color="red")

            plt.title("Relación entre Edad y Días de Hospitalización")
            plt.xlabel("Edad del paciente")
            plt.ylabel("Días de hospitalización")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

            corr = df["AGE"].corr(df["DURATION OF STAY"])
            print("Correlación entre edad y días de hospitalización:", corr)
    except Exception as e:
        st.error(f"Error en la celda 28: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 29")
    if show_code:
        st.code("# Separar variables y objetivo\n#X = pd.concat([df[cat_features], df_numericas_log], axis=1) # variables\nX = df[num_features+cat_features]\ny = df['DURATION OF STAY']  # objetivo\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Separar variables y objetivo
            #X = pd.concat([df[cat_features], df_numericas_log], axis=1) # variables
            X = df[num_features+cat_features]
            y = df['DURATION OF STAY']  # objetivo

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    except Exception as e:
        st.error(f"Error en la celda 29: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 30")
    if show_code:
        st.code("from sklearn.preprocessing import RobustScaler\n\n# Transformador para num\u00e9ricas\nnumeric_transformer = Pipeline(steps=[\n    (\"imputer\", SimpleImputer(strategy=\"mean\")),\n    (\"scaler\", RobustScaler())\n])\n\n# Transformador para categ\u00f3ricas\ncategorical_transformer = Pipeline(steps=[\n    (\"imputer\", SimpleImputer(strategy=\"most_frequent\"))\n])\n\n# Combinamos en un ColumnTransformer\npreprocessor = ColumnTransformer(\n    transformers=[\n        (\"num\", numeric_transformer, num_features),\n        (\"cat\", categorical_transformer, cat_features)\n    ]\n)\n\n# Aplicamos el preprocesamiento\nX_train_processed = preprocessor.fit_transform(X_train)\nX_test_processed = preprocessor.transform(X_test)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.preprocessing import RobustScaler

            # Transformador para numéricas
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", RobustScaler())
            ])

            # Transformador para categóricas
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ])

            # Combinamos en un ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, num_features),
                    ("cat", categorical_transformer, cat_features)
                ]
            )

            # Aplicamos el preprocesamiento
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
    except Exception as e:
        st.error(f"Error en la celda 30: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 31")
    if show_code:
        st.code("# 1. Obtener \u00edndices de columnas num\u00e9ricas en el dataset crudo\nnum_indices = [i for i, col in enumerate(X_train.columns) if col in num_features]\n\n# 2. Filtrar las columnas procesadas usando esos \u00edndices\nX_train_numericas = pd.DataFrame(\n    X_train_processed[:, num_indices],\n    columns=num_features\n)\n\nX_test_numericas = pd.DataFrame(\n    X_test_processed[:, num_indices],\n    columns=num_features\n)\n\n# PCA (elige 70% de var. explicada autom\u00e1ticamente)\npca = PCA(n_components=0.70, random_state=42)\nXn_train_pca = pca.fit_transform(X_train_numericas)\nXn_test_pca  = pca.transform(X_test_numericas)\n\npca_names = [f'PCA{i+1}' for i in range(Xn_train_pca.shape[1])]\nXn_train_pca = pd.DataFrame(Xn_train_pca, columns=pca_names, index=X_train.index)\nXn_test_pca  = pd.DataFrame(Xn_test_pca,  columns=pca_names, index=X_test.index)\n\nprint(f'PCA: {len(pca_names)} componentes, var. explicada acumulada = {pca.explained_variance_ratio_.sum():.3f}')", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # 1. Obtener índices de columnas numéricas en el dataset crudo
            num_indices = [i for i, col in enumerate(X_train.columns) if col in num_features]

            # 2. Filtrar las columnas procesadas usando esos índices
            X_train_numericas = pd.DataFrame(
                X_train_processed[:, num_indices],
                columns=num_features
            )

            X_test_numericas = pd.DataFrame(
                X_test_processed[:, num_indices],
                columns=num_features
            )

            # PCA (elige 70% de var. explicada automáticamente)
            pca = PCA(n_components=0.70, random_state=42)
            Xn_train_pca = pca.fit_transform(X_train_numericas)
            Xn_test_pca  = pca.transform(X_test_numericas)

            pca_names = [f'PCA{i+1}' for i in range(Xn_train_pca.shape[1])]
            Xn_train_pca = pd.DataFrame(Xn_train_pca, columns=pca_names, index=X_train.index)
            Xn_test_pca  = pd.DataFrame(Xn_test_pca,  columns=pca_names, index=X_test.index)

            print(f'PCA: {len(pca_names)} componentes, var. explicada acumulada = {pca.explained_variance_ratio_.sum():.3f}')
    except Exception as e:
        st.error(f"Error en la celda 31: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 32")
    if show_code:
        st.code("# Varianza explicada por cada componente\nvar_exp = pca.explained_variance_ratio_\ncum_var_exp = np.cumsum(var_exp)\n\nplt.figure(figsize=(8,5))\n\n# Barras de varianza explicada individual\nplt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.6, label='Varianza explicada por componente')\n\n# L\u00ednea de varianza acumulada\nplt.step(range(1, len(cum_var_exp)+1), cum_var_exp, where='mid', color='red', label='Varianza acumulada')\n\n# L\u00ednea horizontal del 70% (opcional, ya que usaste 0.70)\nplt.axhline(y=0.70, color='green', linestyle='--', label='70%')\n\nplt.xlabel('Componentes principales')\nplt.ylabel('Proporci\u00f3n de varianza explicada')\nplt.title('PCA - Varianza explicada y acumulada')\nplt.xticks(range(1, len(var_exp)+1))\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Varianza explicada por cada componente
            var_exp = pca.explained_variance_ratio_
            cum_var_exp = np.cumsum(var_exp)

            plt.figure(figsize=(8,5))

            # Barras de varianza explicada individual
            plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.6, label='Varianza explicada por componente')

            # Línea de varianza acumulada
            plt.step(range(1, len(cum_var_exp)+1), cum_var_exp, where='mid', color='red', label='Varianza acumulada')

            # Línea horizontal del 70% (opcional, ya que usaste 0.70)
            plt.axhline(y=0.70, color='green', linestyle='--', label='70%')

            plt.xlabel('Componentes principales')
            plt.ylabel('Proporción de varianza explicada')
            plt.title('PCA - Varianza explicada y acumulada')
            plt.xticks(range(1, len(var_exp)+1))
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 32: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 33")
    if show_code:
        st.code("# Obtener las primeras dos componentes para el gr\u00e1fico\npc1 = Xn_train_pca.iloc[:, 0]\npc2 = Xn_train_pca.iloc[:, 1]\n\n# Crear el gr\u00e1fico de dispersi\u00f3n\nplt.figure(figsize=(10, 8))\nplt.scatter(pc1, pc2, alpha=0.6, s=20) # 'alpha' controla la transparencia y 's' el tama\u00f1o del punto\nplt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')\nplt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')\nplt.title('Distribuci\u00f3n de los datos en el espacio de las primeras 2 componentes principales')\nplt.grid(True)\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Obtener las primeras dos componentes para el gráfico
            pc1 = Xn_train_pca.iloc[:, 0]
            pc2 = Xn_train_pca.iloc[:, 1]

            # Crear el gráfico de dispersión
            plt.figure(figsize=(10, 8))
            plt.scatter(pc1, pc2, alpha=0.6, s=20) # 'alpha' controla la transparencia y 's' el tamaño del punto
            plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
            plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
            plt.title('Distribución de los datos en el espacio de las primeras 2 componentes principales')
            plt.grid(True)
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 33: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 34")
    if show_code:
        st.code("# Crear un DataFrame con los loadings\nloadings = pd.DataFrame(pca.components_, columns=num_features, index=pca_names)\n\nplt.figure(figsize=(12, 8))\nsns.heatmap(loadings.T, annot=True, fmt=\".2f\", cmap='coolwarm', cbar=True)\nplt.title('Mapa de calor de los loadings del PCA')\nplt.xlabel('Componentes Principales')\nplt.ylabel('Variables Originales', )\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Crear un DataFrame con los loadings
            loadings = pd.DataFrame(pca.components_, columns=num_features, index=pca_names)

            plt.figure(figsize=(12, 8))
            sns.heatmap(loadings.T, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
            plt.title('Mapa de calor de los loadings del PCA')
            plt.xlabel('Componentes Principales')
            plt.ylabel('Variables Originales', )
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 34: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 35")
    if show_code:
        st.code("# 1. Obtener \u00edndices de las columnas categ\u00f3ricas en el X_train original\ncat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]\n\n# 2. Filtrar las columnas categ\u00f3ricas procesadas\nX_train_categoricas = pd.DataFrame(\n    X_train_processed[:, cat_indices],\n    columns=cat_features\n)\n\nX_test_categoricas = pd.DataFrame(\n    X_test_processed[:, cat_indices],\n    columns=cat_features\n)\n\n# 3. MCA con prince\nimport prince\n\nmca = prince.MCA(\n    n_components=5,\n    n_iter=5,\n    random_state=42\n)\nmca.fit(X_train_categoricas)\n\nXc_train_mca = mca.transform(X_train_categoricas)\nXc_test_mca  = mca.transform(X_test_categoricas)\nXc_train_mca.index = X_train.index\nXc_test_mca.index  = X_test.index\n\n# 4. Renombrar componentes\nmca_names = [f'MCA{i+1}' for i in range(Xc_train_mca.shape[1])]\nXc_train_mca.columns = mca_names\nXc_test_mca.columns  = mca_names\n\n# 5. Porcentaje de varianza (inercia)\nev = mca.eigenvalues_summary\nprint('MCA inercia por eje:', ev['% of variance'].values)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # 1. Obtener índices de las columnas categóricas en el X_train original
            cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]

            # 2. Filtrar las columnas categóricas procesadas
            X_train_categoricas = pd.DataFrame(
                X_train_processed[:, cat_indices],
                columns=cat_features
            )

            X_test_categoricas = pd.DataFrame(
                X_test_processed[:, cat_indices],
                columns=cat_features
            )

            # 3. MCA con prince
            import prince

            mca = prince.MCA(
                n_components=5,
                n_iter=5,
                random_state=42
            )
            mca.fit(X_train_categoricas)

            Xc_train_mca = mca.transform(X_train_categoricas)
            Xc_test_mca  = mca.transform(X_test_categoricas)
            Xc_train_mca.index = X_train.index
            Xc_test_mca.index  = X_test.index

            # 4. Renombrar componentes
            mca_names = [f'MCA{i+1}' for i in range(Xc_train_mca.shape[1])]
            Xc_train_mca.columns = mca_names
            Xc_test_mca.columns  = mca_names

            # 5. Porcentaje de varianza (inercia)
            ev = mca.eigenvalues_summary
            print('MCA inercia por eje:', ev['% of variance'].values)
    except Exception as e:
        st.error(f"Error en la celda 35: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 36")
    if show_code:
        st.code("# Coordenadas de las categor\u00edas en los ejes principales\ncoords = mca.column_coordinates(X_train_categoricas)\n\n# Normalizar nombres (para saber qu\u00e9 categor\u00eda pertenece a qu\u00e9 variable)\ncoords.index = coords.index.astype(str)\n\n# Heatmap de loadings (categor\u00edas vs componentes)\nplt.figure(figsize=(10,6))\nthreshold = 0.2  # ejemplo\nfiltered_data = coords.loc[:, coords.abs().max() > threshold]\nsns.heatmap(filtered_data, cmap=\"coolwarm\", center=0)\nplt.title(\"Heatmap de loadings - MCA\")\nplt.xlabel(\"Categor\u00edas\")\nplt.ylabel(\"Componentes\")\nplt.xticks(rotation=90, fontsize=7)\nplt.yticks(fontsize=8)\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Coordenadas de las categorías en los ejes principales
            coords = mca.column_coordinates(X_train_categoricas)

            # Normalizar nombres (para saber qué categoría pertenece a qué variable)
            coords.index = coords.index.astype(str)

            # Heatmap de loadings (categorías vs componentes)
            plt.figure(figsize=(10,6))
            threshold = 0.2  # ejemplo
            filtered_data = coords.loc[:, coords.abs().max() > threshold]
            sns.heatmap(filtered_data, cmap="coolwarm", center=0)
            plt.title("Heatmap de loadings - MCA")
            plt.xlabel("Categorías")
            plt.ylabel("Componentes")
            plt.xticks(rotation=90, fontsize=7)
            plt.yticks(fontsize=8)
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 36: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 37")
    if show_code:
        st.code("# Convertir a num\u00e9rico, quitando s\u00edmbolos si es necesario\nev['% of variance'] = ev['% of variance'].replace('%', '', regex=True)  # elimina el s\u00edmbolo %\nev['% of variance'] = ev['% of variance'].str.replace(',', '.', regex=False)  # cambia coma por punto\nev['% of variance'] = pd.to_numeric(ev['% of variance'], errors='coerce')\n\n# Ahora s\u00ed, en proporci\u00f3n\nvar_exp_mca = ev['% of variance'].values / 100\ncum_var_exp_mca = np.cumsum(var_exp_mca)\n\nplt.figure(figsize=(8,5))\nplt.plot(range(1, len(var_exp_mca) + 1), cum_var_exp_mca, marker='o')\nplt.xlabel('Dimensiones')\nplt.ylabel('Varianza Acumulada')\nplt.grid(True)\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Convertir a numérico, quitando símbolos si es necesario
            ev['% of variance'] = ev['% of variance'].replace('%', '', regex=True)  # elimina el símbolo %
            ev['% of variance'] = ev['% of variance'].str.replace(',', '.', regex=False)  # cambia coma por punto
            ev['% of variance'] = pd.to_numeric(ev['% of variance'], errors='coerce')

            # Ahora sí, en proporción
            var_exp_mca = ev['% of variance'].values / 100
            cum_var_exp_mca = np.cumsum(var_exp_mca)

            plt.figure(figsize=(8,5))
            plt.plot(range(1, len(var_exp_mca) + 1), cum_var_exp_mca, marker='o')
            plt.xlabel('Dimensiones')
            plt.ylabel('Varianza Acumulada')
            plt.grid(True)
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 37: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 38")
    if show_code:
        st.code("# Asegurar que los valores sean num\u00e9ricos\nev['% of variance'] = pd.to_numeric(ev['% of variance'], errors='coerce')\n\n# Varianza explicada en proporci\u00f3n y acumulada\nvar_exp_mca = ev['% of variance'].values / 100\ncum_var_exp_mca = np.cumsum(var_exp_mca)\ncomponentes = np.arange(1, len(var_exp_mca) + 1)\n\n# Gr\u00e1fica\nplt.figure(figsize=(8, 5))\n\n# Barras: varianza explicada individual\nplt.bar(componentes, var_exp_mca, alpha=0.7, label='Varianza explicada')\n\n# L\u00ednea: varianza acumulada\nplt.plot(componentes, cum_var_exp_mca, marker='o', color='red', label='Varianza acumulada')\n\n# Formato\nplt.xticks(componentes)\nplt.xlabel('Componentes MCA')\nplt.ylabel('Proporci\u00f3n de varianza')\nplt.title('Scree plot - MCA')\nplt.legend()\nplt.grid(alpha=0.3)\n\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Asegurar que los valores sean numéricos
            ev['% of variance'] = pd.to_numeric(ev['% of variance'], errors='coerce')

            # Varianza explicada en proporción y acumulada
            var_exp_mca = ev['% of variance'].values / 100
            cum_var_exp_mca = np.cumsum(var_exp_mca)
            componentes = np.arange(1, len(var_exp_mca) + 1)

            # Gráfica
            plt.figure(figsize=(8, 5))

            # Barras: varianza explicada individual
            plt.bar(componentes, var_exp_mca, alpha=0.7, label='Varianza explicada')

            # Línea: varianza acumulada
            plt.plot(componentes, cum_var_exp_mca, marker='o', color='red', label='Varianza acumulada')

            # Formato
            plt.xticks(componentes)
            plt.xlabel('Componentes MCA')
            plt.ylabel('Proporción de varianza')
            plt.title('Scree plot - MCA')
            plt.legend()
            plt.grid(alpha=0.3)

            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 38: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 39")
    if show_code:
        st.code("# Supongamos que tu MCA ya est\u00e1 ajustado\nmca = prince.MCA(n_components=2, random_state=42)\n\nmca = mca.fit(df_categoricas)\n\n# Coordenadas de las filas\nrow_coords = mca.row_coordinates(df_categoricas)\n\n# Gr\u00e1fico de dispersi\u00f3n\nplt.figure(figsize=(10,6))\nsns.scatterplot(\n    x=row_coords[0],\n    y=row_coords[1],\n    alpha=0.6\n)\n\n# Use the eigenvalues_ attribute to get the explained variance ratio\nexplained_variance_ratio = mca.eigenvalues_\n\nplt.xlabel(f\"Componente 1 ({explained_variance_ratio[0]*100:.2f}%)\")\nplt.ylabel(f\"Componente 2 ({explained_variance_ratio[1]*100:.2f}%)\")\nplt.title(\"MCA - Scatterplot de las dos primeras componentes\")\nplt.grid(True, linestyle=\"--\", alpha=0.5)\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Supongamos que tu MCA ya está ajustado
            mca = prince.MCA(n_components=2, random_state=42)

            mca = mca.fit(df_categoricas)

            # Coordenadas de las filas
            row_coords = mca.row_coordinates(df_categoricas)

            # Gráfico de dispersión
            plt.figure(figsize=(10,6))
            sns.scatterplot(
                x=row_coords[0],
                y=row_coords[1],
                alpha=0.6
            )

            # Use the eigenvalues_ attribute to get the explained variance ratio
            explained_variance_ratio = mca.eigenvalues_

            plt.xlabel(f"Componente 1 ({explained_variance_ratio[0]*100:.2f}%)")
            plt.ylabel(f"Componente 2 ({explained_variance_ratio[1]*100:.2f}%)")
            plt.title("MCA - Scatterplot de las dos primeras componentes")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 39: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 40")
    if show_code:
        st.code("X_train_reduced = pd.concat([Xn_train_pca, Xc_train_mca], axis=1)\nX_test_reduced  = pd.concat([Xn_test_pca,  Xc_test_mca],  axis=1)\n\nprint('Shape train reducido:', X_train_reduced.shape)\nprint('Shape test  reducido:', X_test_reduced.shape)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            X_train_reduced = pd.concat([Xn_train_pca, Xc_train_mca], axis=1)
            X_test_reduced  = pd.concat([Xn_test_pca,  Xc_test_mca],  axis=1)

            print('Shape train reducido:', X_train_reduced.shape)
            print('Shape test  reducido:', X_test_reduced.shape)
    except Exception as e:
        st.error(f"Error en la celda 40: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 41")
    if show_code:
        st.code("X_train_reduced = pd.concat(\n    [Xn_train_pca.reset_index(drop=True),\n     Xc_train_mca.reset_index(drop=True)],\n    axis=1\n)\nX_train_reduced.head(10)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            X_train_reduced = pd.concat(
                [Xn_train_pca.reset_index(drop=True),
                 Xc_train_mca.reset_index(drop=True)],
                axis=1
            )
            X_train_reduced.head(10)
    except Exception as e:
        st.error(f"Error en la celda 41: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 42")
    if show_code:
        st.code("X_train_reduced.shape", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            X_train_reduced.shape
    except Exception as e:
        st.error(f"Error en la celda 42: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 43")
    if show_code:
        st.code("# ============================\n# 4) PRE-PRUNING con RandomizedSearchCV (solo TRAIN)\n# ============================\nimport numpy as np # Import numpy\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.model_selection import RandomizedSearchCV\nfrom sklearn.metrics import mean_squared_error, r2_score\n\npipe_pre = Pipeline(steps=[\n    ('model', DecisionTreeRegressor(random_state=42))\n])\n\nparam_dist_pre = {\n    'model__max_depth': np.arange(2, 20),\n    'model__min_samples_split': np.arange(10, 50),\n    'model__min_samples_leaf': np.arange(5, 20),\n    'model__max_features': [None, 'sqrt', 'log2']\n}\n\nsearch_pre = RandomizedSearchCV(\n    estimator=pipe_pre,\n    param_distributions=param_dist_pre,\n    n_iter=60,\n    cv=5,\n    scoring='neg_root_mean_squared_error',\n    n_jobs=-1,\n    random_state=42,\n    refit=True\n)\n\nsearch_pre.fit(X_train_reduced, y_train)\n\n\nprint(\"\\n=== PRE-PRUNING ===\")\nprint(\"Mejores hiperpar\u00e1metros:\", search_pre.best_params_)\nprint(f\"Mejor RMSE CV: {-search_pre.best_score_:.4f}\")\n\n# Evaluaci\u00f3n final en test (primera vez que tocamos test para pre-pruning)\ny_pred_pre = search_pre.predict(X_test_reduced)\nrmse_test_pre = np.sqrt(mean_squared_error(y_test, y_pred_pre))\nr2_test_pre = r2_score(y_test, y_pred_pre)\nrmse_rel_test_pre = rmse_test_pre / y_test.mean()\n\nprint(f'\ud83d\udd0e RMSE: {rmse_test_pre:.3f}')\nprint(f'\ud83d\udd0e R\u00b2: {r2_test_pre:.3f}')\nprint(f'\ud83d\udd0e RMSE relativo (test): {rmse_rel_test_pre:.3f}')", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # ============================
            # 4) PRE-PRUNING con RandomizedSearchCV (solo TRAIN)
            # ============================
            import numpy as np # Import numpy
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.metrics import mean_squared_error, r2_score

            pipe_pre = Pipeline(steps=[
                ('model', DecisionTreeRegressor(random_state=42))
            ])

            param_dist_pre = {
                'model__max_depth': np.arange(2, 20),
                'model__min_samples_split': np.arange(10, 50),
                'model__min_samples_leaf': np.arange(5, 20),
                'model__max_features': [None, 'sqrt', 'log2']
            }

            search_pre = RandomizedSearchCV(
                estimator=pipe_pre,
                param_distributions=param_dist_pre,
                n_iter=60,
                cv=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                random_state=42,
                refit=True
            )

            search_pre.fit(X_train_reduced, y_train)


            print("\n=== PRE-PRUNING ===")
            print("Mejores hiperparámetros:", search_pre.best_params_)
            print(f"Mejor RMSE CV: {-search_pre.best_score_:.4f}")

            # Evaluación final en test (primera vez que tocamos test para pre-pruning)
            y_pred_pre = search_pre.predict(X_test_reduced)
            rmse_test_pre = np.sqrt(mean_squared_error(y_test, y_pred_pre))
            r2_test_pre = r2_score(y_test, y_pred_pre)
            rmse_rel_test_pre = rmse_test_pre / y_test.mean()

            print(f'🔎 RMSE: {rmse_test_pre:.3f}')
            print(f'🔎 R²: {r2_test_pre:.3f}')
            print(f'🔎 RMSE relativo (test): {rmse_rel_test_pre:.3f}')
    except Exception as e:
        st.error(f"Error en la celda 43: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 44")
    if show_code:
        st.code("\nfrom sklearn.model_selection import learning_curve\n# ============================\n# 4. Curva de aprendizaje\n# ============================\ntrain_sizes, train_scores, val_scores = learning_curve(\n    estimator=search_pre.best_estimator_,   # usa el mejor modelo del RandomizedSearchCV\n    X=X_train_reduced,\n    y=y_train,\n    cv=5,\n    scoring=\"neg_root_mean_squared_error\",\n    n_jobs=-1,\n    train_sizes=np.linspace(0.1, 1.0, 10), # 10 tama\u00f1os distintos de train\n    random_state=42\n)\n\n# Como sklearn devuelve negativos para errores (porque maximiza), invertimos el signo\ntrain_rmse = -np.mean(train_scores, axis=1)\nval_rmse = -np.mean(val_scores, axis=1)\n\n# Gr\u00e1fico\nplt.figure(figsize=(10, 6))\nplt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Error de Entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Error de Validaci\u00f3n')\n\nplt.title('Curva de Aprendizaje - \u00c1rbol de Decisi\u00f3n con Pre-Pruning')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.legend(loc='best')\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):

            from sklearn.model_selection import learning_curve
            # ============================
            # 4. Curva de aprendizaje
            # ============================
            train_sizes, train_scores, val_scores = learning_curve(
                estimator=search_pre.best_estimator_,   # usa el mejor modelo del RandomizedSearchCV
                X=X_train_reduced,
                y=y_train,
                cv=5,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10), # 10 tamaños distintos de train
                random_state=42
            )

            # Como sklearn devuelve negativos para errores (porque maximiza), invertimos el signo
            train_rmse = -np.mean(train_scores, axis=1)
            val_rmse = -np.mean(val_scores, axis=1)

            # Gráfico
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Error de Entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Error de Validación')

            plt.title('Curva de Aprendizaje - Árbol de Decisión con Pre-Pruning')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 44: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 45")
    if show_code:
        st.code("# ============================\n# 5) POST-PRUNING (\u00e1rbol grande -> path alphas -> CV en TRAIN)\n# ============================\n#import numpy as np\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.model_selection import KFold, cross_val_score\nfrom sklearn.metrics import mean_squared_error\n\n# === 0) Cast a float32 si aplica (acelera y reduce memoria)\nX_train_np = np.asarray(X_train_reduced, dtype=np.float32)\ny_train_np = np.asarray(y_train)  # target puede quedar en float64\nX_test_np  = np.asarray(X_test_reduced,  dtype=np.float32)\ny_test_np  = np.asarray(y_test)\n\n# === 1) Ruta de poda en un SUBSET\nrng = np.random.RandomState(42)\nn_sub = min(len(X_train_np), 15000)\nidx = rng.choice(len(X_train_np), n_sub, replace=False)\n\ntree_full_sub = DecisionTreeRegressor(random_state=42)\ntree_full_sub.fit(X_train_np[idx], y_train_np[idx])\n\npath = tree_full_sub.cost_complexity_pruning_path(X_train_np[idx], y_train_np[idx])\nalphas_full = np.unique(np.round(path.ccp_alphas, 10))\n\n# === 2) Muestrea ~40 alphas representativos (cuantiles)\nif len(alphas_full) > 40:\n    quantiles = np.linspace(0, 1, 40)\n    ccp_alphas = np.quantile(alphas_full, quantiles)\nelse:\n    ccp_alphas = alphas_full\n\n# === 3) CV paralela y \u00e1rbol con l\u00edmites de complejidad (m\u00e1s r\u00e1pido)\nkf = KFold(n_splits=3, shuffle=True, random_state=42)\n\ndef cv_rmse_for_alpha(a):\n    model = DecisionTreeRegressor(\n        random_state=42,\n        ccp_alpha=a,\n        max_depth=6,\n        min_samples_leaf=11\n    )\n    scores = cross_val_score(\n        model, X_train_np, y_train_np,\n        scoring='neg_root_mean_squared_error',\n        cv=kf, n_jobs=-1\n    )\n    return float((-scores).mean())\n\nrmse_list = [cv_rmse_for_alpha(a) for a in ccp_alphas]\nbest_idx   = int(np.argmin(rmse_list))\nbest_alpha = float(ccp_alphas[best_idx])\n\nprint(\"\\n=== POST-PRUNING ===\")\nprint(f\"Mejor ccp_alpha: {best_alpha:.6f} | RMSE CV={rmse_list[best_idx]:.4f}\")\n\n# === 4) Entrena SOLO el \u00e1rbol final y, si quieres, mide tama\u00f1o una vez\ntree_pruned = DecisionTreeRegressor(\n    random_state=42,\n    ccp_alpha=best_alpha,\n    max_depth=20,\n    min_samples_leaf=5\n)\ntree_pruned.fit(X_train_np, y_train_np)\n\nn_leaves = tree_pruned.get_n_leaves()\nn_nodes  = tree_pruned.tree_.node_count\nprint(f\"Hojas: {n_leaves} | Nodos:\u00a0{n_nodes}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # ============================
            # 5) POST-PRUNING (árbol grande -> path alphas -> CV en TRAIN)
            # ============================
            #import numpy as np
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import KFold, cross_val_score
            from sklearn.metrics import mean_squared_error

            # === 0) Cast a float32 si aplica (acelera y reduce memoria)
            X_train_np = np.asarray(X_train_reduced, dtype=np.float32)
            y_train_np = np.asarray(y_train)  # target puede quedar en float64
            X_test_np  = np.asarray(X_test_reduced,  dtype=np.float32)
            y_test_np  = np.asarray(y_test)

            # === 1) Ruta de poda en un SUBSET
            rng = np.random.RandomState(42)
            n_sub = min(len(X_train_np), 15000)
            idx = rng.choice(len(X_train_np), n_sub, replace=False)

            tree_full_sub = DecisionTreeRegressor(random_state=42)
            tree_full_sub.fit(X_train_np[idx], y_train_np[idx])

            path = tree_full_sub.cost_complexity_pruning_path(X_train_np[idx], y_train_np[idx])
            alphas_full = np.unique(np.round(path.ccp_alphas, 10))

            # === 2) Muestrea ~40 alphas representativos (cuantiles)
            if len(alphas_full) > 40:
                quantiles = np.linspace(0, 1, 40)
                ccp_alphas = np.quantile(alphas_full, quantiles)
            else:
                ccp_alphas = alphas_full

            # === 3) CV paralela y árbol con límites de complejidad (más rápido)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            def cv_rmse_for_alpha(a):
                model = DecisionTreeRegressor(
                    random_state=42,
                    ccp_alpha=a,
                    max_depth=6,
                    min_samples_leaf=11
                )
                scores = cross_val_score(
                    model, X_train_np, y_train_np,
                    scoring='neg_root_mean_squared_error',
                    cv=kf, n_jobs=-1
                )
                return float((-scores).mean())

            rmse_list = [cv_rmse_for_alpha(a) for a in ccp_alphas]
            best_idx   = int(np.argmin(rmse_list))
            best_alpha = float(ccp_alphas[best_idx])

            print("\n=== POST-PRUNING ===")
            print(f"Mejor ccp_alpha: {best_alpha:.6f} | RMSE CV={rmse_list[best_idx]:.4f}")

            # === 4) Entrena SOLO el árbol final y, si quieres, mide tamaño una vez
            tree_pruned = DecisionTreeRegressor(
                random_state=42,
                ccp_alpha=best_alpha,
                max_depth=20,
                min_samples_leaf=5
            )
            tree_pruned.fit(X_train_np, y_train_np)

            n_leaves = tree_pruned.get_n_leaves()
            n_nodes  = tree_pruned.tree_.node_count
            print(f"Hojas: {n_leaves} | Nodos: {n_nodes}")
    except Exception as e:
        st.error(f"Error en la celda 45: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 46")
    if show_code:
        st.code("# --- Curva de Aprendizaje con el \u00e1rbol podado ---\ntrain_sizes, train_scores, val_scores = learning_curve(\n    tree_pruned,                     # tu \u00e1rbol ya podado\n    X_train_np, y_train_np,\n    train_sizes=np.linspace(0.1, 1.0, 10),  # desde 10% hasta 100% de los datos\n    cv=5,                             # 5 folds\n    scoring='neg_mean_squared_error',\n    n_jobs=-1\n)\n\n# Convertir MSE negativo a RMSE\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse   = np.sqrt(-val_scores.mean(axis=1))\n\n# --- Gr\u00e1fico ---\nplt.figure(figsize=(8,6))\nplt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Validaci\u00f3n')\n\nplt.title('Curva de Aprendizaje - \u00c1rbol Podado')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.legend(loc='best')\nplt.grid(True)\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # --- Curva de Aprendizaje con el árbol podado ---
            train_sizes, train_scores, val_scores = learning_curve(
                tree_pruned,                     # tu árbol ya podado
                X_train_np, y_train_np,
                train_sizes=np.linspace(0.1, 1.0, 10),  # desde 10% hasta 100% de los datos
                cv=5,                             # 5 folds
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            # Convertir MSE negativo a RMSE
            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse   = np.sqrt(-val_scores.mean(axis=1))

            # --- Gráfico ---
            plt.figure(figsize=(8,6))
            plt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Validación')

            plt.title('Curva de Aprendizaje - Árbol Podado')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.legend(loc='best')
            plt.grid(True)
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 46: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 47")
    if show_code:
        st.code("y_pred_post = tree_pruned.predict(X_test_np)\nrmse_test_post = np.sqrt(mean_squared_error(y_test_np, y_pred_post))\nr2_test_post = r2_score(y_test_np, y_pred_post)\nrmse_rel_test_post = rmse_test_post / y_test_np.mean()\n\nprint('\\n=== Evaluaci\u00f3n Final POST-PRUNING ===')\nprint(f'\ud83d\udd0e RMSE: {rmse_test_post:.3f}')\nprint(f'\ud83d\udd0e R\u00b2: {r2_test_post:.3f}')\nprint(f'\ud83d\udd0e RMSE relativo (test): {rmse_rel_test_post:.3f}')", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            y_pred_post = tree_pruned.predict(X_test_np)
            rmse_test_post = np.sqrt(mean_squared_error(y_test_np, y_pred_post))
            r2_test_post = r2_score(y_test_np, y_pred_post)
            rmse_rel_test_post = rmse_test_post / y_test_np.mean()

            print('\n=== Evaluación Final POST-PRUNING ===')
            print(f'🔎 RMSE: {rmse_test_post:.3f}')
            print(f'🔎 R²: {r2_test_post:.3f}')
            print(f'🔎 RMSE relativo (test): {rmse_rel_test_post:.3f}')
    except Exception as e:
        st.error(f"Error en la celda 47: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 48")
    if show_code:
        st.code("from sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import GridSearchCV, KFold, learning_curve\nimport numpy as np\nimport matplotlib.pyplot as plt\n\npipe_knn = Pipeline(steps=[\n    (\"knn\", KNeighborsRegressor())\n])\nparam_grid = {\n    \"knn__n_neighbors\": list(range(1, 50)),\n    \"knn__weights\": [\"uniform\", \"distance\"],\n    \"knn__metric\": [\"minkowski\"],\n    \"knn__p\": [1, 2],\n    \"knn__algorithm\": ['brute', 'kd_tree', 'ball_tree']\n}\n\ncv = KFold(n_splits=5, shuffle=True, random_state=42)\ngs = GridSearchCV(pipe_knn, param_grid, cv=cv, scoring=\"neg_root_mean_squared_error\", n_jobs=-1)\ngs.fit(X_train_reduced, y_train)\n\n# Obtener el mejor modelo ya optimizado\nbest_knn_model = gs.best_estimator_\n\n# Imprimir los resultados del mejor modelo\nprint(\"=== KNN - Mejores HP ===\")\nprint(\"Mejores hiperpar\u00e1metros:\", gs.best_params_)\nprint(f\"Mejor RMSE CV: {-gs.best_score_:.4f}\")\n\n# Evaluaci\u00f3n final en el conjunto de prueba\ny_pred_test = best_knn_model.predict(X_test_reduced)\nrmse_test_knn = np.sqrt(mean_squared_error(y_test, y_pred_test))\nr2_test_knn = r2_score(y_test, y_pred_test)\nrmse_rel_test_knn = rmse_test_knn / y_test.mean()\n\nprint(f'\ud83d\udd0e RMSE: {rmse_test_knn:.3f}')\nprint(f'\ud83d\udd0e R\u00b2: {r2_test_knn:.3f}')\nprint(f'\ud83d\udd0e RMSE relativo (test): {rmse_rel_test_knn:.3f}')\n\n\n# --- Curva de Aprendizaje del mejor modelo ---\ntrain_sizes, train_scores, val_scores = learning_curve(\n    best_knn_model, X_train_reduced, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=cv,\n    scoring='neg_root_mean_squared_error',\n    random_state=42\n)\n\ntrain_rmse = -train_scores.mean(axis=1)\nval_rmse = -val_scores.mean(axis=1)\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse, 'o-', label='Error de entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', label='Error de validaci\u00f3n')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de aprendizaje del mejor modelo KNN')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import GridSearchCV, KFold, learning_curve
            import numpy as np
            import matplotlib.pyplot as plt

            pipe_knn = Pipeline(steps=[
                ("knn", KNeighborsRegressor())
            ])
            param_grid = {
                "knn__n_neighbors": list(range(1, 50)),
                "knn__weights": ["uniform", "distance"],
                "knn__metric": ["minkowski"],
                "knn__p": [1, 2],
                "knn__algorithm": ['brute', 'kd_tree', 'ball_tree']
            }

            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            gs = GridSearchCV(pipe_knn, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(X_train_reduced, y_train)

            # Obtener el mejor modelo ya optimizado
            best_knn_model = gs.best_estimator_

            # Imprimir los resultados del mejor modelo
            print("=== KNN - Mejores HP ===")
            print("Mejores hiperparámetros:", gs.best_params_)
            print(f"Mejor RMSE CV: {-gs.best_score_:.4f}")

            # Evaluación final en el conjunto de prueba
            y_pred_test = best_knn_model.predict(X_test_reduced)
            rmse_test_knn = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test_knn = r2_score(y_test, y_pred_test)
            rmse_rel_test_knn = rmse_test_knn / y_test.mean()

            print(f'🔎 RMSE: {rmse_test_knn:.3f}')
            print(f'🔎 R²: {r2_test_knn:.3f}')
            print(f'🔎 RMSE relativo (test): {rmse_rel_test_knn:.3f}')


            # --- Curva de Aprendizaje del mejor modelo ---
            train_sizes, train_scores, val_scores = learning_curve(
                best_knn_model, X_train_reduced, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=cv,
                scoring='neg_root_mean_squared_error',
                random_state=42
            )

            train_rmse = -train_scores.mean(axis=1)
            val_rmse = -val_scores.mean(axis=1)

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse, 'o-', label='Error de entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', label='Error de validación')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de aprendizaje del mejor modelo KNN')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 48: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 49")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n Lineal\n#######################################\n\nfrom sklearn.model_selection import learning_curve\n\nlin_pipeline = Pipeline([\n    (\"regressor\", LinearRegression())\n])\n\n# Curva de aprendizaje\ntrain_sizes, train_scores, val_scores = learning_curve(\n    lin_pipeline, X_train_reduced, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=5,\n    scoring='neg_mean_squared_error',\n    random_state=42\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse, 'o-', label='Error de entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', label='Error de validaci\u00f3n')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de aprendizaje - Regresi\u00f3n lineal')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión Lineal
            #######################################

            from sklearn.model_selection import learning_curve

            lin_pipeline = Pipeline([
                ("regressor", LinearRegression())
            ])

            # Curva de aprendizaje
            train_sizes, train_scores, val_scores = learning_curve(
                lin_pipeline, X_train_reduced, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                scoring='neg_mean_squared_error',
                random_state=42
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse, 'o-', label='Error de entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', label='Error de validación')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de aprendizaje - Regresión lineal')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 49: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 50")
    if show_code:
        st.code("lin_pipeline.fit(X_train_reduced, y_train)\ny_pred = lin_pipeline.predict(X_test_reduced)\nprint(f'\ud83d\udd0e RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}')\nprint(f'\ud83d\udd0e R\u00b2: {r2_score(y_test, y_pred):.3f}')\nprint(f\"RMSE relativo (test): {np.sqrt(mean_squared_error(y_test, y_pred)) / y_test.mean():.3f}\")\n# revsiar que el error cuadr\u00e1tico medio sea menor a 10%, si es el caso, es necesario mirar otro modelo.", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            lin_pipeline.fit(X_train_reduced, y_train)
            y_pred = lin_pipeline.predict(X_test_reduced)
            print(f'🔎 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}')
            print(f'🔎 R²: {r2_score(y_test, y_pred):.3f}')
            print(f"RMSE relativo (test): {np.sqrt(mean_squared_error(y_test, y_pred)) / y_test.mean():.3f}")
            # revsiar que el error cuadrático medio sea menor a 10%, si es el caso, es necesario mirar otro modelo.
    except Exception as e:
        st.error(f"Error en la celda 50: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 51")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n Lasso\n#######################################\n\nfrom sklearn.linear_model import Lasso # Import Lasso\nfrom sklearn.model_selection import validation_curve # Import validation_curve\n\nlasso_pipeline = Pipeline([\n    (\"regressor\", Lasso(max_iter=10000))  # Aumentar iteraciones si hay warning\n])\n\n# Rango de valores para alpha\nalphas = np.logspace(-4, 1, 10)\n\n# Validaci\u00f3n cruzada sobre X_train e y_train\ntrain_scores, val_scores = validation_curve(\n    lasso_pipeline, X_train_reduced, y_train,\n    param_name=\"regressor__alpha\",\n    param_range=alphas,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\n# Convertir a RMSE\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\n# Graficar curva de validaci\u00f3n\nplt.figure(figsize=(8, 5))\nplt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(alphas, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Alpha (par\u00e1metro de regularizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de Validaci\u00f3n - Lasso')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión Lasso
            #######################################

            from sklearn.linear_model import Lasso # Import Lasso
            from sklearn.model_selection import validation_curve # Import validation_curve

            lasso_pipeline = Pipeline([
                ("regressor", Lasso(max_iter=10000))  # Aumentar iteraciones si hay warning
            ])

            # Rango de valores para alpha
            alphas = np.logspace(-4, 1, 10)

            # Validación cruzada sobre X_train e y_train
            train_scores, val_scores = validation_curve(
                lasso_pipeline, X_train_reduced, y_train,
                param_name="regressor__alpha",
                param_range=alphas,
                scoring="neg_mean_squared_error",
                cv=5
            )

            # Convertir a RMSE
            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            # Graficar curva de validación
            plt.figure(figsize=(8, 5))
            plt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(alphas, val_rmse, 'o-', label='Validación')
            plt.xlabel('Alpha (parámetro de regularización)')
            plt.ylabel('RMSE')
            plt.title('Curva de Validación - Lasso')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 51: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 52")
    if show_code:
        st.code("# Convertir a RMSE\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\n# Encontrar el mejor alpha\nbest_alpha = alphas[np.argmin(val_rmse)]\nprint(f\"Mejor alpha seg\u00fan validaci\u00f3n cruzada: {best_alpha}\")\n\n# Usamos el mejor alpha si ya lo tienes, o define uno razonable\nlasso_pipeline.set_params(regressor__alpha=best_alpha)\n\ntrain_sizes, train_scores, val_scores = learning_curve(\n    lasso_pipeline, X_train_reduced, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 5),\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse, 'o-', label='Entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de Aprendizaje - Lasso')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Convertir a RMSE
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            # Encontrar el mejor alpha
            best_alpha = alphas[np.argmin(val_rmse)]
            print(f"Mejor alpha según validación cruzada: {best_alpha}")

            # Usamos el mejor alpha si ya lo tienes, o define uno razonable
            lasso_pipeline.set_params(regressor__alpha=best_alpha)

            train_sizes, train_scores, val_scores = learning_curve(
                lasso_pipeline, X_train_reduced, y_train,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse, 'o-', label='Entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', label='Validación')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de Aprendizaje - Lasso')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 52: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 53")
    if show_code:
        st.code("# Entrenar modelo final con el mejor alpha\nlasso_pipeline.fit(X_train_reduced, y_train)\n\n# Evaluar sobre test\ny_pred = lasso_pipeline.predict(X_test_reduced)\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Entrenar modelo final con el mejor alpha
            lasso_pipeline.fit(X_train_reduced, y_train)

            # Evaluar sobre test
            y_pred = lasso_pipeline.predict(X_test_reduced)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 53: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 54")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n Ridge\n#######################################\nfrom sklearn.linear_model import Ridge # Import Ridge\nfrom sklearn.model_selection import validation_curve\n\nridge_pipeline = Pipeline([\n    (\"regressor\", Ridge())\n])\n\nalphas = np.logspace(-4, 4, 10)\ntrain_scores, val_scores = validation_curve(\n    ridge_pipeline, X_train_reduced, y_train,\n    param_name=\"regressor__alpha\",\n    param_range=alphas,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(alphas, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Alpha (par\u00e1metro de regularizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de validaci\u00f3n - Ridge')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión Ridge
            #######################################
            from sklearn.linear_model import Ridge # Import Ridge
            from sklearn.model_selection import validation_curve

            ridge_pipeline = Pipeline([
                ("regressor", Ridge())
            ])

            alphas = np.logspace(-4, 4, 10)
            train_scores, val_scores = validation_curve(
                ridge_pipeline, X_train_reduced, y_train,
                param_name="regressor__alpha",
                param_range=alphas,
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(alphas, val_rmse, 'o-', label='Validación')
            plt.xlabel('Alpha (parámetro de regularización)')
            plt.ylabel('RMSE')
            plt.title('Curva de validación - Ridge')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 54: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 55")
    if show_code:
        st.code("best_alpha = alphas[np.argmin(val_rmse)]\nprint(f\" Mejor alpha seg\u00fan validaci\u00f3n cruzada: {best_alpha}\")\n\n# Curva de aprendizaje con alpha fijo\nridge_pipeline.set_params(regressor__alpha=best_alpha)\n\ntrain_sizes, train_scores_l, val_scores_l = learning_curve(\n    ridge_pipeline, X_train_reduced, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 5),\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse_l = np.sqrt(-train_scores_l.mean(axis=1))\nval_rmse_l = np.sqrt(-val_scores_l.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse_l, 'o-', label='Entrenamiento')\nplt.plot(train_sizes, val_rmse_l, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de aprendizaje - Ridge')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            best_alpha = alphas[np.argmin(val_rmse)]
            print(f" Mejor alpha según validación cruzada: {best_alpha}")

            # Curva de aprendizaje con alpha fijo
            ridge_pipeline.set_params(regressor__alpha=best_alpha)

            train_sizes, train_scores_l, val_scores_l = learning_curve(
                ridge_pipeline, X_train_reduced, y_train,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse_l = np.sqrt(-train_scores_l.mean(axis=1))
            val_rmse_l = np.sqrt(-val_scores_l.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse_l, 'o-', label='Entrenamiento')
            plt.plot(train_sizes, val_rmse_l, 'o-', label='Validación')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de aprendizaje - Ridge')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 55: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 56")
    if show_code:
        st.code("ridge_pipeline.fit(X_train_reduced, y_train)\n\ny_pred = ridge_pipeline.predict(X_test_reduced)\n\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ridge_pipeline.fit(X_train_reduced, y_train)

            y_pred = ridge_pipeline.predict(X_test_reduced)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 56: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 57")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n Ridge + Polin\u00f3mica\n#######################################\nfrom sklearn.preprocessing import PolynomialFeatures\n\npoly_ridge_pipeline = Pipeline([\n    (\"poly\", PolynomialFeatures(degree=2, include_bias=False)),\n    (\"regressor\", Ridge())\n])\n\nalphas = np.logspace(-4, 2, 10)\n\ntrain_scores, val_scores = validation_curve(\n    poly_ridge_pipeline, X_train_reduced, y_train,\n    param_name=\"regressor__alpha\",\n    param_range=alphas,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(alphas, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Alpha (par\u00e1metro de regularizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de validaci\u00f3n - Ridge + Polinomial')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()\n\n# \ud83d\udd0d Mejor alpha\nbest_alpha = alphas[np.argmin(val_rmse)]\nprint(f\"\ud83d\udd0d Mejor alpha seg\u00fan validaci\u00f3n cruzada: {best_alpha}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión Ridge + Polinómica
            #######################################
            from sklearn.preprocessing import PolynomialFeatures

            poly_ridge_pipeline = Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("regressor", Ridge())
            ])

            alphas = np.logspace(-4, 2, 10)

            train_scores, val_scores = validation_curve(
                poly_ridge_pipeline, X_train_reduced, y_train,
                param_name="regressor__alpha",
                param_range=alphas,
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(alphas, val_rmse, 'o-', label='Validación')
            plt.xlabel('Alpha (parámetro de regularización)')
            plt.ylabel('RMSE')
            plt.title('Curva de validación - Ridge + Polinomial')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # 🔍 Mejor alpha
            best_alpha = alphas[np.argmin(val_rmse)]
            print(f"🔍 Mejor alpha según validación cruzada: {best_alpha}")
    except Exception as e:
        st.error(f"Error en la celda 57: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 58")
    if show_code:
        st.code("poly_ridge_pipeline.set_params(regressor__alpha=best_alpha)\n\ntrain_sizes, train_scores_lc, val_scores_lc = learning_curve(\n    poly_ridge_pipeline, X_train_reduced, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=5,\n    scoring=\"neg_mean_squared_error\"\n)\n\ntrain_rmse_lc = np.sqrt(-train_scores_lc.mean(axis=1))\nval_rmse_lc = np.sqrt(-val_scores_lc.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse_lc, 'o-', label=\"Entrenamiento\")\nplt.plot(train_sizes, val_rmse_lc, 'o-', label=\"Validaci\u00f3n\")\nplt.xlabel(\"Tama\u00f1o del conjunto de entrenamiento\")\nplt.ylabel(\"RMSE\")\nplt.title(\"Curva de aprendizaje - Ridge + Polinomial\")\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            poly_ridge_pipeline.set_params(regressor__alpha=best_alpha)

            train_sizes, train_scores_lc, val_scores_lc = learning_curve(
                poly_ridge_pipeline, X_train_reduced, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                scoring="neg_mean_squared_error"
            )

            train_rmse_lc = np.sqrt(-train_scores_lc.mean(axis=1))
            val_rmse_lc = np.sqrt(-val_scores_lc.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse_lc, 'o-', label="Entrenamiento")
            plt.plot(train_sizes, val_rmse_lc, 'o-', label="Validación")
            plt.xlabel("Tamaño del conjunto de entrenamiento")
            plt.ylabel("RMSE")
            plt.title("Curva de aprendizaje - Ridge + Polinomial")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 58: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 59")
    if show_code:
        st.code("poly_ridge_pipeline.fit(X_train_reduced, y_train)\ny_pred = poly_ridge_pipeline.predict(X_test_reduced)\n\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            poly_ridge_pipeline.fit(X_train_reduced, y_train)
            y_pred = poly_ridge_pipeline.predict(X_test_reduced)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 59: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 60")
    if show_code:
        st.code("########################################\n#ElasticNet\n#######################################\nfrom sklearn.linear_model import ElasticNet # Import ElasticNet\n\nelasticnet_pipeline = Pipeline([\n    (\"poly\", PolynomialFeatures(degree=2, include_bias=False)),\n    (\"regressor\", ElasticNet(max_iter=10000))  # aumentar iteraciones por estabilidad\n])\n\n\nalphas = np.logspace(-4, 1, 10)\n\ntrain_scores, val_scores = validation_curve(\n    elasticnet_pipeline, X_train_reduced, y_train,\n    param_name=\"regressor__alpha\",\n    param_range=alphas,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(alphas, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Alpha (par\u00e1metro de regularizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de validaci\u00f3n - ElasticNet + Polinomial')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()\n\n# Mejor alpha\nbest_alpha = alphas[np.argmin(val_rmse)]\nprint(f\"\ud83d\udd0d Mejor alpha seg\u00fan validaci\u00f3n cruzada: {best_alpha}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #ElasticNet
            #######################################
            from sklearn.linear_model import ElasticNet # Import ElasticNet

            elasticnet_pipeline = Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("regressor", ElasticNet(max_iter=10000))  # aumentar iteraciones por estabilidad
            ])


            alphas = np.logspace(-4, 1, 10)

            train_scores, val_scores = validation_curve(
                elasticnet_pipeline, X_train_reduced, y_train,
                param_name="regressor__alpha",
                param_range=alphas,
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(alphas, val_rmse, 'o-', label='Validación')
            plt.xlabel('Alpha (parámetro de regularización)')
            plt.ylabel('RMSE')
            plt.title('Curva de validación - ElasticNet + Polinomial')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Mejor alpha
            best_alpha = alphas[np.argmin(val_rmse)]
            print(f"🔍 Mejor alpha según validación cruzada: {best_alpha}")
    except Exception as e:
        st.error(f"Error en la celda 60: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 61")
    if show_code:
        st.code("elasticnet_pipeline.set_params(regressor__alpha=best_alpha)\n\ntrain_sizes, train_scores_lc, val_scores_lc = learning_curve(\n    elasticnet_pipeline, X_train_reduced, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=5,\n    scoring=\"neg_mean_squared_error\"\n)\n\ntrain_rmse_lc = np.sqrt(-train_scores_lc.mean(axis=1))\nval_rmse_lc = np.sqrt(-val_scores_lc.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse_lc, 'o-', label=\"Entrenamiento\")\nplt.plot(train_sizes, val_rmse_lc, 'o-', label=\"Validaci\u00f3n\")\nplt.xlabel(\"Tama\u00f1o del conjunto de entrenamiento\")\nplt.ylabel(\"RMSE\")\nplt.title(\"Curva de aprendizaje - ElasticNet + Polinomial\")\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            elasticnet_pipeline.set_params(regressor__alpha=best_alpha)

            train_sizes, train_scores_lc, val_scores_lc = learning_curve(
                elasticnet_pipeline, X_train_reduced, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                scoring="neg_mean_squared_error"
            )

            train_rmse_lc = np.sqrt(-train_scores_lc.mean(axis=1))
            val_rmse_lc = np.sqrt(-val_scores_lc.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse_lc, 'o-', label="Entrenamiento")
            plt.plot(train_sizes, val_rmse_lc, 'o-', label="Validación")
            plt.xlabel("Tamaño del conjunto de entrenamiento")
            plt.ylabel("RMSE")
            plt.title("Curva de aprendizaje - ElasticNet + Polinomial")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 61: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 62")
    if show_code:
        st.code("elasticnet_pipeline.fit(X_train_reduced, y_train)\ny_pred = elasticnet_pipeline.predict(X_test_reduced)\n\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            elasticnet_pipeline.fit(X_train_reduced, y_train)
            y_pred = elasticnet_pipeline.predict(X_test_reduced)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 62: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 63")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n SVR\n#######################################\nfrom sklearn.svm import SVR # Import SVR\n\nsvr_pipeline = Pipeline([\n    (\"regressor\", SVR(kernel='rbf'))\n])\n\n# Curva de validaci\u00f3n para encontrar el mejor C\nC_values = np.logspace(-2, 2, 8)  # valores de C de 0.01 a 100\ntrain_scores, val_scores = validation_curve(\n    svr_pipeline, X_train_reduced, y_train,\n    param_name=\"regressor__C\",\n    param_range=C_values,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\n# Graficar curva de validaci\u00f3n\nplt.figure(figsize=(8, 5))\nplt.semilogx(C_values, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(C_values, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('C (par\u00e1metro de penalizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de Validaci\u00f3n - SVR')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()\n\n# Encontrar mejor C\nbest_C = C_values[np.argmin(val_rmse)]\nprint(f\"Mejor C seg\u00fan validaci\u00f3n cruzada: {best_C}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión SVR
            #######################################
            from sklearn.svm import SVR # Import SVR

            svr_pipeline = Pipeline([
                ("regressor", SVR(kernel='rbf'))
            ])

            # Curva de validación para encontrar el mejor C
            C_values = np.logspace(-2, 2, 8)  # valores de C de 0.01 a 100
            train_scores, val_scores = validation_curve(
                svr_pipeline, X_train_reduced, y_train,
                param_name="regressor__C",
                param_range=C_values,
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            # Graficar curva de validación
            plt.figure(figsize=(8, 5))
            plt.semilogx(C_values, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(C_values, val_rmse, 'o-', label='Validación')
            plt.xlabel('C (parámetro de penalización)')
            plt.ylabel('RMSE')
            plt.title('Curva de Validación - SVR')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Encontrar mejor C
            best_C = C_values[np.argmin(val_rmse)]
            print(f"Mejor C según validación cruzada: {best_C}")
    except Exception as e:
        st.error(f"Error en la celda 63: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 64")
    if show_code:
        st.code("# \u2699\ufe0f Usar el mejor C encontrado previamente\nsvr_pipeline.set_params(regressor__C=best_C)\n\n# Curva de aprendizaje\ntrain_sizes, train_scores, val_scores = learning_curve(\n    svr_pipeline, X_train_reduced, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 8),\n    cv=5,\n    scoring='neg_mean_squared_error',\n    shuffle=True,\n    random_state=42\n)\n\ntrain_rmse = np.sqrt(-train_scores)\nval_rmse = np.sqrt(-val_scores)\n\n# Promedios\ntrain_rmse_mean = train_rmse.mean(axis=1)\nval_rmse_mean = val_rmse.mean(axis=1)\n\n# Graficar curva de aprendizaje\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse_mean, 'o-', label='Entrenamiento')\nplt.plot(train_sizes, val_rmse_mean, 'o-', label='Validaci\u00f3n')\nplt.xlabel('N\u00famero de muestras de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de Aprendizaje - SVR')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # ⚙️ Usar el mejor C encontrado previamente
            svr_pipeline.set_params(regressor__C=best_C)

            # Curva de aprendizaje
            train_sizes, train_scores, val_scores = learning_curve(
                svr_pipeline, X_train_reduced, y_train,
                train_sizes=np.linspace(0.1, 1.0, 8),
                cv=5,
                scoring='neg_mean_squared_error',
                shuffle=True,
                random_state=42
            )

            train_rmse = np.sqrt(-train_scores)
            val_rmse = np.sqrt(-val_scores)

            # Promedios
            train_rmse_mean = train_rmse.mean(axis=1)
            val_rmse_mean = val_rmse.mean(axis=1)

            # Graficar curva de aprendizaje
            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse_mean, 'o-', label='Entrenamiento')
            plt.plot(train_sizes, val_rmse_mean, 'o-', label='Validación')
            plt.xlabel('Número de muestras de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de Aprendizaje - SVR')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 64: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 65")
    if show_code:
        st.code("# \ud83d\udccc Reentrenar modelo con mejor C y evaluar sobre test\nsvr_pipeline.fit(X_train_reduced, y_train)\ny_pred = svr_pipeline.predict(X_test_reduced)\n\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # 📌 Reentrenar modelo con mejor C y evaluar sobre test
            svr_pipeline.fit(X_train_reduced, y_train)
            y_pred = svr_pipeline.predict(X_test_reduced)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 65: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 66")
    if show_code:
        st.code("from sklearn.ensemble import RandomForestRegressor\nfrom sklearn.model_selection import GridSearchCV, KFold, learning_curve\nfrom sklearn.metrics import mean_squared_error, r2_score\n\nmodel = RandomForestRegressor(random_state=42)\n\nparam_grid = {\n    'n_estimators': [50, 100, 200],\n    'max_features': ['sqrt', 'log2', None],\n    'max_depth': [None, 10, 20]\n}\n\ncv = KFold(n_splits=5, shuffle=True, random_state=42)\n\ngs = GridSearchCV(\n    estimator=model,\n    param_grid=param_grid,\n    cv=cv,\n    scoring='neg_root_mean_squared_error',\n    n_jobs=-1\n)\n\ngs.fit(X_train_reduced, y_train)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import GridSearchCV, KFold, learning_curve
            from sklearn.metrics import mean_squared_error, r2_score

            model = RandomForestRegressor(random_state=42)

            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [None, 10, 20]
            }

            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )

            gs.fit(X_train_reduced, y_train)
    except Exception as e:
        st.error(f"Error en la celda 66: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 67")
    if show_code:
        st.code("# Obtener el mejor modelo ya optimizado\nbest_rf_model = gs.best_estimator_\n\n# Imprimir los resultados de la b\u00fasqueda\nprint(\"=== Random Forest - Mejores HP ===\")\nprint(\"Mejores hiperpar\u00e1metros:\", gs.best_params_)\nprint(f\"Mejor RMSE CV: {-gs.best_score_:.4f}\")\n\n# Evaluaci\u00f3n final en el conjunto de prueba\ny_pred_rf = best_rf_model.predict(X_test_reduced)\nrmse_test_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))\nr2_test_rf = r2_score(y_test, y_pred_rf)\nrmse_rel_test_rf = rmse_test_rf / y_test.mean()\n\nprint(f'\ud83d\udd0e RMSE: {rmse_test_rf:.3f}')\nprint(f'\ud83d\udd0e R\u00b2: {r2_test_rf:.3f}')\nprint(f'\ud83d\udd0e RMSE relativo (test): {rmse_rel_test_rf:.3f}')\n\n# --- Curva de Aprendizaje ---\n# **Esta secci\u00f3n debe ir aqu\u00ed, despu\u00e9s de gs.fit()**\ntrain_sizes, train_scores, val_scores = learning_curve(\n    best_rf_model,\n    X_train_reduced, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=cv,\n    scoring='neg_mean_squared_error',\n    n_jobs=-1,\n    random_state=42\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(10, 6))\nplt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Error de Entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Error de Validaci\u00f3n')\n\nplt.title('Curva de Aprendizaje - Random Forest')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.legend(loc='best')\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Obtener el mejor modelo ya optimizado
            best_rf_model = gs.best_estimator_

            # Imprimir los resultados de la búsqueda
            print("=== Random Forest - Mejores HP ===")
            print("Mejores hiperparámetros:", gs.best_params_)
            print(f"Mejor RMSE CV: {-gs.best_score_:.4f}")

            # Evaluación final en el conjunto de prueba
            y_pred_rf = best_rf_model.predict(X_test_reduced)
            rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
            r2_test_rf = r2_score(y_test, y_pred_rf)
            rmse_rel_test_rf = rmse_test_rf / y_test.mean()

            print(f'🔎 RMSE: {rmse_test_rf:.3f}')
            print(f'🔎 R²: {r2_test_rf:.3f}')
            print(f'🔎 RMSE relativo (test): {rmse_rel_test_rf:.3f}')

            # --- Curva de Aprendizaje ---
            # **Esta sección debe ir aquí, después de gs.fit()**
            train_sizes, train_scores, val_scores = learning_curve(
                best_rf_model,
                X_train_reduced, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Error de Entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Error de Validación')

            plt.title('Curva de Aprendizaje - Random Forest')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 67: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 68")
    if show_code:
        st.code("from sklearn.ensemble import GradientBoostingRegressor\nfrom sklearn.model_selection import RandomizedSearchCV, KFold\nfrom sklearn.metrics import mean_squared_error\n\n# 1. Configuraci\u00f3n del modelo y de los hiperpar\u00e1metros a buscar\nmodel_gb = GradientBoostingRegressor(random_state=42)\n\n# Par\u00e1metros para la b\u00fasqueda en cuadr\u00edcula aleatoria (Randomized Search)\nparam_dist_gb = {\n    'n_estimators': [100, 200, 300, 400],         # N\u00famero de \u00e1rboles\n    'learning_rate': [0.01, 0.05, 0.1, 0.2],    # Tasa de aprendizaje\n    'max_depth': [3, 4, 5, 6],                     # Profundidad m\u00e1xima de cada \u00e1rbol\n    'subsample': [0.6, 0.8, 1.0],                  # Fracci\u00f3n de muestras para cada \u00e1rbol\n    'max_features': ['sqrt', 'log2', None]         # Caracter\u00edsticas a considerar\n}\n\n# 2. Configuraci\u00f3n de la validaci\u00f3n cruzada\ncv = KFold(n_splits=5, shuffle=True, random_state=42)\n\n# 3. B\u00fasqueda de los mejores hiperpar\u00e1metros con RandomizedSearchCV\ngs_gb = RandomizedSearchCV(\n    estimator=model_gb,\n    param_distributions=param_dist_gb,\n    n_iter=50,  # N\u00famero de combinaciones a probar\n    cv=cv,\n    scoring='neg_root_mean_squared_error',\n    n_jobs=-1,  # Usa todos los n\u00facleos disponibles\n    random_state=42,\n    refit=True\n)\n\n# Entrenar el modelo\ngs_gb.fit(X_train_reduced, y_train)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import RandomizedSearchCV, KFold
            from sklearn.metrics import mean_squared_error

            # 1. Configuración del modelo y de los hiperparámetros a buscar
            model_gb = GradientBoostingRegressor(random_state=42)

            # Parámetros para la búsqueda en cuadrícula aleatoria (Randomized Search)
            param_dist_gb = {
                'n_estimators': [100, 200, 300, 400],         # Número de árboles
                'learning_rate': [0.01, 0.05, 0.1, 0.2],    # Tasa de aprendizaje
                'max_depth': [3, 4, 5, 6],                     # Profundidad máxima de cada árbol
                'subsample': [0.6, 0.8, 1.0],                  # Fracción de muestras para cada árbol
                'max_features': ['sqrt', 'log2', None]         # Características a considerar
            }

            # 2. Configuración de la validación cruzada
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            # 3. Búsqueda de los mejores hiperparámetros con RandomizedSearchCV
            gs_gb = RandomizedSearchCV(
                estimator=model_gb,
                param_distributions=param_dist_gb,
                n_iter=50,  # Número de combinaciones a probar
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,  # Usa todos los núcleos disponibles
                random_state=42,
                refit=True
            )

            # Entrenar el modelo
            gs_gb.fit(X_train_reduced, y_train)
    except Exception as e:
        st.error(f"Error en la celda 68: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 69")
    if show_code:
        st.code("# 4. Obtener el mejor modelo y sus resultados\nbest_gb_model = gs_gb.best_estimator_\n\nprint(\"=== Gradient Boosting - Mejores HP ===\")\nprint(\"Mejores hiperpar\u00e1metros:\", gs_gb.best_params_)\nprint(f\"Mejor RMSE CV: {-gs_gb.best_score_:.4f}\")\n\n# 5. Evaluaci\u00f3n final en el conjunto de prueba\ny_pred_gb = best_gb_model.predict(X_test_reduced)\nrmse_test_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))\nr2_test_gb = r2_score(y_test, y_pred_gb)\nrmse_rel_test_gb = rmse_test_gb / y_test.mean()\n\nprint(f'\ud83d\udd0e RMSE: {rmse_test_gb:.3f}')\nprint(f'\ud83d\udd0e R\u00b2: {r2_test_gb:.3f}')\nprint(f'\ud83d\udd0e RMSE relativo (test): {rmse_rel_test_gb:.3f}')\n# --- Curva de Aprendizaje ---\ntrain_sizes, train_scores, val_scores = learning_curve(\n    best_gb_model,\n    X_train_reduced, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=cv,\n    scoring='neg_mean_squared_error', # Cambiar a neg_mean_squared_error\n    n_jobs=-1,\n    random_state=42\n)\n\n# Ahora toma la ra\u00edz cuadrada de las puntuaciones para obtener RMSE\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(10, 6))\nplt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Error de Entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Error de Validaci\u00f3n')\n\nplt.title('Curva de Aprendizaje - Gradient Boosting')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.legend(loc='best')\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # 4. Obtener el mejor modelo y sus resultados
            best_gb_model = gs_gb.best_estimator_

            print("=== Gradient Boosting - Mejores HP ===")
            print("Mejores hiperparámetros:", gs_gb.best_params_)
            print(f"Mejor RMSE CV: {-gs_gb.best_score_:.4f}")

            # 5. Evaluación final en el conjunto de prueba
            y_pred_gb = best_gb_model.predict(X_test_reduced)
            rmse_test_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
            r2_test_gb = r2_score(y_test, y_pred_gb)
            rmse_rel_test_gb = rmse_test_gb / y_test.mean()

            print(f'🔎 RMSE: {rmse_test_gb:.3f}')
            print(f'🔎 R²: {r2_test_gb:.3f}')
            print(f'🔎 RMSE relativo (test): {rmse_rel_test_gb:.3f}')
            # --- Curva de Aprendizaje ---
            train_sizes, train_scores, val_scores = learning_curve(
                best_gb_model,
                X_train_reduced, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=cv,
                scoring='neg_mean_squared_error', # Cambiar a neg_mean_squared_error
                n_jobs=-1,
                random_state=42
            )

            # Ahora toma la raíz cuadrada de las puntuaciones para obtener RMSE
            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Error de Entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Error de Validación')

            plt.title('Curva de Aprendizaje - Gradient Boosting')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 69: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 70")
    if show_code:
        st.code("# 1. Filtrar solo columnas num\u00e9ricas continuas de tu lista df_numericas\nX_train_num = X_train[num_features].copy()\n\n# 2. Unir con y_train\ntrain_num_with_target = X_train_num.copy()\ntrain_num_with_target[\"DURATION OF STAY\"] = y_train\n\n# 3. Calcular Spearman solo para num\u00e9ricas continuas\ncorrelaciones = train_num_with_target.corr(method='spearman')[\"DURATION OF STAY\"]\n\n# 4. Ordenar de mayor a menor por valor absoluto\ncorrelaciones_ordenadas = correlaciones.reindex(\n    correlaciones.abs().sort_values(ascending=False).index\n)\n\n# 5. Imprimir todas\nprint(\"=== Correlaciones de Spearman (solo num\u00e9ricas continuas) ===\")\nfor col, valor in correlaciones_ordenadas.items():\n    print(f\"{col}: {valor:.4f}\")\n\n# 6. Calcular porcentaje y acumulado\ndf_corr = correlaciones_ordenadas.drop(\"DURATION OF STAY\").to_frame(name=\"correlation\")\ndf_corr[\"abs_corr\"] = df_corr[\"correlation\"].abs()\ndf_corr[\"percentage\"] = df_corr[\"abs_corr\"] / df_corr[\"abs_corr\"].sum() * 100\ndf_corr[\"cum_percentage\"] = df_corr[\"percentage\"].cumsum()\n\n\n# 7. Filtrar por porcentaje acumulado\numbral_acumulado = 90\nnumericas_significativas = df_corr[df_corr[\"cum_percentage\"] <= umbral_acumulado].index.tolist()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # 1. Filtrar solo columnas numéricas continuas de tu lista df_numericas
            X_train_num = X_train[num_features].copy()

            # 2. Unir con y_train
            train_num_with_target = X_train_num.copy()
            train_num_with_target["DURATION OF STAY"] = y_train

            # 3. Calcular Spearman solo para numéricas continuas
            correlaciones = train_num_with_target.corr(method='spearman')["DURATION OF STAY"]

            # 4. Ordenar de mayor a menor por valor absoluto
            correlaciones_ordenadas = correlaciones.reindex(
                correlaciones.abs().sort_values(ascending=False).index
            )

            # 5. Imprimir todas
            print("=== Correlaciones de Spearman (solo numéricas continuas) ===")
            for col, valor in correlaciones_ordenadas.items():
                print(f"{col}: {valor:.4f}")

            # 6. Calcular porcentaje y acumulado
            df_corr = correlaciones_ordenadas.drop("DURATION OF STAY").to_frame(name="correlation")
            df_corr["abs_corr"] = df_corr["correlation"].abs()
            df_corr["percentage"] = df_corr["abs_corr"] / df_corr["abs_corr"].sum() * 100
            df_corr["cum_percentage"] = df_corr["percentage"].cumsum()


            # 7. Filtrar por porcentaje acumulado
            umbral_acumulado = 90
            numericas_significativas = df_corr[df_corr["cum_percentage"] <= umbral_acumulado].index.tolist()
    except Exception as e:
        st.error(f"Error en la celda 70: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 71")
    if show_code:
        st.code("plt.figure(figsize=(10, 6))\nplt.bar(df_corr.index, df_corr[\"percentage\"], label=\"Porcentaje individual\")\nplt.plot(df_corr.index, df_corr[\"cum_percentage\"], color=\"red\", marker=\"o\", label=\"Porcentaje acumulado\")\nplt.axhline(umbral_acumulado, color=\"green\", linestyle=\"--\", label=f\"Umbral {umbral_acumulado}%\")\nplt.xticks(rotation=90)\nplt.ylabel(\"Porcentaje (%)\")\nplt.title(\"Importancia por Spearman y acumulado\")\nplt.legend()\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            plt.figure(figsize=(10, 6))
            plt.bar(df_corr.index, df_corr["percentage"], label="Porcentaje individual")
            plt.plot(df_corr.index, df_corr["cum_percentage"], color="red", marker="o", label="Porcentaje acumulado")
            plt.axhline(umbral_acumulado, color="green", linestyle="--", label=f"Umbral {umbral_acumulado}%")
            plt.xticks(rotation=90)
            plt.ylabel("Porcentaje (%)")
            plt.title("Importancia por Spearman y acumulado")
            plt.legend()
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 71: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 72")
    if show_code:
        st.code("print(\"\\n Variables num\u00e9ricas seleccionadas (por 90% acumulado):\")\nprint(numericas_significativas)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            print("\n Variables numéricas seleccionadas (por 90% acumulado):")
            print(numericas_significativas)
    except Exception as e:
        st.error(f"Error en la celda 72: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 73")
    if show_code:
        st.code("X_train_cat = X_train[cat_features].copy()\n\nsignificativas = []\n\nfor col in X_train_cat:\n    grupos = [\n        df[df[col] == categoria]['DURATION OF STAY'].dropna()\n        for categoria in df[col].unique()\n    ]\n    f_stat, p_val = stats.f_oneway(*grupos)\n\n    # Verificamos si es estad\u00edsticamente significativa\n    if p_val < 0.05:\n        significativas.append(col)\n\n    print(f\"{col}: F={f_stat:.3f}, p-value={p_val:.4f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            X_train_cat = X_train[cat_features].copy()

            significativas = []

            for col in X_train_cat:
                grupos = [
                    df[df[col] == categoria]['DURATION OF STAY'].dropna()
                    for categoria in df[col].unique()
                ]
                f_stat, p_val = stats.f_oneway(*grupos)

                # Verificamos si es estadísticamente significativa
                if p_val < 0.05:
                    significativas.append(col)

                print(f"{col}: F={f_stat:.3f}, p-value={p_val:.4f}")
    except Exception as e:
        st.error(f"Error en la celda 73: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 74")
    if show_code:
        st.code("print(\"\\n Variables categ\u00f3ricas significativas (p < 0.05):\")\nprint(significativas)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            print("\n Variables categóricas significativas (p < 0.05):")
            print(significativas)
    except Exception as e:
        st.error(f"Error en la celda 74: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 75")
    if show_code:
        st.code("var_seleccionadas= numericas_significativas + significativas\n\n# Filtrar el preprocesado, no el crudo\nX_train_filtrado = pd.DataFrame(\n    X_train_processed,  # array preprocesado\n    columns=num_features + cat_features  # columnas despu\u00e9s del preprocesador\n)[var_seleccionadas]\n\nX_test_filtrado = pd.DataFrame(\n    X_test_processed,\n    columns=num_features + cat_features\n)[var_seleccionadas]\n\n# Ahora s\u00ed, aplicar SelectKBest\nfrom sklearn.feature_selection import SelectKBest, f_regression\n\nselector = SelectKBest(score_func=f_regression, k='all')\nselector.fit(X_train_filtrado, y_train)\n\nscores = selector.scores_\nranking = sorted(zip(var_seleccionadas, scores), key=lambda x: x[1], reverse=True)\n\ndf_scores = pd.DataFrame(ranking, columns=[\"Variable\", \"Score\"])\ndf_scores[\"Perc\"] = (df_scores[\"Score\"] / df_scores[\"Score\"].sum()) * 100\ndf_scores[\"CumPerc\"] = df_scores[\"Perc\"].cumsum()\n\n# Mostrar\nprint(df_scores)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            var_seleccionadas= numericas_significativas + significativas

            # Filtrar el preprocesado, no el crudo
            X_train_filtrado = pd.DataFrame(
                X_train_processed,  # array preprocesado
                columns=num_features + cat_features  # columnas después del preprocesador
            )[var_seleccionadas]

            X_test_filtrado = pd.DataFrame(
                X_test_processed,
                columns=num_features + cat_features
            )[var_seleccionadas]

            # Ahora sí, aplicar SelectKBest
            from sklearn.feature_selection import SelectKBest, f_regression

            selector = SelectKBest(score_func=f_regression, k='all')
            selector.fit(X_train_filtrado, y_train)

            scores = selector.scores_
            ranking = sorted(zip(var_seleccionadas, scores), key=lambda x: x[1], reverse=True)

            df_scores = pd.DataFrame(ranking, columns=["Variable", "Score"])
            df_scores["Perc"] = (df_scores["Score"] / df_scores["Score"].sum()) * 100
            df_scores["CumPerc"] = df_scores["Perc"].cumsum()

            # Mostrar
            print(df_scores)
    except Exception as e:
        st.error(f"Error en la celda 75: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 76")
    if show_code:
        st.code("\nfig, ax1 = plt.subplots(figsize=(10,6))\n\ncolor = \"skyblue\"\nax1.bar(df_scores[\"Variable\"], df_scores[\"Score\"], color=color)\nax1.set_xlabel(\"Variables\")\nax1.set_ylabel(\"Puntaje F-test\", color=color)\nax1.tick_params(axis=\"y\", labelcolor=color)\nax1.set_xticklabels(df_scores[\"Variable\"], rotation=45, ha=\"right\")\n\nax2 = ax1.twinx()\ncolor = \"crimson\"\nax2.plot(df_scores[\"Variable\"], df_scores[\"CumPerc\"], color=color, marker=\"o\")\nax2.set_ylabel(\"Porcentaje acumulado\", color=color)\nax2.tick_params(axis=\"y\", labelcolor=color)\nax2.set_ylim(0, 110)\n\n# 5. L\u00ednea de referencia al o 90%\nax2.axhline(y=90, color=\"gray\", linestyle=\"--\", linewidth=1)\nax2.text(len(df_scores)-1, 82, \"90% L\u00edmite\", color=\"gray\")\n\nplt.title(\"SelectKBest - Porcentaje de importancia acumulada\")\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):

            fig, ax1 = plt.subplots(figsize=(10,6))

            color = "skyblue"
            ax1.bar(df_scores["Variable"], df_scores["Score"], color=color)
            ax1.set_xlabel("Variables")
            ax1.set_ylabel("Puntaje F-test", color=color)
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.set_xticklabels(df_scores["Variable"], rotation=45, ha="right")

            ax2 = ax1.twinx()
            color = "crimson"
            ax2.plot(df_scores["Variable"], df_scores["CumPerc"], color=color, marker="o")
            ax2.set_ylabel("Porcentaje acumulado", color=color)
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.set_ylim(0, 110)

            # 5. Línea de referencia al o 90%
            ax2.axhline(y=90, color="gray", linestyle="--", linewidth=1)
            ax2.text(len(df_scores)-1, 82, "90% Límite", color="gray")

            plt.title("SelectKBest - Porcentaje de importancia acumulada")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 76: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 77")
    if show_code:
        st.code("# Filtrar hasta el 90%\nvars_90 = df_scores[df_scores[\"CumPerc\"] <= 90]['Variable'].tolist()\n\nprint(f\"Variables que acumulan el 90%: {vars_90}\")\nprint(f\"Cantidad: {len(vars_90)}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Filtrar hasta el 90%
            vars_90 = df_scores[df_scores["CumPerc"] <= 90]['Variable'].tolist()

            print(f"Variables que acumulan el 90%: {vars_90}")
            print(f"Cantidad: {len(vars_90)}")
    except Exception as e:
        st.error(f"Error en la celda 77: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 78")
    if show_code:
        st.code("from sklearn.feature_selection import RFECV\nfrom sklearn.metrics import mean_squared_error\nfrom sklearn.linear_model import RidgeCV\nfrom sklearn.model_selection import KFold\n\nmodelo_rfe = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])\n\nrfecv = RFECV(\n    estimator=modelo_rfe,\n    step=1,\n    cv=KFold(n_splits=5, shuffle=True, random_state=42),\n    scoring=\"neg_root_mean_squared_error\"\n)\n\nrfecv.fit(X_train_filtrado, y_train)\n\n# Features seleccionadas\nmask = rfecv.support_\nselected_features_rfecv = X_train_filtrado.columns[mask]\n\nprint(\"\\n Variables seleccionadas por RFECV:\")\nprint(f\"Cantidad: {rfecv.n_features_}\")\nprint(selected_features_rfecv.tolist())\n\n# Evaluaci\u00f3n en test (RMSE)\ny_pred = rfecv.predict(X_test_filtrado)   # RFECV aplica internamente la m\u00e1scara de features\nmse = mean_squared_error(y_test, y_pred)\nrmse = np.sqrt(mse)\nprint(f\"\\nRMSE en test (RFECV + LinearRegression): {rmse:.4f}\")\n\n# (Opcional) Si quieres el ranking completo (1 = seleccionada, 2+ = menos importantes):\n# ranking = rfecv.ranking_\n# df_rank = (pd.DataFrame({\"feature\": X_train_filtrado.columns, \"rank\": ranking})\n#              .sort_values(\"rank\"))\n# print(df_rank.head(30))", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.feature_selection import RFECV
            from sklearn.metrics import mean_squared_error
            from sklearn.linear_model import RidgeCV
            from sklearn.model_selection import KFold

            modelo_rfe = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])

            rfecv = RFECV(
                estimator=modelo_rfe,
                step=1,
                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                scoring="neg_root_mean_squared_error"
            )

            rfecv.fit(X_train_filtrado, y_train)

            # Features seleccionadas
            mask = rfecv.support_
            selected_features_rfecv = X_train_filtrado.columns[mask]

            print("\n Variables seleccionadas por RFECV:")
            print(f"Cantidad: {rfecv.n_features_}")
            print(selected_features_rfecv.tolist())

            # Evaluación en test (RMSE)
            y_pred = rfecv.predict(X_test_filtrado)   # RFECV aplica internamente la máscara de features
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            print(f"\nRMSE en test (RFECV + LinearRegression): {rmse:.4f}")

            # (Opcional) Si quieres el ranking completo (1 = seleccionada, 2+ = menos importantes):
            # ranking = rfecv.ranking_
            # df_rank = (pd.DataFrame({"feature": X_train_filtrado.columns, "rank": ranking})
            #              .sort_values("rank"))
            # print(df_rank.head(30))
    except Exception as e:
        st.error(f"Error en la celda 78: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 79")
    if show_code:
        st.code("# Entrenar modelo\nrf_model = RandomForestRegressor(random_state=42)\nrf_model.fit(X_train_filtrado, y_train)\n\n# Importancia de caracter\u00edsticas\nimportances = rf_model.feature_importances_\nimportances_df = pd.DataFrame({\n    'Variable': X_train_filtrado.columns,\n    'Importancia': importances\n}).sort_values(by='Importancia', ascending=False)\n\n# Calcular porcentaje acumulado\nimportances_df['Importancia_Acum'] = importances_df['Importancia'].cumsum()\n\n# Seleccionar variables hasta el 90%\nselected_vars = importances_df[importances_df['Importancia_Acum'] <= 0.90]['Variable'].tolist()\n\nprint(\"\\n Ranking de importancia con RandomForest:\")\nprint(importances_df)\nprint(\"\\n Variables seleccionadas hasta el 90% de importancia acumulada:\")\nprint(selected_vars)\n\n# Filtrar dataset\nX_train_sel = X_train_filtrado[selected_vars]\nX_test_sel = X_test_filtrado[selected_vars]", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Entrenar modelo
            rf_model = RandomForestRegressor(random_state=42)
            rf_model.fit(X_train_filtrado, y_train)

            # Importancia de características
            importances = rf_model.feature_importances_
            importances_df = pd.DataFrame({
                'Variable': X_train_filtrado.columns,
                'Importancia': importances
            }).sort_values(by='Importancia', ascending=False)

            # Calcular porcentaje acumulado
            importances_df['Importancia_Acum'] = importances_df['Importancia'].cumsum()

            # Seleccionar variables hasta el 90%
            selected_vars = importances_df[importances_df['Importancia_Acum'] <= 0.90]['Variable'].tolist()

            print("\n Ranking de importancia con RandomForest:")
            print(importances_df)
            print("\n Variables seleccionadas hasta el 90% de importancia acumulada:")
            print(selected_vars)

            # Filtrar dataset
            X_train_sel = X_train_filtrado[selected_vars]
            X_test_sel = X_test_filtrado[selected_vars]
    except Exception as e:
        st.error(f"Error en la celda 79: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 80")
    if show_code:
        st.code("# Gr\u00e1fica de importancias y acumulado\nplt.figure(figsize=(10, 6))\n\n# Barras de importancia\nplt.bar(importances_df['Variable'], importances_df['Importancia'], alpha=0.7, label='Importancia individual')\n\n# L\u00ednea de importancia acumulada\nplt.plot(importances_df['Variable'], importances_df['Importancia_Acum'], marker='o', color='red', label='Importancia acumulada')\n\n# L\u00ednea de referencia del 90%\nplt.axhline(0.90, color='green', linestyle='--', label='90% acumulado')\n\n# Formato del gr\u00e1fico\nplt.xticks(rotation=90)\nplt.ylabel('Importancia')\nplt.title('Ranking de importancia de variables - RandomForest')\nplt.legend()\nplt.grid(axis='y', alpha=0.3)\n\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Gráfica de importancias y acumulado
            plt.figure(figsize=(10, 6))

            # Barras de importancia
            plt.bar(importances_df['Variable'], importances_df['Importancia'], alpha=0.7, label='Importancia individual')

            # Línea de importancia acumulada
            plt.plot(importances_df['Variable'], importances_df['Importancia_Acum'], marker='o', color='red', label='Importancia acumulada')

            # Línea de referencia del 90%
            plt.axhline(0.90, color='green', linestyle='--', label='90% acumulado')

            # Formato del gráfico
            plt.xticks(rotation=90)
            plt.ylabel('Importancia')
            plt.title('Ranking de importancia de variables - RandomForest')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 80: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 81")
    if show_code:
        st.code("# Convertimos a conjuntos y sacamos la intersecci\u00f3n\ncomunes = list(set(vars_90) & set(selected_features_rfecv) & set(selected_vars) )\n\nprint(\"Nuevo DataFrame (intersecci\u00f3n):\")\nprint(f\"numero de variables en commun {len(comunes)}\")\nprint(\"Variables en com\u00fan:\", comunes)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Convertimos a conjuntos y sacamos la intersección
            comunes = list(set(vars_90) & set(selected_features_rfecv) & set(selected_vars) )

            print("Nuevo DataFrame (intersección):")
            print(f"numero de variables en commun {len(comunes)}")
            print("Variables en común:", comunes)
    except Exception as e:
        st.error(f"Error en la celda 81: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 82")
    if show_code:
        st.code("# Filtrar DataFrame solo con esas columnas\ndf_filtrado = df[comunes]\nprint(\"\\nDataFrame filtrado:\")\ndf_filtrado.head()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Filtrar DataFrame solo con esas columnas
            df_filtrado = df[comunes]
            print("\nDataFrame filtrado:")
            df_filtrado.head()
    except Exception as e:
        st.error(f"Error en la celda 82: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 83")
    if show_code:
        st.code("X_train_new = X_train_filtrado[comunes].copy()\nX_test_new  = X_test_filtrado[comunes].copy()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            X_train_new = X_train_filtrado[comunes].copy()
            X_test_new  = X_test_filtrado[comunes].copy()
    except Exception as e:
        st.error(f"Error en la celda 83: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 84")
    if show_code:
        st.code("from sklearn.metrics import r2_score\n# ============================\n# PRE-PRUNING con RandomizedSearchCV (solo TRAIN)\n# ============================\nimport numpy as np # Import numpy\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.model_selection import RandomizedSearchCV\nfrom sklearn.metrics import mean_squared_error\n\npipe_pre = Pipeline(steps=[\n    ('model', DecisionTreeRegressor(random_state=42))\n])\n\nparam_dist_pre = {\n    'model__max_depth': np.arange(2, 20),\n    'model__min_samples_split': np.arange(10, 50),\n    'model__min_samples_leaf': np.arange(5, 20),\n    'model__max_features': [None, 'sqrt', 'log2']\n}\n\nsearch_pre = RandomizedSearchCV(\n    estimator=pipe_pre,\n    param_distributions=param_dist_pre,\n    n_iter=60,\n    cv=5,\n    scoring='neg_root_mean_squared_error',\n    n_jobs=-1,\n    random_state=42,\n    refit=True\n)\n\nsearch_pre.fit(X_train_new, y_train)\n\n\nprint(\"\\n=== PRE-PRUNING ===\")\nprint(\"Mejores hiperpar\u00e1metros:\", search_pre.best_params_)\nprint(f\"Mejor RMSE CV: {-search_pre.best_score_:.4f}\")\n\n# Evaluaci\u00f3n final en test (primera vez que tocamos test para pre-pruning)\ny_pred_pre = search_pre.predict(X_test_new)\nrmse_test_pre = np.sqrt(mean_squared_error(y_test, y_pred_pre))\nr2_test_pre = r2_score(y_test, y_pred_pre)\nprint(f\"RMSE Test (pre-pruning): {rmse_test_pre:.4f}\")\nprint(f\"RMSE relativo (pre-pruning): {rmse_test_pre / y_test.mean():.3f}\")\nprint(f\"R\u00b2 (test): {r2_test_pre:.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.metrics import r2_score
            # ============================
            # PRE-PRUNING con RandomizedSearchCV (solo TRAIN)
            # ============================
            import numpy as np # Import numpy
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.metrics import mean_squared_error

            pipe_pre = Pipeline(steps=[
                ('model', DecisionTreeRegressor(random_state=42))
            ])

            param_dist_pre = {
                'model__max_depth': np.arange(2, 20),
                'model__min_samples_split': np.arange(10, 50),
                'model__min_samples_leaf': np.arange(5, 20),
                'model__max_features': [None, 'sqrt', 'log2']
            }

            search_pre = RandomizedSearchCV(
                estimator=pipe_pre,
                param_distributions=param_dist_pre,
                n_iter=60,
                cv=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                random_state=42,
                refit=True
            )

            search_pre.fit(X_train_new, y_train)


            print("\n=== PRE-PRUNING ===")
            print("Mejores hiperparámetros:", search_pre.best_params_)
            print(f"Mejor RMSE CV: {-search_pre.best_score_:.4f}")

            # Evaluación final en test (primera vez que tocamos test para pre-pruning)
            y_pred_pre = search_pre.predict(X_test_new)
            rmse_test_pre = np.sqrt(mean_squared_error(y_test, y_pred_pre))
            r2_test_pre = r2_score(y_test, y_pred_pre)
            print(f"RMSE Test (pre-pruning): {rmse_test_pre:.4f}")
            print(f"RMSE relativo (pre-pruning): {rmse_test_pre / y_test.mean():.3f}")
            print(f"R² (test): {r2_test_pre:.3f}")
    except Exception as e:
        st.error(f"Error en la celda 84: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 85")
    if show_code:
        st.code("from sklearn.tree import DecisionTreeRegressor\nfrom sklearn.model_selection import learning_curve\nimport numpy as np\nimport matplotlib.pyplot as plt\n# 4. Graficar la curva de aprendizaje\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.model_selection import learning_curve\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Modelo base (puedes a\u00f1adir pre-pruning con max_depth, min_samples_leaf, etc.)\nmodel = DecisionTreeRegressor(max_depth=5, random_state=42)\n\n# Calcular curva de aprendizaje\ntrain_sizes, train_scores, val_scores = learning_curve(\n    estimator=model,\n    X=X_train_new, y=y_train,\n    train_sizes=np.linspace(0.1, 1.0, 8),   # 8 puntos de 10% a 100% del train\n    cv=5,\n    scoring=\"neg_root_mean_squared_error\",  # RMSE pero negativo\n    shuffle=True,\n    random_state=42,\n    n_jobs=-1\n)\n\n# Convertir a RMSE positivo\ntrain_rmse = -train_scores.mean(axis=1)\nval_rmse   = -val_scores.mean(axis=1)\n\n# === 4. Graficar la curva ===\nplt.figure(figsize=(10, 6))\nplt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Error de Entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Error de Validaci\u00f3n')\n\nplt.title('Curva de Aprendizaje - \u00c1rbol de Decisi\u00f3n con Pre-Pruning')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.legend(loc='best')\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import learning_curve
            import numpy as np
            import matplotlib.pyplot as plt
            # 4. Graficar la curva de aprendizaje
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import learning_curve
            import numpy as np
            import matplotlib.pyplot as plt

            # Modelo base (puedes añadir pre-pruning con max_depth, min_samples_leaf, etc.)
            model = DecisionTreeRegressor(max_depth=5, random_state=42)

            # Calcular curva de aprendizaje
            train_sizes, train_scores, val_scores = learning_curve(
                estimator=model,
                X=X_train_new, y=y_train,
                train_sizes=np.linspace(0.1, 1.0, 8),   # 8 puntos de 10% a 100% del train
                cv=5,
                scoring="neg_root_mean_squared_error",  # RMSE pero negativo
                shuffle=True,
                random_state=42,
                n_jobs=-1
            )

            # Convertir a RMSE positivo
            train_rmse = -train_scores.mean(axis=1)
            val_rmse   = -val_scores.mean(axis=1)

            # === 4. Graficar la curva ===
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Error de Entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Error de Validación')

            plt.title('Curva de Aprendizaje - Árbol de Decisión con Pre-Pruning')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 85: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 86")
    if show_code:
        st.code("# ============================\n# 5) POST-PRUNING (\u00e1rbol grande -> path alphas -> CV en TRAIN)\n# ============================\n#import numpy as np\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.model_selection import KFold, cross_val_score\nfrom sklearn.metrics import mean_squared_error\n\n# === 0) Cast a float32 si aplica (acelera y reduce memoria)\nX_train_np = np.asarray(X_train_new, dtype=np.float32)\ny_train_np = np.asarray(y_train)  # target puede quedar en float64\nX_test_np  = np.asarray(X_test_new,  dtype=np.float32)\ny_test_np  = np.asarray(y_test)\n\n# === 1) Ruta de poda en un SUBSET\nrng = np.random.RandomState(42)\nn_sub = min(len(X_train_np), 15000)\nidx = rng.choice(len(X_train_np), n_sub, replace=False)\n\ntree_full_sub = DecisionTreeRegressor(random_state=42)\ntree_full_sub.fit(X_train_np[idx], y_train_np[idx])\n\npath = tree_full_sub.cost_complexity_pruning_path(X_train_np[idx], y_train_np[idx])\nalphas_full = np.unique(np.round(path.ccp_alphas, 10))\n\n# === 2) Muestrea ~40 alphas representativos (cuantiles)\nif len(alphas_full) > 40:\n    quantiles = np.linspace(0, 1, 40)\n    ccp_alphas = np.quantile(alphas_full, quantiles)\nelse:\n    ccp_alphas = alphas_full\n\n# === 3) CV paralela y \u00e1rbol con l\u00edmites de complejidad (m\u00e1s r\u00e1pido)\nkf = KFold(n_splits=3, shuffle=True, random_state=42)\n\ndef cv_rmse_for_alpha(a):\n    model = DecisionTreeRegressor(\n        random_state=42,\n        ccp_alpha=a,\n        max_depth=6,\n        min_samples_leaf=11\n    )\n    scores = cross_val_score(\n        model, X_train_np, y_train_np,\n        scoring='neg_root_mean_squared_error',\n        cv=kf, n_jobs=-1\n    )\n    return float((-scores).mean())\n\nrmse_list = [cv_rmse_for_alpha(a) for a in ccp_alphas]\nbest_idx   = int(np.argmin(rmse_list))\nbest_alpha = float(ccp_alphas[best_idx])\n\nprint(\"\\n=== POST-PRUNING ===\")\nprint(f\"Mejor ccp_alpha: {best_alpha:.6f}\")\nprint(f\"RMSE CV={rmse_list[best_idx]:.4f}\")\n\n# === 4) Entrena SOLO el \u00e1rbol final y, si quieres, mide tama\u00f1o una vez\ntree_pruned = DecisionTreeRegressor(\n    random_state=42,\n    ccp_alpha=best_alpha,\n    max_depth=20,\n    min_samples_leaf=5\n)\ntree_pruned.fit(X_train_np, y_train_np)\n\nn_leaves = tree_pruned.get_n_leaves()\nn_nodes  = tree_pruned.tree_.node_count\nprint(f\"Hojas: {n_leaves} | Nodos: {n_nodes}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # ============================
            # 5) POST-PRUNING (árbol grande -> path alphas -> CV en TRAIN)
            # ============================
            #import numpy as np
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.model_selection import KFold, cross_val_score
            from sklearn.metrics import mean_squared_error

            # === 0) Cast a float32 si aplica (acelera y reduce memoria)
            X_train_np = np.asarray(X_train_new, dtype=np.float32)
            y_train_np = np.asarray(y_train)  # target puede quedar en float64
            X_test_np  = np.asarray(X_test_new,  dtype=np.float32)
            y_test_np  = np.asarray(y_test)

            # === 1) Ruta de poda en un SUBSET
            rng = np.random.RandomState(42)
            n_sub = min(len(X_train_np), 15000)
            idx = rng.choice(len(X_train_np), n_sub, replace=False)

            tree_full_sub = DecisionTreeRegressor(random_state=42)
            tree_full_sub.fit(X_train_np[idx], y_train_np[idx])

            path = tree_full_sub.cost_complexity_pruning_path(X_train_np[idx], y_train_np[idx])
            alphas_full = np.unique(np.round(path.ccp_alphas, 10))

            # === 2) Muestrea ~40 alphas representativos (cuantiles)
            if len(alphas_full) > 40:
                quantiles = np.linspace(0, 1, 40)
                ccp_alphas = np.quantile(alphas_full, quantiles)
            else:
                ccp_alphas = alphas_full

            # === 3) CV paralela y árbol con límites de complejidad (más rápido)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)

            def cv_rmse_for_alpha(a):
                model = DecisionTreeRegressor(
                    random_state=42,
                    ccp_alpha=a,
                    max_depth=6,
                    min_samples_leaf=11
                )
                scores = cross_val_score(
                    model, X_train_np, y_train_np,
                    scoring='neg_root_mean_squared_error',
                    cv=kf, n_jobs=-1
                )
                return float((-scores).mean())

            rmse_list = [cv_rmse_for_alpha(a) for a in ccp_alphas]
            best_idx   = int(np.argmin(rmse_list))
            best_alpha = float(ccp_alphas[best_idx])

            print("\n=== POST-PRUNING ===")
            print(f"Mejor ccp_alpha: {best_alpha:.6f}")
            print(f"RMSE CV={rmse_list[best_idx]:.4f}")

            # === 4) Entrena SOLO el árbol final y, si quieres, mide tamaño una vez
            tree_pruned = DecisionTreeRegressor(
                random_state=42,
                ccp_alpha=best_alpha,
                max_depth=20,
                min_samples_leaf=5
            )
            tree_pruned.fit(X_train_np, y_train_np)

            n_leaves = tree_pruned.get_n_leaves()
            n_nodes  = tree_pruned.tree_.node_count
            print(f"Hojas: {n_leaves} | Nodos: {n_nodes}")
    except Exception as e:
        st.error(f"Error en la celda 86: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 87")
    if show_code:
        st.code("from sklearn.model_selection import learning_curve\nimport matplotlib.pyplot as plt\nimport numpy as np\n\n# --- Curva de Aprendizaje con el \u00e1rbol podado ---\ntrain_sizes, train_scores, val_scores = learning_curve(\n    tree_pruned,                     # tu \u00e1rbol ya podado\n    X_train_np, y_train_np,\n    train_sizes=np.linspace(0.1, 1.0, 10),  # desde 10% hasta 100% de los datos\n    cv=5,                             # 5 folds\n    scoring='neg_mean_squared_error',\n    n_jobs=-1\n)\n\n# Convertir MSE negativo a RMSE\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse   = np.sqrt(-val_scores.mean(axis=1))\n\n# --- Gr\u00e1fico ---\nplt.figure(figsize=(8,6))\nplt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Validaci\u00f3n')\n\nplt.title('Curva de Aprendizaje - \u00c1rbol Podado')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.legend(loc='best')\nplt.grid(True)\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.model_selection import learning_curve
            import matplotlib.pyplot as plt
            import numpy as np

            # --- Curva de Aprendizaje con el árbol podado ---
            train_sizes, train_scores, val_scores = learning_curve(
                tree_pruned,                     # tu árbol ya podado
                X_train_np, y_train_np,
                train_sizes=np.linspace(0.1, 1.0, 10),  # desde 10% hasta 100% de los datos
                cv=5,                             # 5 folds
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            # Convertir MSE negativo a RMSE
            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse   = np.sqrt(-val_scores.mean(axis=1))

            # --- Gráfico ---
            plt.figure(figsize=(8,6))
            plt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', color='orange', label='Validación')

            plt.title('Curva de Aprendizaje - Árbol Podado')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.legend(loc='best')
            plt.grid(True)
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 87: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 88")
    if show_code:
        st.code("# ============================\n# 7) Mostrar resultados\n# ============================\n\ny_pred_post = tree_pruned.predict(X_test_np)\nrmse_test_post = np.sqrt(mean_squared_error(y_test_np, y_pred_post))\nrmse_relativo = rmse_test_post / y_test_np.mean()\nr2_test = r2_score(y_test_np, y_pred_post)\n\nprint(f\"RMSE Test (post-pruning): {rmse_test_post:.4f}\")\nprint(f\"RMSE Relativo (post-pruning): {rmse_relativo:.3f}\")\nprint(f\"R\u00b2 (test): {r2_test:.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # ============================
            # 7) Mostrar resultados
            # ============================

            y_pred_post = tree_pruned.predict(X_test_np)
            rmse_test_post = np.sqrt(mean_squared_error(y_test_np, y_pred_post))
            rmse_relativo = rmse_test_post / y_test_np.mean()
            r2_test = r2_score(y_test_np, y_pred_post)

            print(f"RMSE Test (post-pruning): {rmse_test_post:.4f}")
            print(f"RMSE Relativo (post-pruning): {rmse_relativo:.3f}")
            print(f"R² (test): {r2_test:.3f}")
    except Exception as e:
        st.error(f"Error en la celda 88: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 89")
    if show_code:
        st.code("from sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import GridSearchCV, KFold, learning_curve\nimport numpy as np\nimport matplotlib.pyplot as plt\n\npipe_knn = Pipeline(steps=[\n    (\"knn\", KNeighborsRegressor())\n])\nparam_grid = {\n    \"knn__n_neighbors\": list(range(1, 50)),\n    \"knn__weights\": [\"uniform\", \"distance\"],\n    \"knn__metric\": [\"minkowski\"],\n    \"knn__p\": [1, 2],\n    \"knn__algorithm\": ['brute', 'kd_tree', 'ball_tree']\n}\n\ncv = KFold(n_splits=5, shuffle=True, random_state=42)\ngs = GridSearchCV(pipe_knn, param_grid, cv=cv, scoring=\"neg_root_mean_squared_error\", n_jobs=-1)\ngs.fit(X_train_new, y_train)\n\n# Obtener el mejor modelo ya optimizado\nbest_knn_model = gs.best_estimator_\n\n# Imprimir los resultados del mejor modelo\nprint(\"=== KNN - Mejores HP ===\")\nprint(\"Mejores hiperpar\u00e1metros:\", gs.best_params_)\nprint(f\"Mejor RMSE CV: {-gs.best_score_:.4f}\")\n\n# Evaluaci\u00f3n final en el conjunto de prueba\ny_pred_test = best_knn_model.predict(X_test_new)\nrmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))\nr2_test = r2_score(y_test, y_pred_test)\nrmse_relativo = rmse_test / y_test.mean()\nprint(f\"RMSE Test: {rmse_test:.4f}\")\nprint(f\"R\u00b2 (test): {r2_test:.3f}\")\nprint(f\"RMSE Test relativo: {rmse_relativo:.4f}\")\n\n# --- Curva de Aprendizaje del mejor modelo ---\ntrain_sizes, train_scores, val_scores = learning_curve(\n    best_knn_model, X_train_new, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=cv,\n    scoring='neg_root_mean_squared_error',\n    random_state=42\n)\n\ntrain_rmse = -train_scores.mean(axis=1)\nval_rmse = -val_scores.mean(axis=1)\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse, 'o-', label='Error de entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', label='Error de validaci\u00f3n')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de aprendizaje del mejor modelo KNN')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import GridSearchCV, KFold, learning_curve
            import numpy as np
            import matplotlib.pyplot as plt

            pipe_knn = Pipeline(steps=[
                ("knn", KNeighborsRegressor())
            ])
            param_grid = {
                "knn__n_neighbors": list(range(1, 50)),
                "knn__weights": ["uniform", "distance"],
                "knn__metric": ["minkowski"],
                "knn__p": [1, 2],
                "knn__algorithm": ['brute', 'kd_tree', 'ball_tree']
            }

            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            gs = GridSearchCV(pipe_knn, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(X_train_new, y_train)

            # Obtener el mejor modelo ya optimizado
            best_knn_model = gs.best_estimator_

            # Imprimir los resultados del mejor modelo
            print("=== KNN - Mejores HP ===")
            print("Mejores hiperparámetros:", gs.best_params_)
            print(f"Mejor RMSE CV: {-gs.best_score_:.4f}")

            # Evaluación final en el conjunto de prueba
            y_pred_test = best_knn_model.predict(X_test_new)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test = r2_score(y_test, y_pred_test)
            rmse_relativo = rmse_test / y_test.mean()
            print(f"RMSE Test: {rmse_test:.4f}")
            print(f"R² (test): {r2_test:.3f}")
            print(f"RMSE Test relativo: {rmse_relativo:.4f}")

            # --- Curva de Aprendizaje del mejor modelo ---
            train_sizes, train_scores, val_scores = learning_curve(
                best_knn_model, X_train_new, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=cv,
                scoring='neg_root_mean_squared_error',
                random_state=42
            )

            train_rmse = -train_scores.mean(axis=1)
            val_rmse = -val_scores.mean(axis=1)

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse, 'o-', label='Error de entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', label='Error de validación')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de aprendizaje del mejor modelo KNN')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 89: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 90")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n Lineal\n#######################################\n\nfrom sklearn.model_selection import learning_curve\n\nlin_pipeline = Pipeline([\n    (\"regressor\", LinearRegression())\n])\n\n# Curva de aprendizaje\ntrain_sizes, train_scores, val_scores = learning_curve(\n    lin_pipeline, X_train_new, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=5,\n    scoring='neg_mean_squared_error',\n    random_state=42\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse, 'o-', label='Error de entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', label='Error de validaci\u00f3n')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de aprendizaje - Regresi\u00f3n lineal')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión Lineal
            #######################################

            from sklearn.model_selection import learning_curve

            lin_pipeline = Pipeline([
                ("regressor", LinearRegression())
            ])

            # Curva de aprendizaje
            train_sizes, train_scores, val_scores = learning_curve(
                lin_pipeline, X_train_new, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                scoring='neg_mean_squared_error',
                random_state=42
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse, 'o-', label='Error de entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', label='Error de validación')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de aprendizaje - Regresión lineal')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 90: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 91")
    if show_code:
        st.code("lin_pipeline.fit(X_train_new, y_train)\ny_pred = lin_pipeline.predict(X_test_new)\nprint(f' RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}')\nprint(f' R\u00b2: {r2_score(y_test, y_pred):.3f}')\nprint(f\"RMSE relativo (test): {np.sqrt(mean_squared_error(y_test, y_pred)) / y_test.mean():.3f}\")\n# revsiar que el error cuadr\u00e1tico medio sea menor a 10%, si es el caso, es necesario mirar otro modelo.", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            lin_pipeline.fit(X_train_new, y_train)
            y_pred = lin_pipeline.predict(X_test_new)
            print(f' RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}')
            print(f' R²: {r2_score(y_test, y_pred):.3f}')
            print(f"RMSE relativo (test): {np.sqrt(mean_squared_error(y_test, y_pred)) / y_test.mean():.3f}")
            # revsiar que el error cuadrático medio sea menor a 10%, si es el caso, es necesario mirar otro modelo.
    except Exception as e:
        st.error(f"Error en la celda 91: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 92")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n Lasso\n#######################################\n\nfrom sklearn.linear_model import Lasso # Import Lasso\nfrom sklearn.model_selection import validation_curve # Import validation_curve\n\nlasso_pipeline = Pipeline([\n    (\"regressor\", Lasso(max_iter=10000))  # Aumentar iteraciones si hay warning\n])\n\n# Rango de valores para alpha\nalphas = np.logspace(-4, 1, 10)\n\n# Validaci\u00f3n cruzada sobre X_train e y_train\ntrain_scores, val_scores = validation_curve(\n    lasso_pipeline, X_train_new, y_train,\n    param_name=\"regressor__alpha\",\n    param_range=alphas,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\n# Convertir a RMSE\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\n# Graficar curva de validaci\u00f3n\nplt.figure(figsize=(8, 5))\nplt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(alphas, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Alpha (par\u00e1metro de regularizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de Validaci\u00f3n - Lasso')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión Lasso
            #######################################

            from sklearn.linear_model import Lasso # Import Lasso
            from sklearn.model_selection import validation_curve # Import validation_curve

            lasso_pipeline = Pipeline([
                ("regressor", Lasso(max_iter=10000))  # Aumentar iteraciones si hay warning
            ])

            # Rango de valores para alpha
            alphas = np.logspace(-4, 1, 10)

            # Validación cruzada sobre X_train e y_train
            train_scores, val_scores = validation_curve(
                lasso_pipeline, X_train_new, y_train,
                param_name="regressor__alpha",
                param_range=alphas,
                scoring="neg_mean_squared_error",
                cv=5
            )

            # Convertir a RMSE
            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            # Graficar curva de validación
            plt.figure(figsize=(8, 5))
            plt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(alphas, val_rmse, 'o-', label='Validación')
            plt.xlabel('Alpha (parámetro de regularización)')
            plt.ylabel('RMSE')
            plt.title('Curva de Validación - Lasso')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 92: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 93")
    if show_code:
        st.code("# Convertir a RMSE\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\n# Encontrar el mejor alpha\nbest_alpha = alphas[np.argmin(val_rmse)]\nprint(f\"Mejor alpha seg\u00fan validaci\u00f3n cruzada: {best_alpha}\")\n\n# Usamos el mejor alpha si ya lo tienes, o define uno razonable\nlasso_pipeline.set_params(regressor__alpha=best_alpha)\n\ntrain_sizes, train_scores, val_scores = learning_curve(\n    lasso_pipeline, X_train_new, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 5),\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse, 'o-', label='Entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de Aprendizaje - Lasso')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Convertir a RMSE
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            # Encontrar el mejor alpha
            best_alpha = alphas[np.argmin(val_rmse)]
            print(f"Mejor alpha según validación cruzada: {best_alpha}")

            # Usamos el mejor alpha si ya lo tienes, o define uno razonable
            lasso_pipeline.set_params(regressor__alpha=best_alpha)

            train_sizes, train_scores, val_scores = learning_curve(
                lasso_pipeline, X_train_new, y_train,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse, 'o-', label='Entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', label='Validación')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de Aprendizaje - Lasso')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 93: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 94")
    if show_code:
        st.code("# Entrenar modelo final con el mejor alpha\nlasso_pipeline.fit(X_train_new, y_train)\n\n# Evaluar sobre test\ny_pred = lasso_pipeline.predict(X_test_new)\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Entrenar modelo final con el mejor alpha
            lasso_pipeline.fit(X_train_new, y_train)

            # Evaluar sobre test
            y_pred = lasso_pipeline.predict(X_test_new)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 94: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 95")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n Ridge\n#######################################\nfrom sklearn.linear_model import Ridge # Import Ridge\nfrom sklearn.model_selection import validation_curve\n\nridge_pipeline = Pipeline([\n    (\"regressor\", Ridge())\n])\n\nalphas = np.logspace(-4, 4, 10)\ntrain_scores, val_scores = validation_curve(\n    ridge_pipeline, X_train_new, y_train,\n    param_name=\"regressor__alpha\",\n    param_range=alphas,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(alphas, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Alpha (par\u00e1metro de regularizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de validaci\u00f3n - Ridge')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión Ridge
            #######################################
            from sklearn.linear_model import Ridge # Import Ridge
            from sklearn.model_selection import validation_curve

            ridge_pipeline = Pipeline([
                ("regressor", Ridge())
            ])

            alphas = np.logspace(-4, 4, 10)
            train_scores, val_scores = validation_curve(
                ridge_pipeline, X_train_new, y_train,
                param_name="regressor__alpha",
                param_range=alphas,
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(alphas, val_rmse, 'o-', label='Validación')
            plt.xlabel('Alpha (parámetro de regularización)')
            plt.ylabel('RMSE')
            plt.title('Curva de validación - Ridge')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 95: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 96")
    if show_code:
        st.code("best_alpha = alphas[np.argmin(val_rmse)]\nprint(f\" Mejor alpha seg\u00fan validaci\u00f3n cruzada: {best_alpha}\")\n\n# Curva de aprendizaje con alpha fijo\nridge_pipeline.set_params(regressor__alpha=best_alpha)\n\ntrain_sizes, train_scores_l, val_scores_l = learning_curve(\n    ridge_pipeline, X_train_new, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 5),\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse_l = np.sqrt(-train_scores_l.mean(axis=1))\nval_rmse_l = np.sqrt(-val_scores_l.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse_l, 'o-', label='Entrenamiento')\nplt.plot(train_sizes, val_rmse_l, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de aprendizaje - Ridge')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            best_alpha = alphas[np.argmin(val_rmse)]
            print(f" Mejor alpha según validación cruzada: {best_alpha}")

            # Curva de aprendizaje con alpha fijo
            ridge_pipeline.set_params(regressor__alpha=best_alpha)

            train_sizes, train_scores_l, val_scores_l = learning_curve(
                ridge_pipeline, X_train_new, y_train,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse_l = np.sqrt(-train_scores_l.mean(axis=1))
            val_rmse_l = np.sqrt(-val_scores_l.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse_l, 'o-', label='Entrenamiento')
            plt.plot(train_sizes, val_rmse_l, 'o-', label='Validación')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de aprendizaje - Ridge')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 96: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 97")
    if show_code:
        st.code("ridge_pipeline.fit(X_train_new, y_train)\n\ny_pred = ridge_pipeline.predict(X_test_new)\n\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ridge_pipeline.fit(X_train_new, y_train)

            y_pred = ridge_pipeline.predict(X_test_new)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 97: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 98")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n Ridge + Polin\u00f3mica\n#######################################\nfrom sklearn.preprocessing import PolynomialFeatures\n\npoly_ridge_pipeline = Pipeline([\n    (\"poly\", PolynomialFeatures(degree=2, include_bias=False)),\n    (\"regressor\", Ridge())\n])\n\nalphas = np.logspace(-4, 2, 10)\n\ntrain_scores, val_scores = validation_curve(\n    poly_ridge_pipeline, X_train_new, y_train,\n    param_name=\"regressor__alpha\",\n    param_range=alphas,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(alphas, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Alpha (par\u00e1metro de regularizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de validaci\u00f3n - Ridge + Polinomial')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()\n\n# \ud83d\udd0d Mejor alpha\nbest_alpha = alphas[np.argmin(val_rmse)]\nprint(f\"\ud83d\udd0d Mejor alpha seg\u00fan validaci\u00f3n cruzada: {best_alpha}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión Ridge + Polinómica
            #######################################
            from sklearn.preprocessing import PolynomialFeatures

            poly_ridge_pipeline = Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("regressor", Ridge())
            ])

            alphas = np.logspace(-4, 2, 10)

            train_scores, val_scores = validation_curve(
                poly_ridge_pipeline, X_train_new, y_train,
                param_name="regressor__alpha",
                param_range=alphas,
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(alphas, val_rmse, 'o-', label='Validación')
            plt.xlabel('Alpha (parámetro de regularización)')
            plt.ylabel('RMSE')
            plt.title('Curva de validación - Ridge + Polinomial')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # 🔍 Mejor alpha
            best_alpha = alphas[np.argmin(val_rmse)]
            print(f"🔍 Mejor alpha según validación cruzada: {best_alpha}")
    except Exception as e:
        st.error(f"Error en la celda 98: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 99")
    if show_code:
        st.code("poly_ridge_pipeline.set_params(regressor__alpha=best_alpha)\n\ntrain_sizes, train_scores_lc, val_scores_lc = learning_curve(\n    poly_ridge_pipeline, X_train_new, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=5,\n    scoring=\"neg_mean_squared_error\"\n)\n\ntrain_rmse_lc = np.sqrt(-train_scores_lc.mean(axis=1))\nval_rmse_lc = np.sqrt(-val_scores_lc.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse_lc, 'o-', label=\"Entrenamiento\")\nplt.plot(train_sizes, val_rmse_lc, 'o-', label=\"Validaci\u00f3n\")\nplt.xlabel(\"Tama\u00f1o del conjunto de entrenamiento\")\nplt.ylabel(\"RMSE\")\nplt.title(\"Curva de aprendizaje - Ridge + Polinomial\")\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            poly_ridge_pipeline.set_params(regressor__alpha=best_alpha)

            train_sizes, train_scores_lc, val_scores_lc = learning_curve(
                poly_ridge_pipeline, X_train_new, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                scoring="neg_mean_squared_error"
            )

            train_rmse_lc = np.sqrt(-train_scores_lc.mean(axis=1))
            val_rmse_lc = np.sqrt(-val_scores_lc.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse_lc, 'o-', label="Entrenamiento")
            plt.plot(train_sizes, val_rmse_lc, 'o-', label="Validación")
            plt.xlabel("Tamaño del conjunto de entrenamiento")
            plt.ylabel("RMSE")
            plt.title("Curva de aprendizaje - Ridge + Polinomial")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 99: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 100")
    if show_code:
        st.code("poly_ridge_pipeline.fit(X_train_new, y_train)\ny_pred = poly_ridge_pipeline.predict(X_test_new)\n\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            poly_ridge_pipeline.fit(X_train_new, y_train)
            y_pred = poly_ridge_pipeline.predict(X_test_new)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 100: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 101")
    if show_code:
        st.code("########################################\n#ElasticNet\n#######################################\nfrom sklearn.linear_model import ElasticNet # Import ElasticNet\n\nelasticnet_pipeline = Pipeline([\n    (\"poly\", PolynomialFeatures(degree=2, include_bias=False)),\n    (\"regressor\", ElasticNet(max_iter=10000))  # aumentar iteraciones por estabilidad\n])\n\n\nalphas = np.logspace(-4, 1, 10)\n\ntrain_scores, val_scores = validation_curve(\n    elasticnet_pipeline, X_train_new, y_train,\n    param_name=\"regressor__alpha\",\n    param_range=alphas,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(alphas, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('Alpha (par\u00e1metro de regularizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de validaci\u00f3n - ElasticNet + Polinomial')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()\n\n# Mejor alpha\nbest_alpha = alphas[np.argmin(val_rmse)]\nprint(f\"\ud83d\udd0d Mejor alpha seg\u00fan validaci\u00f3n cruzada: {best_alpha}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #ElasticNet
            #######################################
            from sklearn.linear_model import ElasticNet # Import ElasticNet

            elasticnet_pipeline = Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("regressor", ElasticNet(max_iter=10000))  # aumentar iteraciones por estabilidad
            ])


            alphas = np.logspace(-4, 1, 10)

            train_scores, val_scores = validation_curve(
                elasticnet_pipeline, X_train_new, y_train,
                param_name="regressor__alpha",
                param_range=alphas,
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.semilogx(alphas, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(alphas, val_rmse, 'o-', label='Validación')
            plt.xlabel('Alpha (parámetro de regularización)')
            plt.ylabel('RMSE')
            plt.title('Curva de validación - ElasticNet + Polinomial')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Mejor alpha
            best_alpha = alphas[np.argmin(val_rmse)]
            print(f"🔍 Mejor alpha según validación cruzada: {best_alpha}")
    except Exception as e:
        st.error(f"Error en la celda 101: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 102")
    if show_code:
        st.code("elasticnet_pipeline.set_params(regressor__alpha=best_alpha)\n\ntrain_sizes, train_scores_lc, val_scores_lc = learning_curve(\n    elasticnet_pipeline, X_train_new, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=5,\n    scoring=\"neg_mean_squared_error\"\n)\n\ntrain_rmse_lc = np.sqrt(-train_scores_lc.mean(axis=1))\nval_rmse_lc = np.sqrt(-val_scores_lc.mean(axis=1))\n\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse_lc, 'o-', label=\"Entrenamiento\")\nplt.plot(train_sizes, val_rmse_lc, 'o-', label=\"Validaci\u00f3n\")\nplt.xlabel(\"Tama\u00f1o del conjunto de entrenamiento\")\nplt.ylabel(\"RMSE\")\nplt.title(\"Curva de aprendizaje - ElasticNet + Polinomial\")\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            elasticnet_pipeline.set_params(regressor__alpha=best_alpha)

            train_sizes, train_scores_lc, val_scores_lc = learning_curve(
                elasticnet_pipeline, X_train_new, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                scoring="neg_mean_squared_error"
            )

            train_rmse_lc = np.sqrt(-train_scores_lc.mean(axis=1))
            val_rmse_lc = np.sqrt(-val_scores_lc.mean(axis=1))

            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse_lc, 'o-', label="Entrenamiento")
            plt.plot(train_sizes, val_rmse_lc, 'o-', label="Validación")
            plt.xlabel("Tamaño del conjunto de entrenamiento")
            plt.ylabel("RMSE")
            plt.title("Curva de aprendizaje - ElasticNet + Polinomial")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 102: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 103")
    if show_code:
        st.code("elasticnet_pipeline.fit(X_train_new, y_train)\ny_pred = elasticnet_pipeline.predict(X_test_new)\n\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            elasticnet_pipeline.fit(X_train_new, y_train)
            y_pred = elasticnet_pipeline.predict(X_test_new)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 103: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 104")
    if show_code:
        st.code("########################################\n#Regresi\u00f3n SVR\n#######################################\nfrom sklearn.svm import SVR # Import SVR\n\nsvr_pipeline = Pipeline([\n    (\"regressor\", SVR(kernel='rbf'))\n])\n\n# Curva de validaci\u00f3n para encontrar el mejor C\nC_values = np.logspace(-2, 2, 8)  # valores de C de 0.01 a 100\ntrain_scores, val_scores = validation_curve(\n    svr_pipeline, X_train_new, y_train,\n    param_name=\"regressor__C\",\n    param_range=C_values,\n    scoring=\"neg_mean_squared_error\",\n    cv=5\n)\n\ntrain_rmse = np.sqrt(-train_scores.mean(axis=1))\nval_rmse = np.sqrt(-val_scores.mean(axis=1))\n\n# Graficar curva de validaci\u00f3n\nplt.figure(figsize=(8, 5))\nplt.semilogx(C_values, train_rmse, 'o-', label='Entrenamiento')\nplt.semilogx(C_values, val_rmse, 'o-', label='Validaci\u00f3n')\nplt.xlabel('C (par\u00e1metro de penalizaci\u00f3n)')\nplt.ylabel('RMSE')\nplt.title('Curva de Validaci\u00f3n - SVR')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()\n\n# Encontrar mejor C\nbest_C = C_values[np.argmin(val_rmse)]\nprint(f\"Mejor C seg\u00fan validaci\u00f3n cruzada: {best_C}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            ########################################
            #Regresión SVR
            #######################################
            from sklearn.svm import SVR # Import SVR

            svr_pipeline = Pipeline([
                ("regressor", SVR(kernel='rbf'))
            ])

            # Curva de validación para encontrar el mejor C
            C_values = np.logspace(-2, 2, 8)  # valores de C de 0.01 a 100
            train_scores, val_scores = validation_curve(
                svr_pipeline, X_train_new, y_train,
                param_name="regressor__C",
                param_range=C_values,
                scoring="neg_mean_squared_error",
                cv=5
            )

            train_rmse = np.sqrt(-train_scores.mean(axis=1))
            val_rmse = np.sqrt(-val_scores.mean(axis=1))

            # Graficar curva de validación
            plt.figure(figsize=(8, 5))
            plt.semilogx(C_values, train_rmse, 'o-', label='Entrenamiento')
            plt.semilogx(C_values, val_rmse, 'o-', label='Validación')
            plt.xlabel('C (parámetro de penalización)')
            plt.ylabel('RMSE')
            plt.title('Curva de Validación - SVR')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Encontrar mejor C
            best_C = C_values[np.argmin(val_rmse)]
            print(f"Mejor C según validación cruzada: {best_C}")
    except Exception as e:
        st.error(f"Error en la celda 104: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 105")
    if show_code:
        st.code("# \u2699\ufe0f Usar el mejor C encontrado previamente\nsvr_pipeline.set_params(regressor__C=best_C)\n\n# Curva de aprendizaje\ntrain_sizes, train_scores, val_scores = learning_curve(\n    svr_pipeline, X_train_new, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 8),\n    cv=5,\n    scoring='neg_mean_squared_error',\n    shuffle=True,\n    random_state=42\n)\n\ntrain_rmse = np.sqrt(-train_scores)\nval_rmse = np.sqrt(-val_scores)\n\n# Promedios\ntrain_rmse_mean = train_rmse.mean(axis=1)\nval_rmse_mean = val_rmse.mean(axis=1)\n\n# Graficar curva de aprendizaje\nplt.figure(figsize=(8, 5))\nplt.plot(train_sizes, train_rmse_mean, 'o-', label='Entrenamiento')\nplt.plot(train_sizes, val_rmse_mean, 'o-', label='Validaci\u00f3n')\nplt.xlabel('N\u00famero de muestras de entrenamiento')\nplt.ylabel('RMSE')\nplt.title('Curva de Aprendizaje - SVR')\nplt.legend()\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # ⚙️ Usar el mejor C encontrado previamente
            svr_pipeline.set_params(regressor__C=best_C)

            # Curva de aprendizaje
            train_sizes, train_scores, val_scores = learning_curve(
                svr_pipeline, X_train_new, y_train,
                train_sizes=np.linspace(0.1, 1.0, 8),
                cv=5,
                scoring='neg_mean_squared_error',
                shuffle=True,
                random_state=42
            )

            train_rmse = np.sqrt(-train_scores)
            val_rmse = np.sqrt(-val_scores)

            # Promedios
            train_rmse_mean = train_rmse.mean(axis=1)
            val_rmse_mean = val_rmse.mean(axis=1)

            # Graficar curva de aprendizaje
            plt.figure(figsize=(8, 5))
            plt.plot(train_sizes, train_rmse_mean, 'o-', label='Entrenamiento')
            plt.plot(train_sizes, val_rmse_mean, 'o-', label='Validación')
            plt.xlabel('Número de muestras de entrenamiento')
            plt.ylabel('RMSE')
            plt.title('Curva de Aprendizaje - SVR')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 105: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 106")
    if show_code:
        st.code("# Reentrenar modelo con mejor C y evaluar sobre test\nsvr_pipeline.fit(X_train_new, y_train)\ny_pred = svr_pipeline.predict(X_test_new)\n\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE (test): {rmse:.3f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # Reentrenar modelo con mejor C y evaluar sobre test
            svr_pipeline.fit(X_train_new, y_train)
            y_pred = svr_pipeline.predict(X_test_new)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE (test): {rmse:.3f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 106: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 107")
    if show_code:
        st.code("import numpy as np\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.model_selection import GridSearchCV, KFold\nfrom sklearn.metrics import mean_squared_error\n\nmodel = RandomForestRegressor(random_state=42)\n\nparam_grid = {\n    'n_estimators': [50, 100, 200],         # N\u00famero de \u00e1rboles en el bosque\n    'max_features': ['sqrt', 'log2', None], # N\u00famero de caracter\u00edsticas a considerar por cada \u00e1rbol\n    'max_depth': [None, 10, 20]             # Profundidad m\u00e1xima de cada \u00e1rbol\n}\n\n# 2. Configuraci\u00f3n de la validaci\u00f3n cruzada\ncv = KFold(n_splits=5, shuffle=True, random_state=42)\n\n# Se busca la combinaci\u00f3n que minimice el RMSE\ngs = GridSearchCV(\n    estimator=model,\n    param_grid=param_grid,\n    cv=cv,\n    scoring='neg_root_mean_squared_error',\n    n_jobs=-1  # Usa todos los n\u00facleos disponibles\n)\n\n# Entrenar el modelo con los datos de entrenamiento\ngs.fit(X_train_new, y_train)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import GridSearchCV, KFold
            from sklearn.metrics import mean_squared_error

            model = RandomForestRegressor(random_state=42)

            param_grid = {
                'n_estimators': [50, 100, 200],         # Número de árboles en el bosque
                'max_features': ['sqrt', 'log2', None], # Número de características a considerar por cada árbol
                'max_depth': [None, 10, 20]             # Profundidad máxima de cada árbol
            }

            # 2. Configuración de la validación cruzada
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            # Se busca la combinación que minimice el RMSE
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1  # Usa todos los núcleos disponibles
            )

            # Entrenar el modelo con los datos de entrenamiento
            gs.fit(X_train_new, y_train)
    except Exception as e:
        st.error(f"Error en la celda 107: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 108")
    if show_code:
        st.code("# --- Gr\u00e1fico de la Curva de Aprendizaje ---\nplt.figure(figsize=(10, 6))\nplt.plot(train_sizes, train_rmse, 'o-', label='Error de Entrenamiento')\nplt.plot(train_sizes, val_rmse, 'o-', label='Error de Validaci\u00f3n')\n\nplt.title('Curva de Aprendizaje')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.legend(loc='best')\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # --- Gráfico de la Curva de Aprendizaje ---
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_rmse, 'o-', label='Error de Entrenamiento')
            plt.plot(train_sizes, val_rmse, 'o-', label='Error de Validación')

            plt.title('Curva de Aprendizaje')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 108: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 109")
    if show_code:
        st.code("# 4. Obtener el mejor modelo y sus resultados\nbest_rf_model = gs.best_estimator_\n\nprint(\"=== Random Forest - Mejores HP ===\")\nprint(\"Mejores hiperpar\u00e1metros:\", gs.best_params_)\nprint(f\"Mejor RMSE CV: {-gs.best_score_:.4f}\")\n\n# 5. Evaluaci\u00f3n final en el conjunto de prueba\ny_pred_test = best_rf_model.predict(X_test_new)\nrmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))\nr2 = r2_score(y_test, y_pred)\n\nprint(f\"RMSE Test: {rmse_test:.4f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse_test / y_test.mean():.3f}\")", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            # 4. Obtener el mejor modelo y sus resultados
            best_rf_model = gs.best_estimator_

            print("=== Random Forest - Mejores HP ===")
            print("Mejores hiperparámetros:", gs.best_params_)
            print(f"Mejor RMSE CV: {-gs.best_score_:.4f}")

            # 5. Evaluación final en el conjunto de prueba
            y_pred_test = best_rf_model.predict(X_test_new)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE Test: {rmse_test:.4f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse_test / y_test.mean():.3f}")
    except Exception as e:
        st.error(f"Error en la celda 109: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 110")
    if show_code:
        st.code("from sklearn.ensemble import GradientBoostingRegressor\nfrom sklearn.model_selection import RandomizedSearchCV, KFold\nfrom sklearn.metrics import mean_squared_error\nfrom sklearn.metrics import make_scorer\n\n# 1. Configuraci\u00f3n del modelo y de los hiperpar\u00e1metros a buscar\nmodel_gb = GradientBoostingRegressor(random_state=42)\n\n# Par\u00e1metros para la b\u00fasqueda en cuadr\u00edcula aleatoria (Randomized Search)\nparam_dist_gb = {\n    'n_estimators': [100, 200, 300, 400],         # N\u00famero de \u00e1rboles\n    'learning_rate': [0.01, 0.05, 0.1, 0.2],    # Tasa de aprendizaje\n    'max_depth': [3, 4, 5, 6],                     # Profundidad m\u00e1xima de cada \u00e1rbol\n    'subsample': [0.6, 0.8, 1.0],                  # Fracci\u00f3n de muestras para cada \u00e1rbol\n    'max_features': ['sqrt', 'log2', None]         # Caracter\u00edsticas a considerar\n}\n\n# 2. Configuraci\u00f3n de la validaci\u00f3n cruzada\ncv = KFold(n_splits=5, shuffle=True, random_state=42)\n\nscoring = 'neg_root_mean_squared_error'\n\n# 3. B\u00fasqueda de los mejores hiperpar\u00e1metros con RandomizedSearchCV\ngs_gb = RandomizedSearchCV(\n    estimator=model_gb,\n    param_distributions=param_dist_gb,\n    n_iter=50,\n    cv=cv,\n    scoring='neg_root_mean_squared_error',\n    n_jobs=-1,\n    random_state=42,\n    refit=True,\n    error_score='raise'   # fuerza a mostrar el error\n)\n\n# Entrenar el modelo\ngs_gb.fit(X_train_new, y_train)", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import RandomizedSearchCV, KFold
            from sklearn.metrics import mean_squared_error
            from sklearn.metrics import make_scorer

            # 1. Configuración del modelo y de los hiperparámetros a buscar
            model_gb = GradientBoostingRegressor(random_state=42)

            # Parámetros para la búsqueda en cuadrícula aleatoria (Randomized Search)
            param_dist_gb = {
                'n_estimators': [100, 200, 300, 400],         # Número de árboles
                'learning_rate': [0.01, 0.05, 0.1, 0.2],    # Tasa de aprendizaje
                'max_depth': [3, 4, 5, 6],                     # Profundidad máxima de cada árbol
                'subsample': [0.6, 0.8, 1.0],                  # Fracción de muestras para cada árbol
                'max_features': ['sqrt', 'log2', None]         # Características a considerar
            }

            # 2. Configuración de la validación cruzada
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            scoring = 'neg_root_mean_squared_error'

            # 3. Búsqueda de los mejores hiperparámetros con RandomizedSearchCV
            gs_gb = RandomizedSearchCV(
                estimator=model_gb,
                param_distributions=param_dist_gb,
                n_iter=50,
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                random_state=42,
                refit=True,
                error_score='raise'   # fuerza a mostrar el error
            )

            # Entrenar el modelo
            gs_gb.fit(X_train_new, y_train)
    except Exception as e:
        st.error(f"Error en la celda 110: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    

    st.markdown("#### Celda 111")
    if show_code:
        st.code("from sklearn.metrics import r2_score, mean_squared_error\nfrom sklearn.model_selection import learning_curve\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# 4) Mejor modelo y resultados\nbest_gb_model = gs_gb.best_estimator_\nprint(\"=== Gradient Boosting - Mejores HP ===\")\nprint(\"Mejores hiperpar\u00e1metros:\", gs_gb.best_params_)\nprint(f\"Mejor RMSE CV: {-gs_gb.best_score_:.4f}\")  # neg -> positivo\n\ny_pred_gb = best_gb_model.predict(X_test_new)\n\n# 5) Evaluaci\u00f3n en TEST\nrmse_test_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))\nr2_test = r2_score(y_test, y_pred_gb)\nrmse_rel = rmse_test_gb / np.mean(y_test)\n\nprint(f\"RMSE Test: {rmse_test_gb:.4f}\")\nprint(f\"R\u00b2 (test): {r2:.3f}\")\nprint(f\"RMSE relativo (test): {rmse_test_gb / y_test.mean():.3f}\")\n\n# --- Curva de Aprendizaje ---\ntrain_sizes, train_scores, val_scores = learning_curve(\n    best_gb_model,\n    X_train_new, y_train,\n    train_sizes=np.linspace(0.1, 1.0, 10),\n    cv=cv,\n    scoring='neg_root_mean_squared_error',  # ya devuelve -RMSE\n    n_jobs=-1\n)\n\n\ntrain_rmse = -train_scores.mean(axis=1)\nval_rmse   = -val_scores.mean(axis=1)\n\nplt.figure(figsize=(10, 6))\nplt.plot(train_sizes, train_rmse, 'o-', label='Error de Entrenamiento')\nplt.plot(train_sizes, val_rmse,   'o-', label='Error de Validaci\u00f3n')\n\n# L\u00ednea de referencia con el RMSE de test (opcional)\nplt.axhline(rmse_test_gb, linestyle='--', linewidth=1,\n            label=f'RMSE Test = {rmse_test_gb:.2f}')\n\nplt.title('Curva de Aprendizaje - Gradient Boosting')\nplt.xlabel('Tama\u00f1o del conjunto de entrenamiento')\nplt.ylabel('RMSE')\nplt.legend(loc='best')\nplt.grid(True)\nplt.tight_layout()\nplt.show()", language="python")

    # Ejecutar y capturar stdout
    _stdout = io.StringIO()
    try:
        with redirect_stdout(_stdout):
            from sklearn.metrics import r2_score, mean_squared_error
            from sklearn.model_selection import learning_curve
            import numpy as np
            import matplotlib.pyplot as plt

            # 4) Mejor modelo y resultados
            best_gb_model = gs_gb.best_estimator_
            print("=== Gradient Boosting - Mejores HP ===")
            print("Mejores hiperparámetros:", gs_gb.best_params_)
            print(f"Mejor RMSE CV: {-gs_gb.best_score_:.4f}")  # neg -> positivo

            y_pred_gb = best_gb_model.predict(X_test_new)

            # 5) Evaluación en TEST
            rmse_test_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
            r2_test = r2_score(y_test, y_pred_gb)
            rmse_rel = rmse_test_gb / np.mean(y_test)

            print(f"RMSE Test: {rmse_test_gb:.4f}")
            print(f"R² (test): {r2:.3f}")
            print(f"RMSE relativo (test): {rmse_test_gb / y_test.mean():.3f}")

            # --- Curva de Aprendizaje ---
            train_sizes, train_scores, val_scores = learning_curve(
                best_gb_model,
                X_train_new, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=cv,
                scoring='neg_root_mean_squared_error',  # ya devuelve -RMSE
                n_jobs=-1
            )


            train_rmse = -train_scores.mean(axis=1)
            val_rmse   = -val_scores.mean(axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_rmse, 'o-', label='Error de Entrenamiento')
            plt.plot(train_sizes, val_rmse,   'o-', label='Error de Validación')

            # Línea de referencia con el RMSE de test (opcional)
            plt.axhline(rmse_test_gb, linestyle='--', linewidth=1,
                        label=f'RMSE Test = {rmse_test_gb:.2f}')

            plt.title('Curva de Aprendizaje - Gradient Boosting')
            plt.xlabel('Tamaño del conjunto de entrenamiento')
            plt.ylabel('RMSE')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        st.error(f"Error en la celda 111: {e}")

    # Mostrar stdout
    out_text = _stdout.getvalue()
    if out_text.strip():
        st.text(out_text)

    # Mostrar figuras nuevas generadas en esta celda
    new_figs = set(plt.get_fignums()) - prev_figs
    for fnum in sorted(new_figs):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    prev_figs = set(plt.get_fignums())
    
    st.success('Ejecución del notebook finalizada.')
