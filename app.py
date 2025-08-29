
# streamlit run hdhi_public_app.py
import os
import io
import gc
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import traceback

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

# Plotly optional
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except Exception:
    px = None; go = None; pio = None
    HAS_PLOTLY = False

import nbformat

st.set_page_config(page_title="HDHI — Public App (GitHub RAW)", layout="wide")

# ===================== CONFIG =====================
# 👇 Pega aquí TU URL RAW pública del CSV (sin tokens), por ejemplo:
# "https://raw.githubusercontent.com/USER/REPO/BRANCH/data/HDHI%20Admission%20data.csv"
RAW_CSV_URL = "https://raw.githubusercontent.com/Juansebastianrde/Reduccion-de-dimensionalidad/main/HDHI%20Admission%20data.csv"
# Ruta del notebook incluida en el repo (no se muestra el código)
NOTEBOOK_PATH = "Proyecto_ML (1).ipynb"


# ==================================================

# --------------------- Helpers ---------------------
@st.cache_data(ttl=900)
def load_csv_public_raw(url: str) -> pd.DataFrame:
    if not url or "raw.githubusercontent.com" not in url:
        raise ValueError("Configura RAW_CSV_URL con el enlace RAW público de GitHub.")
    # Pandas puede leer directamente desde HTTPS
    return pd.read_csv(url)

def pretty_gender(v):
    s = str(v).strip().lower()
    if s in ["m","male","1"]: return "Hombre"
    if s in ["f","female","0"]: return "Mujer"
    return str(v)

def pretty_rural(v):
    s = str(v).strip().lower()
    if s in ["1","urban","urbano","u"]: return "Urbano"
    if s in ["0","rural","r"]: return "Rural"
    return str(v)

def display_matplotlib_new_figs(prev_fignums):
    new_fignums = set(plt.get_fignums()) - prev_fignums
    for fnum in sorted(new_fignums):
        fig_obj = plt.figure(fnum)
        st.pyplot(fig_obj)
    return set(plt.get_fignums())

def discover_and_display_fig_objects(ns):
    try:
        from matplotlib.figure import Figure as MplFigure
    except Exception:
        MplFigure = None
    for name, obj in list(ns.items()):
        if MplFigure is not None and isinstance(obj, MplFigure):
            st.pyplot(obj)
        if hasattr(obj, "fig") and hasattr(obj.fig, "savefig"):
            try: st.pyplot(obj.fig)
            except Exception: pass
        if HAS_PLOTLY and (hasattr(obj, "to_dict") and obj.__class__.__name__.endswith("Figure")):
            try: st.plotly_chart(obj, use_container_width=True)
            except Exception: pass

def patch_show_functions(ns):
    def _plt_show(*args, **kwargs):
        fig = plt.gcf()
        try: st.pyplot(fig)
        except Exception: pass
    ns['plt'].show = _plt_show
    if HAS_PLOTLY and 'pio' in ns:
        def _plotly_show(fig=None, *args, **kwargs):
            if fig is None: return
            try: st.plotly_chart(fig, use_container_width=True)
            except Exception: pass
        ns['pio'].show = _plotly_show

def clean_code(src: str) -> str:
    lines = []
    for line in (src or "").splitlines():
        s = line.strip()
        if s.startswith("%") or s.startswith("%%") or s.startswith("!"):
            continue
        lines.append(line)
    return "\n".join(lines)

# --------------------- UI / Styles ---------------------
st.markdown("""
<style>
.card { background:#f6f8fb; border-radius:14px; padding:16px 18px; }
.card h3 { font-size:1.1rem; margin:0 0 10px 2px; color:#0f172a; font-weight:700; }
div[data-baseweb="select"] > div { border-radius:12px !important; }
.filter-item { margin-bottom:14px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Navegación")
page = st.sidebar.radio("Secciones", ["Datos", "Filtros", "Análisis"], index=0)

st.title("HDHI — Public Streamlit (GitHub RAW)")

# Estado
if "df" not in st.session_state:
    st.session_state.df = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None

# ======= Datos (siempre desde RAW público, sin upload/token) =======
if page == "Datos":
    st.subheader("Fuente de datos")
    st.caption("Esta app siempre lee de **GitHub RAW** (público), no se necesitan tokens ni subir archivos.")

    st.write("RAW_CSV_URL actual:")
    st.code(RAW_CSV_URL, language="text")

    try:
        df = load_csv_public_raw(RAW_CSV_URL)
        st.session_state.df = df.copy()
        st.success(f"Datos cargados desde GitHub. Shape: {df.shape}")
        st.dataframe(df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"No se pudo cargar el CSV desde RAW_CSV_URL.\n{e}")
        st.info("Asegúrate de que tu repo sea PÚBLICO y la URL sea el enlace RAW correcto.")
        st.stop()

# ======= Filtros (GENDER / RURAL) =======
elif page == "Filtros":
    if st.session_state.df is None:
        st.info("Ve a la pestaña **Datos** para cargar desde GitHub primero.")
        st.stop()
    df = st.session_state.df.copy()

    card_col, _ = st.columns([1.2, 2])
    with card_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Filters</h3>", unsafe_allow_html=True)

        # Gender
        gender_value = None
        if "GENDER" in df.columns:
            gvals = df["GENDER"].dropna().unique().tolist()
            gopts = ["Choose an option"] + [pretty_gender(v) for v in gvals]
            gsel = st.selectbox("Gender", gopts, index=0, key="flt_gender")
            if gsel != "Choose an option":
                rev = {pretty_gender(v): v for v in gvals}
                gender_value = rev.get(gsel, gsel)
        else:
            st.selectbox("Gender", ["Choose an option"], index=0, key="flt_gender_disabled")

        # RURAL
        rural_value = None
        if "RURAL" in df.columns:
            rvals = df["RURAL"].dropna().unique().tolist()
            ropts = ["Choose an option"] + [pretty_rural(v) for v in rvals]
            rsel = st.selectbox("Urban/Rural", ropts, index=0, key="flt_rural")
            if rsel != "Choose an option":
                rev = {pretty_rural(v): v for v in rvals}
                rural_value = rev.get(rsel, rsel)
        else:
            st.selectbox("Urban/Rural", ["Choose an option"], index=0, key="flt_rural_disabled")

        st.markdown('</div>', unsafe_allow_html=True)

    df_view = df.copy()
    if gender_value is not None:
        df_view = df_view[df_view["GENDER"] == gender_value]
    if rural_value is not None:
        df_view = df_view[df_view["RURAL"] == rural_value]

    st.session_state.filtered_df = df_view
    st.metric("Filas después de filtrar", len(df_view))
    st.dataframe(df_view.head(50), use_container_width=True)

# ======= Análisis: ejecutar notebook sin mostrar código =======
elif page == "Análisis":
    if st.session_state.df is None:
        st.info("Ve a la pestaña **Datos** para cargar desde GitHub primero.")
        st.stop()

    df_base = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.df

    st.subheader("Ejecutar notebook (sin código)")
    st.caption("El notebook debe estar en el **mismo repo** que esta app (archivo local en el deploy).")
    st.write("Notebook actual:", NOTEBOOK_PATH)

    if not os.path.exists(NOTEBOOK_PATH):
        st.error(f"No se encontró el notebook en: {NOTEBOOK_PATH}")
        st.info("Añade el .ipynb al repo público o ajusta NOTEBOOK_PATH.")
        st.stop()

    # Espacio de nombres con librerías + DF inyectado
    ns = {
        "np": np, "pd": pd, "plt": plt, "st": st,
        "stats": stats,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline, "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer, "StandardScaler": StandardScaler,
        "RFE": RFE, "SelectKBest": SelectKBest, "f_regression": f_regression,
        "LinearRegression": LinearRegression, "PCA": PCA,
        "RandomForestRegressor": RandomForestRegressor,
        "px": px, "go": None, "pio": None,
        "df": df_base.copy(), "bd": df_base.copy(), "data": df_base.copy(), "dataset": df_base.copy(),
        "io": io,
    }
    patch_show_functions(ns)

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    prev_fignums = set(plt.get_fignums())
    total, errors = 0, 0
    rendered_any = False

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        total += 1
        code = clean_code(cell.source or "")
        if not code.strip():
            continue

        stdout_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf):
                exec(code, ns, ns)
        except Exception as e:
            errors += 1
            with st.expander("⚠️ Error en una celda (click para ver detalle)", expanded=False):
                st.error(str(e))
                st.code(traceback.format_exc())
        finally:
            out = stdout_buf.getvalue().strip()
            if out:
                with st.expander("📝 Conclusiones / prints", expanded=False):
                    st.text(out)
                rendered_any = True

            prev_fignums = display_matplotlib_new_figs(prev_fignums)
            discover_and_display_fig_objects(ns)
            plt.close('all'); gc.collect()
            rendered_any = True

    if not rendered_any:
        st.info("No se detectaron figuras ni textos. Asegura que tu notebook genere gráficos (Matplotlib/Plotly).")
    else:
        st.success(f"Notebook ejecutado. Celdas: {total}. Errores: {errors}.")
