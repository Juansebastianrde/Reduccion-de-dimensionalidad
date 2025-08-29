
# streamlit run hdhi_final_app.py
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

# Plotly optional (we'll prefer to have it installed, but handle fallback gracefully)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except Exception:
    px = None; go = None; pio = None
    HAS_PLOTLY = False

import nbformat

st.set_page_config(page_title="HDHI Admission — Final App", layout="wide")

# --------------------- Helpers ---------------------
@st.cache_data
def load_csv_robust():
    """Try to load the dataset from common filenames in repo root, else raise."""
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

def ensure_df_loaded():
    if "df" not in st.session_state or st.session_state.df is None:
        st.info("Primero carga datos en la sección '1) Cargar base de datos'.")
        st.stop()

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
    """Find and render fig-like objects in the namespace."""
    try:
        from matplotlib.figure import Figure as MplFigure
    except Exception:
        MplFigure = None
    for name, obj in list(ns.items()):
        # Matplotlib Figure
        if MplFigure is not None and isinstance(obj, MplFigure):
            st.pyplot(obj)
        # Seaborn grids (FacetGrid/JointGrid) usually store .fig
        if hasattr(obj, "fig") and hasattr(obj.fig, "savefig"):
            try: st.pyplot(obj.fig)
            except Exception: pass
        # Plotly Figure
        if HAS_PLOTLY and (hasattr(obj, "to_dict") and obj.__class__.__name__.endswith("Figure")):
            try: st.plotly_chart(obj, use_container_width=True)
            except Exception: pass

def patch_show_functions(ns):
    """Patch plt.show() and plotly.io.show() to render inside Streamlit."""
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
            continue  # strip magics and shell
        lines.append(line)
    return "\n".join(lines)

# --------------------- UI ---------------------
# Soft card look & select placeholders like screenshot
st.markdown("""
<style>
.card {
  background: #f6f8fb;
  border-radius: 14px;
  padding: 16px 18px;
}
.card h3 { font-size: 1.1rem; margin: 0 0 10px 2px; color: #0f172a; font-weight: 700; }
div[data-baseweb="select"] > div { border-radius: 12px !important; }
.filter-item { margin-bottom: 14px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Navegación")
page = st.sidebar.radio("Secciones", ["1) Cargar base de datos", "2) Filtros", "3) Análisis (ejecutar notebook)"], index=0)

st.title("HDHI — Streamlit (Final)")

# Persistencia
if "df" not in st.session_state:
    st.session_state.df = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None

# 1) Cargar base
if page == "1) Cargar base de datos":
    st.header("1) Cargar base de datos")
    colA, colB = st.columns([1,1])
    with colA:
        st.write("Carga automática (busca 'HDHI Admission data.csv' en la raíz):")
        if st.button("Cargar automáticamente"):
            try:
                bd = load_csv_robust()
                st.session_state.df = bd.copy()
                st.success("Datos cargados automáticamente.")
                st.dataframe(bd.head(), use_container_width=True)
            except Exception as e:
                st.error(str(e))
    with colB:
        st.write("O sube un CSV:")
        up = st.file_uploader("Sube tu archivo .csv", type=["csv"], key="csv_up")
        if up is not None:
            try:
                bd = pd.read_csv(up)
                st.session_state.df = bd.copy()
                st.success("Datos cargados desde el archivo subido.")
                st.dataframe(bd.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error leyendo el CSV: {e}")

# 2) Filtros (GENDER/RURAL + opcionales)
elif page == "2) Filtros":
    st.header("2) Filtros")
    ensure_df_loaded()
    df = st.session_state.df.copy()

    # Build filter card
    card_col, _ = st.columns([1.2, 2])
    with card_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Filters</h3>", unsafe_allow_html=True)

        # GENDER
        gender_value = None
        if "GENDER" in df.columns:
            gvals = df["GENDER"].dropna().unique().tolist()
            gopts = ["Choose an option"] + [pretty_gender(v) for v in gvals]
            gsel_label = st.selectbox("Gender", gopts, index=0, key="flt_gender")
            if gsel_label != "Choose an option":
                # reverse map
                rev = {pretty_gender(v): v for v in gvals}
                gender_value = rev.get(gsel_label, gsel_label)
        else:
            st.selectbox("Gender", ["Choose an option"], index=0, key="flt_gender_disabled")

        # RURAL (Urban/Rural)
        rural_value = None
        if "RURAL" in df.columns:
            rvals = df["RURAL"].dropna().unique().tolist()
            ropts = ["Choose an option"] + [pretty_rural(v) for v in rvals]
            rsel_label = st.selectbox("Urban/Rural", ropts, index=0, key="flt_rural")
            if rsel_label != "Choose an option":
                rev = {pretty_rural(v): v for v in rvals}
                rural_value = rev.get(rsel_label, rsel_label)
        else:
            st.selectbox("Urban/Rural", ["Choose an option"], index=0, key="flt_rural_disabled")

        st.markdown('</div>', unsafe_allow_html=True)

    # Apply filters
    df_view = df.copy()
    if gender_value is not None:
        df_view = df_view[df_view["GENDER"] == gender_value]
    if rural_value is not None:
        df_view = df_view[df_view["RURAL"] == rural_value]

    st.session_state.filtered_df = df_view
    st.metric("Filas después de filtrar", len(df_view))
    st.dataframe(df_view.head(50), use_container_width=True)

# 3) Ejecutar notebook (sin mostrar código)
elif page == "3) Análisis (ejecutar notebook)":
    st.header("3) Análisis y visualizaciones")
    ensure_df_loaded()

    # Path del notebook (por defecto el que subiste)
    nb_default = "/mnt/data/Proyecto_ML (1).ipynb"
    nb_path = st.text_input("Ruta del notebook .ipynb (no se mostrará el código)", value=nb_default)

    if not os.path.exists(nb_path):
        st.error(f"No se encontró el notebook en: {nb_path}")
        st.stop()

    df_base = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.df
    st.caption(f"Ejecutando el notebook sobre el DataFrame actual (shape={df_base.shape}).")

    # Namespace con librerías comunes y DF inyectado con nombres típicos
    ns = {
        "np": np, "pd": pd, "plt": plt, "st": st,
        "stats": stats,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline, "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer, "StandardScaler": StandardScaler,
        "RFE": RFE, "SelectKBest": SelectKBest, "f_regression": f_regression,
        "LinearRegression": LinearRegression, "PCA": PCA,
        "RandomForestRegressor": RandomForestRegressor,
        "px": px, "go": go, "pio": pio,
        # DF con varios alias
        "df": df_base.copy(), "bd": df_base.copy(),
        "data": df_base.copy(), "dataset": df_base.copy(),
        "io": io,
    }
    # parchear show
    patch_show_functions(ns)

    # Leer notebook
    nb = nbformat.read(nb_path, as_version=4)

    prev_fignums = set(plt.get_fignums())
    total_cells = 0
    error_cells = 0

    progress = st.progress(0.0)
    rendered_anything = False

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        total_cells += 1
        src = clean_code(cell.source or "")
        if not src.strip():
            progress.progress(min(1.0, i/len(nb.cells)))  # step
            continue

        # Capturar stdout
        stdout_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf):
                exec(src, ns, ns)
        except Exception as e:
            error_cells += 1
            with st.expander(f"⚠️ Error en una celda (click para ver detalle)", expanded=False):
                st.error(str(e))
                st.code(traceback.format_exc())
        finally:
            out_text = stdout_buf.getvalue().strip()
            if out_text:
                with st.expander("📝 Salida de texto (conclusiones / prints)", expanded=False):
                    st.text(out_text)
                rendered_anything = True

            # Mostrar figuras nuevas
            prev_fignums = display_matplotlib_new_figs(prev_fignums)
            # Detectar otras figuras en variables
            discover_and_display_fig_objects(ns)
            rendered_anything = True

            # Cerrar figuras acumuladas para liberar memoria
            plt.close('all'); gc.collect()

        progress.progress(min(1.0, (i+1)/len(nb.cells)))

    if not rendered_anything:
        st.info("No se detectaron figuras ni textos. Si tu notebook genera gráficos, asegúrate de que use Matplotlib o Plotly.")
    else:
        st.success(f"Notebook ejecutado. Celdas ejecutadas: {total_cells}. Errores: {error_cells}.")

