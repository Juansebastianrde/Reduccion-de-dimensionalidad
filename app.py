
# streamlit run app_from_notebook_full.py
# Streamlit app generated from your notebook (robust figure capture)

import os
import io
import types
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
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

# Optional Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except Exception:
    px = None
    go = None
    pio = None
    HAS_PLOTLY = False

import nbformat
import traceback

st.set_page_config(page_title="HDHI Admission — Notebook App (Full)", layout="wide")

# ---------------- Helpers ----------------
@st.cache_data
def load_csv_robust():
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
    """Find fig-like objects left in the namespace (matplotlib figures, seaborn grids, plotly figs)."""
    try:
        from matplotlib.figure import Figure as MplFigure
    except Exception:
        MplFigure = None
    for name, obj in list(ns.items()):
        # Matplotlib Figure
        if MplFigure is not None and isinstance(obj, MplFigure):
            st.pyplot(obj)
        # Seaborn FacetGrid or JointGrid has .fig
        if hasattr(obj, "fig") and hasattr(obj.fig, "savefig"):
            try:
                st.pyplot(obj.fig)
            except Exception:
                pass
        # Plotly Figure
        if HAS_PLOTLY and (hasattr(obj, "to_dict") and obj.__class__.__name__.endswith("Figure")):
            try:
                st.plotly_chart(obj, use_container_width=True)
            except Exception:
                pass

def patch_show_functions(ns):
    """Monkey-patch plt.show() and plotly.io.show() to render within Streamlit."""
    def _plt_show(*args, **kwargs):
        fig = plt.gcf()
        try:
            st.pyplot(fig)
        except Exception:
            pass
    ns['plt'].show = _plt_show
    if HAS_PLOTLY and 'pio' in ns:
        def _plotly_show(fig=None, *args, **kwargs):
            if fig is None:
                return
            try:
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        ns['pio'].show = _plotly_show

# ---------------- UI ----------------
st.sidebar.header("⚙️ Controles")
app_mode = st.sidebar.radio(
    "Secciones",
    ["1) Cargar base de datos", "2) Filtros (GENDER / RURAL)", "3) Ejecutar Notebook completo"],
    index=0
)

st.title("HDHI — App desde Notebook (Full)")

if "df" not in st.session_state:
    st.session_state.df = None
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = None

# 1) Cargar base
if app_mode == "1) Cargar base de datos":
    st.header("1) Cargar base de datos")
    try:
        bd = load_csv_robust()
        st.session_state.df = bd.copy()
        st.success("Datos cargados.")
        st.dataframe(bd.head(), use_container_width=True)
    except Exception as e:
        st.error(str(e))
        st.stop()

# 2) Filtros
elif app_mode == "2) Filtros (GENDER / RURAL)":
    st.header("2) Filtros por GENDER y RURAL")
    ensure_df_loaded()
    df = st.session_state.df.copy()

    missing = [c for c in ["GENDER", "RURAL"] if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas requeridas: {missing}")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        gvals = sorted(df["GENDER"].dropna().unique().tolist(), key=lambda x: str(x))
        map_g = {v: pretty_gender(v) for v in gvals}
        sel_g_labels = st.multiselect("GENDER", [map_g[v] for v in gvals], default=[map_g[v] for v in gvals])
        sel_g = [k for k,v in map_g.items() if v in sel_g_labels]
    with c2:
        rvals = sorted(df["RURAL"].dropna().unique().tolist(), key=lambda x: str(x))
        map_r = {v: pretty_rural(v) for v in rvals}
        sel_r_labels = st.multiselect("RURAL (Urbano/Rural)", [map_r[v] for v in rvals], default=[map_r[v] for v in rvals])
        sel_r = [k for k,v in map_r.items() if v in sel_r_labels]

    filtered = df[df["GENDER"].isin(sel_g) & df["RURAL"].isin(sel_r)].copy()
    st.session_state.filtered_df = filtered

    m1, m2, m3 = st.columns(3)
    m1.metric("Filas totales", len(df))
    m2.metric("Filtradas", len(filtered))
    m3.metric("Columnas", filtered.shape[1])
    st.dataframe(filtered.head(50), use_container_width=True)

    grp = filtered.groupby(["GENDER","RURAL"]).size().reset_index(name="count")
    if len(grp) > 0:
        if HAS_PLOTLY:
            fig = px.bar(grp, x="GENDER", y="count", color="RURAL", barmode="group", title="Conteo por GENDER y RURAL")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig2, ax2 = plt.subplots()
            for r in sorted(grp['RURAL'].unique(), key=lambda x: str(x)):
                sub = grp[grp['RURAL']==r]
                ax2.bar(sub['GENDER'].astype(str), sub['count'], label=str(r))
            ax2.set_title("Conteo por GENDER y RURAL")
            ax2.legend(title="RURAL")
            st.pyplot(fig2)

# 3) Ejecutar Notebook completo
elif app_mode == "3) Ejecutar Notebook completo":
    st.header("3) Ejecutar Notebook completo")
    ensure_df_loaded()

    nb_path = st.text_input("Ruta del notebook .ipynb", value="/mnt/data/Proyecto_ML (1).ipynb")
    show_code = st.checkbox("Mostrar código de cada celda", value=False)

    if not os.path.exists(nb_path):
        st.error(f"No se encontró el notebook en: {nb_path}")
        st.stop()

    st.info("Las celdas correrán sobre el DataFrame filtrado si existe; de lo contrario, se usa el completo.")
    df_base = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.df

    # Build namespace
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
        # DataFrames disponibles con nombres comunes
        "df": df_base.copy(), "bd": df_base.copy(),
        "data": df_base.copy(), "dataset": df_base.copy(),
        "io": io,
    }

    # Patch show functions to display figures inline
    patch_show_functions(ns)

    # Read notebook
    nb = nbformat.read(nb_path, as_version=4)

    # Clean helper
    def clean_code(src: str) -> str:
        cleaned = []
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("%") or s.startswith("%%") or s.startswith("!"):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    prev_fignums = set(plt.get_fignums())

    cell_idx = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell_idx += 1
        src = clean_code(cell.source or "")
        if not src.strip():
            continue

        st.markdown(f"#### Celda {cell_idx}")
        if show_code:
            st.code(src, language="python")

        stdout_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf):
                exec(src, ns, ns)
        except Exception as e:
            st.error(f"Error en la celda {cell_idx}: {e}")
            st.code(traceback.format_exc())
        out_text = stdout_buf.getvalue().strip()
        if out_text:
            st.text(out_text)

        # Show any new Matplotlib figures
        prev_fignums = display_matplotlib_new_figs(prev_fignums)

        # Discover fig-like objects in namespace (Plotly, Seaborn grids, Mpl Figures)
        discover_and_display_fig_objects(ns)

    st.success("Ejecución del notebook finalizada.")
