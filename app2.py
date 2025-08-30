# Notebook → Streamlit runner (1:1 por celdas)
# Pega este código en app.py y ejecútalo con: streamlit run app.py

import streamlit as st
st.set_page_config(page_title="Notebook → Streamlit (1:1)", layout="wide")
st.title("Notebook → Streamlit (réplica celda por celda)")

import io, sys, re, ast, types
from pathlib import Path
from datetime import datetime

# Dependencias comunes
import pandas as pd
import matplotlib.pyplot as plt

# --- Compat shims: hacer que print/display/plt.show/plotly.show funcionen en Streamlit ---
# 1) print -> Streamlit
print = st.write  # noqa: F811

# 2) display() tipo Jupyter
def display(*objs):
    for obj in objs:
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                st.dataframe(obj)
            else:
                st.write(obj)
        except Exception:
            st.text(repr(obj))

# 3) Markdown/HTML helpers si el notebook los usa
class Markdown(str):
    pass
class HTML(str):
    pass

# 4) Matplotlib: mostrar figuras cuando se llama plt.show()
def _plt_show(*args, **kwargs):
    st.pyplot(plt.gcf(), clear_figure=False)
plt.show = _plt_show

# 5) Plotly: fig.show() -> st.plotly_chart
try:
    import plotly.graph_objects as go
    def _plotly_show(self, *_, **__):
        st.plotly_chart(self, use_container_width=True)
    go.Figure.show = _plotly_show
except Exception:
    pass

# --- Utilidades ---
def sanitize_code(code: str) -> str:
    """Quita magics de Jupyter y comandos shell; mantiene el resto igual."""
    out = []
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("%") or s.startswith("!"):
            out.append(f"# [omitido del notebook] {ln}")
        elif "get_ipython()" in ln:
            out.append(f"# [omitido get_ipython] {ln}")
        else:
            out.append(ln)
    return "\n".join(out)

def exec_cell(code: str, ns: dict, show_last_expr: bool = True):
    """Ejecuta una celda de código en el mismo namespace. Si la última línea es expresión, la muestra."""
    clean = sanitize_code(code)
    if not show_last_expr:
        exec(clean, ns)
        return

    try:
        tree = ast.parse(clean, mode="exec")
        if not tree.body:
            return
        # Si la última sentencia es una expresión, la evaluamos y mostramos estilo notebook
        last_is_expr = isinstance(tree.body[-1], ast.Expr)
        if last_is_expr:
            body = ast.Module(body=tree.body[:-1], type_ignores=[])
            last = ast.Expression(body=tree.body[-1].value)
            compiled_body = compile(body, filename="<cell>", mode="exec")
            compiled_last = compile(last, filename="<cell>", mode="eval")
            exec(compiled_body, ns)
            val = eval(compiled_last, ns)
            if val is not None:
                display(val)
        else:
            exec(clean, ns)
    except Exception:
        # Reintento simple sin AST por si hay constructs especiales
        exec(clean, ns)

# --- UI: subir .ipynb o usar ruta local ---
st.sidebar.header("Cargar notebook")
uploaded = st.sidebar.file_uploader("Arrastra tu .ipynb aquí", type=["ipynb"])
default_path = Path("/mnt/data/Proyecto_ML (1).ipynb")
use_default = False

if uploaded is None:
    if default_path.exists():
        use_default = st.sidebar.checkbox(f"Usar notebook detectado: {default_path.name}", value=True)
        if use_default:
            st.sidebar.success(f"Usando: {default_path}")
    else:
        st.info("Sube un archivo .ipynb para iniciar (o coloca tu ruta local en el siguiente cuadro).")

manual_path = st.sidebar.text_input("...o escribe una ruta local a .ipynb (opcional)")

stop_on_error = st.sidebar.checkbox("Detener al primer error de ejecución", value=False)
show_code = st.sidebar.checkbox("Mostrar el código de cada celda", value=True)

# --- Cargar notebook ---
nb_bytes = None
nb_label = None
if uploaded is not None:
    nb_bytes = uploaded.read()
    nb_label = uploaded.name
elif use_default and default_path.exists():
    nb_bytes = default_path.read_bytes()
    nb_label = str(default_path)
elif manual_path:
    p = Path(manual_path).expanduser()
    if p.exists():
        nb_bytes = p.read_bytes()
        nb_label = str(p)

if nb_bytes is None:
    st.stop()

import nbformat
nb = nbformat.read(io.BytesIO(nb_bytes), as_version=4)
st.caption(f"Notebook cargado: **{nb_label}** · {len(nb.cells)} celdas · convertido {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Namespace compartido entre celdas (como en Jupyter)
ns = {
    "__name__": "__main__",
    "st": st,
    "pd": pd,
    "plt": plt,
    "display": display,
    "Markdown": Markdown,
    "HTML": HTML,
    "print": print,
}

# Intentar facilitar imports comunes si el notebook los usa más tarde
try:
    import numpy as np
    ns["np"] = np
except Exception:
    pass

# --- Ejecutar celdas en orden ---
error_found = False
for i, cell in enumerate(nb.cells, start=1):
    cell_type = cell.get("cell_type")
    if cell_type == "markdown":
        st.markdown(cell.get("source", ""), unsafe_allow_html=True)
        st.divider()
        continue

    if cell_type != "code":
        continue  # ignorar otros tipos raros

    code = cell.get("source", "") or ""
    if not code.strip():
        continue

    with st.container(border=True):
        st.caption(f"Code cell {i}")
        if show_code:
            st.code(code, language="python")

        try:
            exec_cell(code, ns, show_last_expr=True)
        except Exception as e:
            error_found = True
            st.exception(e)
            if stop_on_error:
                st.error(f"Detenido por error en la celda {i}. Activa 'Detener al primer error' para continuar tras errores.")
                break

# Fin
st.success("Ejecución completada." if not error_found else "Ejecución finalizada con errores (revisa las celdas marcadas).")
