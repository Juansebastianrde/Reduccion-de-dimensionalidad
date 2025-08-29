# app.py
# Streamlit ML App - "Proyecto ML" adaptada desde tu notebook
# Autor: Tú :)
# Uso: streamlit run app.py

import io
import json
import pickle
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from scipy import stats
from scipy.stats import spearmanr, f_oneway

from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFECV, RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer

# Modelos
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# MCA opcional
try:
    import prince  # pip install prince
    HAS_PRINCE = True
except Exception:
    HAS_PRINCE = False

# -----------------------
# Utilidades generales
# -----------------------
st.set_page_config(page_title="Proyecto ML · Streamlit", layout="wide")

def to_raw_github(url: str) -> str:
    """Convierte un enlace github.com/.../blob/main/... a raw.githubusercontent.com/.../main/...."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")
    return url

@st.cache_data(show_spinner=False)
def load_csv_from_url(url: str) -> pd.DataFrame:
    raw = to_raw_github(url)
    try:
        return pd.read_csv(raw)
    except Exception:
        return pd.read_csv(raw, sep=";", engine="python")

@st.cache_data(show_spinner=False)
def load_csv_from_upload(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=";", engine="python")

def detect_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def download_button_bytes(data: bytes, filename: str, label: str):
    st.download_button(label, data=data, file_name=filename, mime="application/octet-stream")

def learning_curve_fig(estimator, X, y, cv_splits=5, scoring="neg_root_mean_squared_error", train_sizes=None, random_state=42):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=-1
    )
    # Convertir a RMSE positivo
    train_rmse = np.sqrt(np.maximum(0, -train_scores))
    valid_rmse = np.sqrt(np.maximum(0, -valid_scores))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_rmse.mean(axis=1),
                             mode="lines+markers", name="Train RMSE"))
    fig.add_trace(go.Scatter(x=train_sizes, y=valid_rmse.mean(axis=1),
                             mode="lines+markers", name="CV RMSE"))
    fig.update_layout(title="Learning Curve (RMSE)", xaxis_title="N muestras", yaxis_title="RMSE")
    return fig

def pca_figures(X_scaled: np.ndarray, feature_names: List[str], n_components: int):
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    # Scree
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained))],
                               y=explained, name="Explicada"))
    fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(explained))],
                                   y=np.cumsum(explained), mode="lines+markers",
                                   name="Acumulada"))
    fig_scree.update_layout(title="Scree Plot (PCA)",
                            xaxis_title="Componente", yaxis_title="Proporción")

    # Loadings
    loadings = pca.components_.T
    load_df = pd.DataFrame(loadings, index=feature_names,
                           columns=[f"PC{i+1}" for i in range(loadings.shape[1])])

    # Biplot
    fig_biplot = None
    if n_components >= 2:
        fig_biplot = go.Figure()
        fig_biplot.add_trace(go.Scatter(x=scores[:, 0], y=scores[:, 1],
                                        mode="markers", name="Scores"))
        scale = 3.0
        for i, feat in enumerate(feature_names):
            fig_biplot.add_trace(go.Scatter(
                x=[0, loadings[i, 0]*scale],
                y=[0, loadings[i, 1]*scale],
                mode="lines+markers",
                name=feat,
                showlegend=False
            ))
        fig_biplot.update_layout(title="Biplot (PC1 vs PC2)", xaxis_title="PC1", yaxis_title="PC2")

    return pca, scores, explained, load_df, fig_scree, fig_biplot

def build_model(model_name: str, params: dict):
    if model_name == "Linear Regression":
        return LinearRegression(**params)
    if model_name == "Lasso":
        return Lasso(**params)
    if model_name == "Ridge":
        return Ridge(**params)
    if model_name == "ElasticNet":
        return ElasticNet(**params)
    if model_name == "KNN Regressor":
        return KNeighborsRegressor(**params)
    if model_name == "Decision Tree":
        return DecisionTreeRegressor(**params)
    if model_name == "SVR (RBF)":
        return SVR(**params)
    if model_name == "Random Forest":
        return RandomForestRegressor(**params)
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(**params)
    raise ValueError("Modelo no reconocido")

# -----------------------
# Barra lateral: Datos + Config
# -----------------------
st.sidebar.header("1) Datos")
default_url = "https://github.com/Juansebastianrde/Reduccion-de-dimensionalidad/blob/main/HDHI%20Admission%20data.csv"
use_github = st.sidebar.toggle("Usar CSV desde GitHub", value=True)
gh_url = st.sidebar.text_input("Enlace CSV de GitHub", value=default_url,
                               help="Pega un enlace con /blob/. Lo convertimos a raw automáticamente.")
uploaded = None
if not use_github:
    uploaded = st.sidebar.file_uploader("O sube un CSV", type=["csv"])

if use_github and gh_url:
    df = load_csv_from_url(gh_url)
elif uploaded:
    df = load_csv_from_upload(uploaded)
else:
    st.stop()

# -----------------------
# Sección: Base de datos
# -----------------------
st.title("📘 Proyecto ML · App interactiva")
st.markdown("Esta app replica el flujo de tu notebook: **EDA → Selección de variables → PCA/MCA → Modelos → Pruning → Comparación → Descargas**.")

st.header("Base de datos")
st.write(df.head())
st.write("**Forma de la tabla:**", df.shape)

# Selección de target
default_target = "DURATION OF STAY" if "DURATION OF STAY" in df.columns else None
target = st.selectbox(
    "Variable objetivo (target)",
    options=[None] + df.columns.tolist(),
    index=(df.columns.tolist().index(default_target) + 1) if default_target else 0
)
if target is None:
    st.warning("Selecciona la variable objetivo para continuar.")
    st.stop()

num_cols_all, cat_cols_all = detect_types(df.drop(columns=[target], errors="ignore"))

st.sidebar.header("2) Selección de variables")
with st.sidebar.expander("Columnas numéricas", expanded=True):
    num_cols = st.multiselect("Numéricas a usar", num_cols_all, default=num_cols_all)
with st.sidebar.expander("Columnas categóricas", expanded=True):
    cat_cols = st.multiselect("Categóricas a usar", cat_cols_all, default=cat_cols_all)

st.sidebar.header("3) Preprocesamiento")
imp_num_strategy = st.sidebar.selectbox("Imputación numérica", ["mean", "median"], index=0)
imp_cat_strategy = st.sidebar.selectbox("Imputación categórica", ["most_frequent", "constant"], index=0)
use_scaler = st.sidebar.checkbox("Estandarizar numéricas (z-score)", value=True)
add_polynomial = st.sidebar.checkbox("Añadir rasgos polinomiales (numéricas)", value=False)
poly_degree = st.sidebar.slider("Grado polinomial", 2, 4, 2, disabled=not add_polynomial)

st.sidebar.header("4) Train/Test")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
cv_splits = st.sidebar.slider("CV splits (k-fold)", 3, 10, 5, 1)

# Construcción X/y
work_cols = [c for c in (num_cols + cat_cols) if c in df.columns and c != target]
if len(work_cols) == 0:
    st.error("No hay columnas de entrada seleccionadas.")
    st.stop()

y = df[target]
X = df[work_cols].copy()

# -----------------------
# Tabs principales
# -----------------------
tabs = st.tabs([
    "📊 EDA",
    "🧪 Selección de variables",
    "🌀 PCA / MCA",
    "🤖 Modelos",
    "🌲 Árbol: Pruning",
    "⚖️ Comparación de modelos",
    "📥 Descargas"
])

# =======================
# EDA
# =======================
with tabs[0]:
    st.subheader("Análisis exploratorio de datos")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Resumen estadístico (numéricas)**")
        if len(num_cols) > 0:
            st.dataframe(df[num_cols].describe().T, use_container_width=True)
        else:
            st.info("No hay columnas numéricas seleccionadas.")

    with c2:
        st.markdown("**Valores faltantes (%)**")
        miss = df[work_cols + [target]].isna().mean().sort_values(ascending=False) * 100
        st.dataframe(miss.to_frame("faltantes_%"), use_container_width=True)

    st.markdown("**Distribuciones (histogramas)**")
    if len(num_cols) > 0:
        cols_sel = st.multiselect("Selecciona numéricas para histograma", num_cols, default=num_cols[:min(6, len(num_cols))])
        ncols = 3
        nrows = int(np.ceil(len(cols_sel) / ncols)) if cols_sel else 0
        if cols_sel:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4*nrows))
            axes = np.array(axes).reshape(-1) if len(cols_sel) > 1 else [axes]
            for ax, col in zip(axes, cols_sel):
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Histograma: {col}")
            # Ocultar ejes sobrantes
            for ax in axes[len(cols_sel):]:
                ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Selecciona al menos una numérica.")
    else:
        st.info("No hay numéricas para histogramas.")

    st.markdown("**Relaciones bivariadas**")
    if len(num_cols) >= 2:
        x_var = st.selectbox("X", num_cols, index=0)
        y_var = st.selectbox("Y", num_cols, index=min(1, len(num_cols)-1))
        hue = st.selectbox("Color (opcional)", [None] + cat_cols, index=0)
        fig = px.scatter(df, x=x_var, y=y_var, color=hue if hue else None,
                         trendline="ols" if len(df.dropna(subset=[x_var, y_var])) > 5 else None,
                         title=f"{x_var} vs {y_var}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Se requieren ≥2 variables numéricas para scatter.")

    st.markdown("**Correlación (Spearman/Pearson)**")
    if len(num_cols) >= 2:
        corr_method = st.radio("Método de correlación", ["spearman", "pearson"], horizontal=True)
        corr = df[num_cols].corr(method=corr_method)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
        ax.set_title(f"Matriz de correlación ({corr_method})")
        st.pyplot(fig)
    else:
        st.info("Se requieren ≥2 numéricas para correlación.")

# =======================
# Selección de variables
# =======================
with tabs[1]:
    st.subheader("Selección por filtrado")
    # Spearman para numéricas vs target (si target es numérica)
    if pd.api.types.is_numeric_dtype(y):
        st.markdown("**Correlación de Spearman (|rho| con el target)**")
        abs_corr = []
        for col in num_cols:
            s = df[[col, target]].dropna()
            if s.shape[0] > 3:
                rho, p = spearmanr(s[col], s[target])
                abs_corr.append((col, abs(rho), p))
        if abs_corr:
            spearman_df = (pd.DataFrame(abs_corr, columns=["Variable", "abs_corr", "pvalue"])
                           .sort_values("abs_corr", ascending=False))
            st.dataframe(spearman_df, use_container_width=True)
        else:
            st.info("No hay suficientes datos para Spearman.")
    else:
        st.info("El target no es numérico. Salta Spearman.")

    st.markdown("**ANOVA (target numérico vs categorías) — F-test por categoría**")
    # Si hay categóricas: ANOVA unidireccional de y ~ cada cat
    if len(cat_cols) > 0 and pd.api.types.is_numeric_dtype(y):
        rows = []
        for col in cat_cols:
            try:
                groups = [grp.dropna().values for _, grp in df.groupby(col)[target]]
                groups = [g for g in groups if len(g) > 1]
                if len(groups) > 1:
                    F, p = f_oneway(*groups)
                    rows.append((col, F, p))
            except Exception:
                pass
        if rows:
            anova_df = pd.DataFrame(rows, columns=["Variable", "F", "pvalue"]).sort_values("F", ascending=False)
            st.dataframe(anova_df, use_container_width=True)
        else:
            st.info("No fue posible calcular ANOVA para las categóricas seleccionadas.")
    else:
        st.info("ANOVA requiere target numérico y al menos 1 categórica.")

    st.subheader("Selección automática")
    k_feat = st.slider("SelectKBest (f_regression): número de features", 1, max(1, len(work_cols)), min(10, len(work_cols)))
    # Preprocesamiento básico para selectK (numéricas escaladas + OneHot para cats)
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=imp_num_strategy)),
        ("scaler", StandardScaler() if use_scaler else "passthrough"),
    ])
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=imp_cat_strategy, fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre_SK = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, [c for c in num_cols if c in X.columns]),
            ("cat", cat_pipeline, [c for c in cat_cols if c in X.columns]),
        ],
        verbose_feature_names_out=False
    )
    X_pre = pre_SK.fit_transform(X)
    try:
        feat_names = pre_SK.get_feature_names_out()
    except Exception:
        feat_names = [f"f_{i}" for i in range(X_pre.shape[1])]
    # SelectKBest
    if pd.api.types.is_numeric_dtype(y) and X_pre.shape[1] >= k_feat:
        skb = SelectKBest(score_func=f_regression, k=k_feat)
        X_sel = skb.fit_transform(X_pre, y)
        scores = skb.scores_
        mask = skb.get_support()
        sel_df = (pd.DataFrame({"feature": feat_names, "score": scores})
                    .sort_values("score", ascending=False))
        st.markdown("**SelectKBest - Mejores features**")
        st.dataframe(sel_df.head(k_feat), use_container_width=True)
    else:
        st.info("SelectKBest requiere target numérico y suficientes features.")

    st.markdown("**RFECV (eliminación recursiva con CV) con LinearRegression**")
    try:
        base_est = LinearRegression()
        rfecv = RFECV(base_est, step=1, cv=KFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
                      scoring="neg_root_mean_squared_error", n_jobs=-1)
        rfecv.fit(X_pre, y)
        mask = rfecv.support_
        ranking = rfecv.ranking_
        rfecv_df = pd.DataFrame({"feature": feat_names, "rank": ranking, "selected": mask})
        st.dataframe(rfecv_df.sort_values(["selected","rank"], ascending=[False, True]), use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(rfecv.cv_results_['mean_test_score'])+1)),
                                 y=-rfecv.cv_results_['mean_test_score'], mode="lines+markers",
                                 name="CV RMSE"))
        fig.update_layout(title="RFECV - CV RMSE vs N° features", xaxis_title="N° features", yaxis_title="RMSE")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo ejecutar RFECV: {e}")

    st.markdown("**Random Forest - Importancias**")
    try:
        rf = RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1)
        rf.fit(X_pre, y)
        importances = rf.feature_importances_
        imp_df = (pd.DataFrame({"feature": feat_names, "importance": importances})
                    .sort_values("importance", ascending=False))
        st.dataframe(imp_df.head(30), use_container_width=True)
        st.plotly_chart(px.bar(imp_df.head(30), x="importance", y="feature", orientation="h"),
                        use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo entrenar RandomForest para importancias: {e}")

# =======================
# PCA / MCA
# =======================
with tabs[2]:
    st.subheader("Reducción de dimensionalidad")

    c1, c2 = st.columns(2)
    with c1:
        do_pca = st.checkbox("Activar PCA (numéricas)", value=False)
        n_pcs = st.slider("N° componentes PCA", 2, max(2, len(num_cols) if num_cols else 2), min(8, max(2, len(num_cols) if num_cols else 2)), 1, disabled=not do_pca)
    with c2:
        do_mca = st.checkbox("Activar MCA (categóricas, requiere 'prince')", value=False, disabled=not HAS_PRINCE)
        n_mca = st.slider("N° componentes MCA", 2, max(2, len(cat_cols) if cat_cols else 2), 2, 1, disabled=(not do_mca or not HAS_PRINCE))

    if do_pca and len(num_cols) >= 2:
        # Imputación + escalado garantizado
        num_only = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=imp_num_strategy)),
            ("scaler", StandardScaler())
        ])
        X_num_scaled = num_only.fit_transform(X[num_cols].copy())
        pca, scores, explained, load_df, fig_scree, fig_biplot = pca_figures(X_num_scaled, num_cols, n_pcs)
        st.plotly_chart(fig_scree, use_container_width=True)
        if fig_biplot is not None:
            st.plotly_chart(fig_biplot, use_container_width=True)

        st.markdown("**Cargas (loadings)**")
        st.dataframe(load_df.style.format("{:.3f}"), use_container_width=True)

        # Descargas
        download_button_bytes(load_df.to_csv().encode("utf-8"), "pca_loadings.csv", "⬇️ Cargas PCA")
        scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
        download_button_bytes(scores_df.to_csv(index=False).encode("utf-8"), "pca_scores.csv", "⬇️ Scores PCA")
        exp_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(explained))],
                               "ExplainedVarianceRatio": explained,
                               "Cumulative": np.cumsum(explained)})
        download_button_bytes(exp_df.to_csv(index=False).encode("utf-8"), "pca_variance.csv", "⬇️ Varianza explicada")
    else:
        st.info("Activa PCA si quieres analizar componentes (requiere ≥2 numéricas).")

    if do_mca and HAS_PRINCE and len(cat_cols) >= 2:
        try:
            st.markdown("**MCA (Multiple Correspondence Analysis)**")
            X_cat = X[cat_cols].astype("category").copy()
            mca = prince.MCA(n_components=n_mca, random_state=42)
            mca_scores = mca.fit_transform(X_cat)
            fig_mca = px.scatter(x=mca_scores.iloc[:, 0], y=mca_scores.iloc[:, 1],
                                 labels={"x": "Dim 1", "y": "Dim 2"}, title="MCA - Dim 1 vs Dim 2")
            st.plotly_chart(fig_mca, use_container_width=True)

            coords = mca.column_coordinates(X_cat)
            st.markdown("**Coordenadas de variables (contribuciones)**")
            st.dataframe(coords, use_container_width=True)

            download_button_bytes(mca_scores.to_csv(index=False).encode("utf-8"), "mca_scores.csv", "⬇️ Scores MCA")
            download_button_bytes(coords.to_csv().encode("utf-8"), "mca_coords.csv", "⬇️ Coordenadas MCA")
        except Exception as e:
            st.warning(f"No se pudo calcular MCA: {e}")
    elif do_mca and not HAS_PRINCE:
        st.info("Instala 'prince' para MCA: pip install prince")

# =======================
# Modelos
# =======================
with tabs[3]:
    st.subheader("Entrenamiento de modelos")

    # Preprocesamiento para modelado
    num_steps = [("imputer", SimpleImputer(strategy=imp_num_strategy))]
    if use_scaler:
        num_steps.append(("scaler", StandardScaler()))
    if add_polynomial:
        num_steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=imp_cat_strategy, fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, [c for c in num_cols if c in X.columns]),
            ("cat", cat_pipe, [c for c in cat_cols if c in X.columns]),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=random_state
    )

    model_name = st.selectbox(
        "Modelo",
        [
            "Linear Regression",
            "Lasso",
            "Ridge",
            "ElasticNet",
            "KNN Regressor",
            "Decision Tree",
            "SVR (RBF)",
            "Random Forest",
            "Gradient Boosting",
        ],
        index=8  # Gradient Boosting por defecto
    )

    # Hiperparámetros
    params = {}
    if model_name in ("Lasso", "Ridge", "ElasticNet"):
        params["alpha"] = st.number_input("alpha", 0.0001, 10.0, 1.0, 0.1)
        if model_name == "ElasticNet":
            params["l1_ratio"] = st.slider("l1_ratio", 0.0, 1.0, 0.5, 0.05)
        params["max_iter"] = st.number_input("max_iter", 100, 100000, 10000, 100)
    elif model_name == "KNN Regressor":
        params["n_neighbors"] = st.slider("n_neighbors", 1, 50, 5, 1)
        params["weights"] = st.selectbox("weights", ["uniform", "distance"], 0)
    elif model_name == "Decision Tree":
        params["max_depth"] = st.slider("max_depth", 1, 50, 8, 1)
        params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2, 1)
        params["random_state"] = random_state
    elif model_name == "SVR (RBF)":
        params["C"] = st.number_input("C", 0.1, 1000.0, 10.0, 0.1)
        params["epsilon"] = st.number_input("epsilon", 0.0, 5.0, 0.1, 0.1)
        params["gamma"] = st.selectbox("gamma", ["scale", "auto"], 0)
        params["kernel"] = "rbf"
    elif model_name == "Random Forest":
        params["n_estimators"] = st.slider("n_estimators", 50, 1000, 400, 50)
        params["max_depth"] = st.slider("max_depth", 1, 50, 12, 1)
        params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 2, 1)
        params["n_jobs"] = -1
        params["random_state"] = random_state
    elif model_name == "Gradient Boosting":
        params["n_estimators"] = st.slider("n_estimators", 50, 1000, 400, 50)
        params["learning_rate"] = st.number_input("learning_rate", 0.001, 1.0, 0.01, 0.001, format="%.3f")
        params["max_depth"] = st.slider("max_depth", 1, 20, 6, 1)
        params["subsample"] = st.slider("subsample", 0.1, 1.0, 0.8, 0.1)
        params["random_state"] = random_state

    model = build_model(model_name, params)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    # Métricas
    y_pred = pipe.predict(X_test)
    _rmse = rmse(y_test, y_pred)
    _mae = float(mean_absolute_error(y_test, y_pred))
    _r2 = float(r2_score(y_test, y_pred))

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (test)", f"{_rmse:,.4f}")
    c2.metric("MAE (test)", f"{_mae:,.4f}")
    c3.metric("R² (test)", f"{_r2:,.4f}")

    st.markdown("**Residuos (y_pred - y_true)**")
    fig_res = px.scatter(x=y_test, y=y_pred - y_test, labels={"x": "y_true", "y": "residuo"})
    fig_res.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

    # Importancias (si disponibles)
    try:
        pre_fit = pipe.named_steps["pre"]
        feature_names = pre_fit.get_feature_names_out().tolist()
        if hasattr(pipe.named_steps["model"], "feature_importances_"):
            importances = pipe.named_steps["model"].feature_importances_
            imp_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
                        .sort_values("importance", ascending=False).head(30))
            st.markdown("**Top 30 features por importancia**")
            st.dataframe(imp_df, use_container_width=True)
            st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h"),
                            use_container_width=True)
    except Exception:
        pass

    st.markdown("**Curva de aprendizaje (RMSE)**")
    try:
        # Construimos un estimador standalone con el mismo preprocesamiento
        est = Pipeline(steps=[("pre", pre), ("model", build_model(model_name, params))])
        lc_fig = learning_curve_fig(est, X, y, cv_splits=cv_splits, scoring="neg_root_mean_squared_error", random_state=random_state)
        st.plotly_chart(lc_fig, use_container_width=True)
    except Exception as e:
        st.info(f"No fue posible generar la learning curve: {e}")

# =======================
# Árbol: Pre/Post Pruning (CCP)
# =======================
with tabs[4]:
    st.subheader("Árbol de decisión · Pre y Post-Pruning (Cost Complexity)")

    # Preprocesamiento simple para árbol (OneHot + imputación)
    num_pipe_tree = Pipeline(steps=[("imputer", SimpleImputer(strategy=imp_num_strategy))])
    cat_pipe_tree = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=imp_cat_strategy, fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre_tree = ColumnTransformer(
        transformers=[
            ("num", num_pipe_tree, [c for c in num_cols if c in X.columns]),
            ("cat", cat_pipe_tree, [c for c in cat_cols if c in X.columns]),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    X_tree = pre_tree.fit_transform(X)
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tree, y, test_size=float(test_size), random_state=random_state)

    # Árbol "grande"
    tree_full = DecisionTreeRegressor(random_state=random_state)
    tree_full.fit(X_train_t, y_train_t)

    # Ruta de complejidad (CCP)
    path = tree_full.cost_complexity_pruning_path(X_train_t, y_train_t)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas = np.round(ccp_alphas, 10)  # estabilidad numérica

    st.markdown("**Trayectoria CCP (impureza vs alpha)**")
    fig_ccp = go.Figure()
    fig_ccp.add_trace(go.Scatter(x=ccp_alphas, y=impurities, mode="lines+markers", name="Impureza"))
    fig_ccp.update_layout(xaxis_title="ccp_alpha", yaxis_title="Impureza", title="Cost Complexity Path")
    st.plotly_chart(fig_ccp, use_container_width=True)

    alpha_sel = st.select_slider("Selecciona ccp_alpha", options=sorted(ccp_alphas.tolist()), value=float(np.median(ccp_alphas)))
    tree_pruned = DecisionTreeRegressor(random_state=random_state, ccp_alpha=alpha_sel)
    tree_pruned.fit(X_train_t, y_train_t)
    y_pred_t = tree_pruned.predict(X_test_t)
    st.write(f"**RMSE (test)** con ccp_alpha={alpha_sel}: {rmse(y_test_t, y_pred_t):,.4f}")
    st.write(f"**R² (test)** con ccp_alpha={alpha_sel}: {r2_score(y_test_t, y_pred_t):,.4f}")

# =======================
# Comparación de modelos
# =======================
with tabs[5]:
    st.subheader("Comparación rápida de modelos")
    models_to_compare = st.multiselect(
        "Selecciona modelos a comparar",
        ["Linear Regression", "Lasso", "Ridge", "ElasticNet", "KNN Regressor", "SVR (RBF)",
         "Decision Tree", "Random Forest", "Gradient Boosting"],
        default=["Linear Regression", "Random Forest", "Gradient Boosting"]
    )

    # Preprocesamiento común
    num_steps_cmp = [("imputer", SimpleImputer(strategy=imp_num_strategy))]
    if use_scaler:
        num_steps_cmp.append(("scaler", StandardScaler()))
    num_pipe_cmp = Pipeline(steps=num_steps_cmp)

    cat_pipe_cmp = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=imp_cat_strategy, fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre_cmp = ColumnTransformer(
        transformers=[
            ("num", num_pipe_cmp, [c for c in num_cols if c in X.columns]),
            ("cat", cat_pipe_cmp, [c for c in cat_cols if c in X.columns]),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y, test_size=float(test_size), random_state=random_state)

    def default_params_for(name: str) -> Dict:
        if name == "SVR (RBF)":
            return {"C": 10.0, "epsilon": 0.1, "gamma": "scale", "kernel": "rbf"}
        if name == "Random Forest":
            return {"n_estimators": 400, "max_depth": 12, "min_samples_split": 2, "n_jobs": -1, "random_state": random_state}
        if name == "Gradient Boosting":
            return {"n_estimators": 400, "learning_rate": 0.01, "max_depth": 6, "subsample": 0.8, "random_state": random_state}
        return {}

    rows = []
    for name in models_to_compare:
        try:
            params = default_params_for(name)
            est = Pipeline(steps=[("pre", pre_cmp), ("model", build_model(name, params))])
            est.fit(X_train_c, y_train_c)
            pred = est.predict(X_test_c)
            rows.append({
                "Modelo": name,
                "RMSE": rmse(y_test_c, pred),
                "MAE": float(mean_absolute_error(y_test_c, pred)),
                "R2": float(r2_score(y_test_c, pred)),
            })
        except Exception as e:
            rows.append({"Modelo": name, "RMSE": np.nan, "MAE": np.nan, "R2": np.nan})
    cmp_df = pd.DataFrame(rows).sort_values("RMSE")
    st.dataframe(cmp_df, use_container_width=True)
    st.plotly_chart(px.bar(cmp_df, x="Modelo", y="RMSE", title="Comparación RMSE"), use_container_width=True)

# =======================
# Descargas
# =======================
with tabs[6]:
    st.subheader("Exportar resultados")

    # Reentrenar un pipeline "final" con Gradient Boosting por defecto (tus HParams)
    num_steps_final = [("imputer", SimpleImputer(strategy=imp_num_strategy))]
    if use_scaler:
        num_steps_final.append(("scaler", StandardScaler()))
    if add_polynomial:
        num_steps_final.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    num_pipe_final = Pipeline(steps=num_steps_final)

    cat_pipe_final = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=imp_cat_strategy, fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre_final = ColumnTransformer(
        transformers=[
            ("num", num_pipe_final, [c for c in num_cols if c in X.columns]),
            ("cat", cat_pipe_final, [c for c in cat_cols if c in X.columns]),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    model_final = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.01, max_depth=6, subsample=0.8, random_state=random_state
    )
    pipe_final = Pipeline(steps=[("pre", pre_final), ("model", model_final)])
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y, test_size=float(test_size), random_state=random_state)
    pipe_final.fit(X_train_f, y_train_f)
    y_pred_f = pipe_final.predict(X_test_f)

    preds_df = pd.DataFrame({
        "y_true": y_test_f.reset_index(drop=True),
        "y_pred": pd.Series(y_pred_f).reset_index(drop=True),
        "residuo": pd.Series(y_pred_f).reset_index(drop=True) - y_test_f.reset_index(drop=True),
    })
    st.markdown("**Vista previa de predicciones**")
    st.dataframe(preds_df.head(30), use_container_width=True)
    download_button_bytes(preds_df.to_csv(index=False).encode("utf-8"), "predicciones.csv", "⬇️ Descargar predicciones (.csv)")

    model_bytes = io.BytesIO()
    pickle.dump(pipe_final, model_bytes)
    model_bytes.seek(0)
    download_button_bytes(model_bytes.getvalue(), "modelo_entrenado.pkl", "⬇️ Descargar modelo entrenado (.pkl)")

    config = {
        "target": target,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "impute_num": imp_num_strategy,
        "impute_cat": imp_cat_strategy,
        "scale_num": use_scaler,
        "poly_features": add_polynomial,
        "poly_degree": poly_degree if add_polynomial else None,
        "test_size": test_size,
        "random_state": random_state,
        "cv_splits": cv_splits,
        "github_url_used": gh_url if use_github else None
    }
    st.download_button("⬇️ Descargar configuración (.json)",
                       data=json.dumps(config, indent=2).encode("utf-8"),
                       file_name="config.json",
                       mime="application/json")

st.markdown("---")
st.caption("Listo. Esta app te permite replicar la lógica de tu notebook: EDA, selección de variables, PCA/MCA, modelos, pruning, comparación y descargas.")

