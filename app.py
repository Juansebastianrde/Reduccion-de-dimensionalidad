# app.py
# Streamlit ML + PCA App (inspirada en NHANES PCA) adaptada a tu flujo
# Autor: Tú :)
# Uso: streamlit run app.py

import io
import json
import pickle
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Modelos
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Visualizaciones
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Opcional (para MCA de variables categóricas, si lo quieres activar)
try:
    import prince  # pip install prince
    HAS_PRINCE = True
except Exception:
    HAS_PRINCE = False


# -----------------------
# Utilidades
# -----------------------
def load_sample_data() -> pd.DataFrame:
    """Dataset de ejemplo si el usuario no sube CSV."""
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True)
    df = d.frame.copy()
    df.rename(columns={"target": "DURATION OF STAY"}, inplace=True)  # para simular tu target
    return df

def detect_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols

def safe_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def make_download_button(data: bytes, filename: str, label: str):
    st.download_button(label, data=data, file_name=filename, mime="application/octet-stream")

def build_model(model_name: str, params: dict):
    if model_name == "Linear Regression":
        return LinearRegression(**params)
    if model_name == "Lasso":
        return Lasso(**params)
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

def pca_figures(X_scaled: np.ndarray, feature_names: List[str], n_components: int):
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    # Scree plot
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained))],
                               y=explained,
                               name="Varianza explicada"))
    fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(explained))],
                                   y=np.cumsum(explained),
                                   mode="lines+markers",
                                   name="Acumulada"))
    fig_scree.update_layout(title="Scree Plot (PCA)", xaxis_title="Componente", yaxis_title="Proporción")

    # Loadings (coeficientes)
    loadings = pca.components_.T  # shape: [features, components]
    load_df = pd.DataFrame(loadings, index=feature_names, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])

    # Biplot (PC1 vs PC2 si existen)
    fig_biplot = None
    if n_components >= 2:
        fig_biplot = go.Figure()
        fig_biplot.add_trace(go.Scatter(x=scores[:, 0], y=scores[:, 1],
                                        mode="markers",
                                        name="Scores"))
        # vectores de loadings
        scale = 3.0  # factor para visualizar flechas
        for i, feat in enumerate(feature_names):
            fig_biplot.add_trace(go.Scatter(
                x=[0, loadings[i, 0]*scale],
                y=[0, loadings[i, 1]*scale],
                mode="lines+markers",
                name=feat,
                showlegend=False
            ))
        fig_biplot.update_layout(title="Biplot (PC1 vs PC2)",
                                 xaxis_title="PC1", yaxis_title="PC2")

    return pca, scores, explained, load_df, fig_scree, fig_biplot


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="ML + PCA App", layout="wide")

st.title("📊 ML + PCA App (inspirada en NHANES) · Tu proyecto")
st.caption("Sube tu CSV, elige target y configura el preprocesamiento, PCA y modelo. Exporta cargas PCA, predicciones y el modelo.")

# Sidebar: datos
st.sidebar.header("1) Datos")
uploaded = st.sidebar.file_uploader("Sube tu CSV (separador por defecto: coma)", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        # Reintento común: punto y coma
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=";")
else:
    st.sidebar.info("Sin CSV: usando dataset de ejemplo.")
    df = load_sample_data()

st.subheader("Vista rápida de datos")
st.write(df.head())
st.write("Forma:", df.shape)

# Selección de target
default_target = "DURATION OF STAY" if "DURATION OF STAY" in df.columns else None
target = st.selectbox("Variable objetivo (target)", options=[None] + df.columns.tolist(), index=(df.columns.tolist().index(default_target) + 1) if default_target else 0)

if target is None:
    st.warning("Selecciona la variable objetivo para continuar con el entrenamiento de modelos.")
else:
    # Detectar tipos
    num_cols_all, cat_cols_all = detect_types(df.drop(columns=[target], errors="ignore"))

    st.sidebar.header("2) Selección de variables")
    with st.sidebar.expander("Columnas numéricas", expanded=True):
        num_cols = st.multiselect("Numéricas a usar", num_cols_all, default=num_cols_all)
    with st.sidebar.expander("Columnas categóricas", expanded=True):
        cat_cols = st.multiselect("Categóricas a usar", cat_cols_all, default=cat_cols_all)

    # Preprocesamiento
    st.sidebar.header("3) Preprocesamiento")
    imp_num_strategy = st.sidebar.selectbox("Imputación numérica", ["mean", "median"], index=0)
    imp_cat_strategy = st.sidebar.selectbox("Imputación categórica", ["most_frequent", "constant"], index=0)
    use_scaler = st.sidebar.checkbox("Estandarizar numéricas (z-score)", value=True)
    add_polynomial = st.sidebar.checkbox("Añadir rasgos polinomiales (numéricas)", value=False)
    poly_degree = st.sidebar.slider("Grado polinomial", 2, 4, 2, disabled=not add_polynomial, help="Aplica solo a numéricas")

    # Split
    st.sidebar.header("4) Partición Train/Test")
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    # PCA
    st.sidebar.header("5) PCA (opcional, numéricas)")
    do_pca = st.sidebar.checkbox("Activar PCA", value=False)
    pca_components = st.sidebar.slider("N° componentes (PCA)", 2, max(2, len(num_cols) if num_cols else 2), min(5, len(num_cols) if num_cols else 2), 1, disabled=not do_pca)

    # (Opcional) MCA
    st.sidebar.header("6) MCA (solo categóricas, opcional)")
    do_mca = st.sidebar.checkbox("Activar MCA (si prince disponible)", value=False, disabled=not HAS_PRINCE)
    mca_components = st.sidebar.slider("N° componentes (MCA)", 2, max(2, len(cat_cols) if cat_cols else 2), 2, 1, disabled=(not do_mca or not HAS_PRINCE))

    # Modelo
    st.sidebar.header("7) Modelo")
    model_name = st.sidebar.selectbox(
        "Selecciona modelo",
        [
            "Linear Regression",
            "Lasso",
            "ElasticNet",
            "KNN Regressor",
            "Decision Tree",
            "SVR (RBF)",
            "Random Forest",
            "Gradient Boosting",
        ],
        index=6  # Random Forest por defecto
    )

    # Hiperparámetros UI
    params = {}
    if model_name in ("Lasso", "ElasticNet"):
        params["alpha"] = st.sidebar.number_input("alpha", 0.0001, 10.0, 1.0, 0.1)
        if model_name == "ElasticNet":
            params["l1_ratio"] = st.sidebar.slider("l1_ratio", 0.0, 1.0, 0.5, 0.05)
        params["max_iter"] = st.sidebar.number_input("max_iter", 100, 100000, 10000, 100)
    elif model_name == "KNN Regressor":
        params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 50, 5, 1)
        params["weights"] = st.sidebar.selectbox("weights", ["uniform", "distance"], 0)
    elif model_name == "Decision Tree":
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 50, 8, 1)
        params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
    elif model_name == "SVR (RBF)":
        params["C"] = st.sidebar.number_input("C", 0.1, 1000.0, 10.0, 0.1)
        params["epsilon"] = st.sidebar.number_input("epsilon", 0.0, 5.0, 0.1, 0.1)
        params["gamma"] = st.sidebar.selectbox("gamma", ["scale", "auto"], 0)
        params["kernel"] = "rbf"
    elif model_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 1000, 400, 50)
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 50, 12, 1)
        params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
        params["n_jobs"] = -1
        params["random_state"] = random_state
    elif model_name == "Gradient Boosting":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 1000, 400, 50)
        params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.001, 1.0, 0.01, 0.001, format="%.3f")
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 20, 6, 1)
        params["subsample"] = st.sidebar.slider("subsample", 0.1, 1.0, 0.8, 0.1)
        params["random_state"] = random_state
    else:
        params = {}

    # Construcción X/y
    work_cols = [c for c in (num_cols + cat_cols) if c in df.columns and c != target]
    if len(work_cols) == 0:
        st.error("No hay columnas de entrada seleccionadas. Elige al menos una columna numérica o categórica.")
        st.stop()

    y = df[target]
    X = df[work_cols].copy()

    # Preprocesamiento ColumnTransformer
    numeric_transformers = []
    numeric_transformers.append(("imputer", SimpleImputer(strategy=imp_num_strategy)))
    if use_scaler:
        numeric_transformers.append(("scaler", StandardScaler()))
    if add_polynomial:
        numeric_transformers.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    num_pipe = Pipeline(steps=numeric_transformers)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imp_cat_strategy, fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, [c for c in num_cols if c in X.columns]),
            ("cat", cat_pipe, [c for c in cat_cols if c in X.columns]),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=random_state)

    model = build_model(model_name, params)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    # Métricas
    y_pred = pipe.predict(X_test)
    rmse = safe_rmse(y_test, y_pred)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    st.subheader("Resultados del modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE (test)", f"{rmse:,.4f}")
    c2.metric("MAE (test)", f"{mae:,.4f}")
    c3.metric("R² (test)", f"{r2:,.4f}")

    # Gráfico de residuos
    st.markdown("**Residuos (y_pred vs. y_true)**")
    fig_res = px.scatter(x=y_test, y=y_pred - y_test, labels={"x": "y_true", "y": "residuo"})
    fig_res.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

    # Importancias (si el modelo lo soporta)
    try:
        # Recuperar nombres de features expandido tras preprocesamiento
        pre_fit = pipe.named_steps["pre"]
        feature_names = pre_fit.get_feature_names_out().tolist()
        if hasattr(pipe.named_steps["model"], "feature_importances_"):
            importances = pipe.named_steps["model"].feature_importances_
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(30)
            st.markdown("**Top 30 features por importancia**")
            st.dataframe(imp_df, use_container_width=True)
            fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h")
            st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.info("No se pudieron calcular importancias de características.")

    # PCA (sobre numéricas ya imputadas/estandarizadas)
    st.subheader("PCA (numéricas)")
    if do_pca and len(num_cols) >= 2:
        # Creamos un pipeline solo numérico para obtener la matriz transformada limpia
        num_only = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=imp_num_strategy)),
            ("scaler", StandardScaler())  # Forzamos estandarización en PCA
        ])
        X_num = X[num_cols].copy()
        X_num_scaled = num_only.fit_transform(X_num)

        pca, scores, explained, load_df, fig_scree, fig_biplot = pca_figures(
            X_scaled=X_num_scaled,
            feature_names=num_cols,
            n_components=pca_components
        )

        st.plotly_chart(fig_scree, use_container_width=True)
        if fig_biplot is not None:
            st.plotly_chart(fig_biplot, use_container_width=True)

        st.markdown("**Cargas (loadings)**")
        st.dataframe(load_df.style.format("{:.3f}"), use_container_width=True)

        # Descargas PCA
        load_csv = load_df.to_csv().encode("utf-8")
        make_download_button(load_csv, "pca_loadings.csv", "⬇️ Descargar cargas PCA (.csv)")

        scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
        scores_csv = scores_df.to_csv(index=False).encode("utf-8")
        make_download_button(scores_csv, "pca_scores.csv", "⬇️ Descargar scores PCA (.csv)")

        exp_df = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(len(explained))],
            "ExplainedVarianceRatio": explained,
            "Cumulative": np.cumsum(explained)
        })
        exp_csv = exp_df.to_csv(index=False).encode("utf-8")
        make_download_button(exp_csv, "pca_variance.csv", "⬇️ Descargar varianza explicada (.csv)")
    else:
        st.info("Activa PCA en la barra lateral para ver scree plot, biplot y cargas (requiere ≥2 numéricas).")

    # MCA (opcional) si hay categóricas y prince disponible
    if do_mca and HAS_PRINCE and len(cat_cols) >= 2:
        st.subheader("MCA (categóricas)")
        try:
            X_cat = X[cat_cols].astype("category").copy()
            mca = prince.MCA(n_components=mca_components, random_state=42)
            mca_scores = mca.fit_transform(X_cat)

            # Varianza explicada aprox (inercia)
            eig = mca.eigenvalues_
            total = np.sum(eig)
            explained_mca = eig / total if total > 0 else np.zeros_like(eig)

            fig_mca = px.scatter(x=mca_scores.iloc[:, 0], y=mca_scores.iloc[:, 1],
                                 labels={"x": "Dim 1", "y": "Dim 2"},
                                 title="MCA - Dim 1 vs Dim 2")
            st.plotly_chart(fig_mca, use_container_width=True)

            st.markdown("**Coordenadas de variables (contribuciones)**")
            coords = mca.column_coordinates(X_cat)
            st.dataframe(coords, use_container_width=True)

            # Descargas MCA
            scores_csv = mca_scores.to_csv(index=False).encode("utf-8")
            make_download_button(scores_csv, "mca_scores.csv", "⬇️ Descargar scores MCA (.csv)")
            coords_csv = coords.to_csv().encode("utf-8")
            make_download_button(coords_csv, "mca_coords.csv", "⬇️ Descargar coordenadas MCA (.csv)")
        except Exception as e:
            st.warning(f"No se pudo calcular MCA: {e}")
    elif do_mca and not HAS_PRINCE:
        st.info("Instala 'prince' para activar MCA: pip install prince")

    # Descarga de predicciones y del modelo
    st.subheader("Descargas de salida")
    preds_df = pd.DataFrame({
        "y_true": y_test.reset_index(drop=True),
        "y_pred": pd.Series(y_pred).reset_index(drop=True),
        "residuo": pd.Series(y_pred).reset_index(drop=True) - y_test.reset_index(drop=True),
    })
    st.dataframe(preds_df.head(20), use_container_width=True)

    preds_csv = preds_df.to_csv(index=False).encode("utf-8")
    make_download_button(preds_csv, "predicciones.csv", "⬇️ Descargar predicciones (.csv)")

    # Serializar pipeline entrenado
    model_bytes = io.BytesIO()
    pickle.dump(pipe, model_bytes)
    model_bytes.seek(0)
    make_download_button(model_bytes.getvalue(), "modelo_entrenado.pkl", "⬇️ Descargar modelo entrenado (.pkl)")

    # Guardar configuración como JSON
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
        "model": model_name,
        "params": params,
        "pca_enabled": do_pca,
        "pca_components": int(pca_components) if do_pca else None,
        "mca_enabled": do_mca and HAS_PRINCE,
        "mca_components": int(mca_components) if (do_mca and HAS_PRINCE) else None,
    }
    cfg_bytes = json.dumps(config, indent=2).encode("utf-8")
    make_download_button(cfg_bytes, "config.json", "⬇️ Descargar configuración (.json)")

st.markdown("---")
st.caption("Tip: si tu CSV trae `DURATION OF STAY`, la app lo usa por defecto como target. Asegúrate de estandarizar antes de PCA.")
