

# ================================================
# Streamlit App: Exploración, PCA y Modelado (Regresión)
# Basado en el notebook "Proyecto_ML (1).ipynb"
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# (Opcional) Multiple Correspondence Analysis (prince)
try:
    import prince
    HAS_PRINCE = True
except Exception:
    HAS_PRINCE = False

st.set_page_config(page_title="Proyecto ML - Streamlit", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def split_features(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def model_dict(random_state: int = 42):
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(random_state=random_state),
        "ElasticNet": ElasticNet(random_state=random_state),
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(random_state=random_state),
        "SVR (RBF)": SVR(kernel="rbf"),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }

def default_target_guess(df: pd.DataFrame):
    candidates = ["DURATION OF STAY", "target", "Target", "y", "duration", "label"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------- Sidebar ----------
st.sidebar.title("Configuración")
st.sidebar.write("Sube tu CSV o pega una URL al CSV (GitHub row link).")

data_file = st.sidebar.file_uploader("Cargar CSV", type=["csv"])
data_url = st.sidebar.text_input("... o URL del CSV", value="")

if data_file is not None:
    df = load_csv(data_file)
elif data_url.strip():
    try:
        df = load_csv(data_url.strip())
    except Exception as e:
        st.sidebar.error(f"No se pudo cargar desde la URL. Detalle: {e}")
        df = None
else:
    df = None

st.title("Proyecto ML · EDA · PCA · Modelado (Regresión)")
st.caption("Adaptación a Streamlit del notebook proporcionado.")

if df is None:
    st.info("➡️ Carga un CSV para comenzar. Si tu dataset es 'HDHI Admission data.csv', súbelo aquí.")
    st.stop()

# ---------- DATA TAB ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Datos", "EDA", "Transformaciones", "PCA / MCA", "Selección de Variables (RFE)", "Modelado"
])

with tab1:
    st.subheader("Visión General de los Datos")
    st.write("Dimensiones:", df.shape)
    st.dataframe(df.head(30))
    st.write("Tipos de datos:")
    st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str)}))

    st.write("Valores nulos por columna:")
    nulls = df.isnull().sum().sort_values(ascending=False)
    st.dataframe(nulls.to_frame("n_nulos"))

    # Gráfico de nulos
    fig, ax = plt.subplots()
    nulls.plot(kind="bar", ax=ax)
    ax.set_title("Conteo de nulos por columna")
    ax.set_ylabel("Nulos")
    st.pyplot(fig)

with tab2:
    st.subheader("EDA Rápida")
    num_cols, cat_cols = split_features(df)

    st.markdown("**Distribuciones (numéricas)**")
    sel_num = st.multiselect("Selecciona columnas numéricas", num_cols, default=num_cols[:min(6, len(num_cols))])
    if sel_num:
        for c in sel_num:
            fig, ax = plt.subplots()
            sns.histplot(df[c].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribución: {c}")
            st.pyplot(fig)

    st.markdown("**Correlaciones (numéricas)**")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matriz de correlación (Spearman)")
        st.pyplot(fig)

    st.markdown("**Exploración por categorías**")
    if cat_cols:
        col_cat = st.selectbox("Categoría", cat_cols)
        if col_cat:
            fig, ax = plt.subplots()
            df[col_cat].value_counts(dropna=False).head(20).plot(kind="bar", ax=ax)
            ax.set_title(f"Frecuencias: {col_cat}")
            st.pyplot(fig)

with tab3:
    st.subheader("Transformaciones")
    st.write("Puedes aplicar imputación y escalado. Estas transformaciones se reutilizan en PCA/RFE/Modelado.")
    num_cols, cat_cols = split_features(df)
    target = st.selectbox("Selecciona la variable objetivo (y)", options=[None] + df.columns.tolist(), index=0)
    target = target or default_target_guess(df)

    if target is None:
        st.warning("Selecciona primero tu variable objetivo.")
    else:
        st.success(f"Objetivo: **{target}**")

    st.markdown("**Imputación**")
    num_impute_strategy = st.selectbox("Estrategia numérica", ["mean", "median", "most_frequent"], index=0)
    cat_impute_strategy = st.selectbox("Estrategia categórica", ["most_frequent", "constant"], index=0)
    cat_fill_value = st.text_input("Valor constante para categóricas (si aplica)", value="missing")

    st.markdown("**Escalado (numéricas)**")
    scaler_choice = st.selectbox("Escalador", ["StandardScaler", "RobustScaler", "None"], index=0)

    # Construimos el preprocesador
    if target and target in df.columns:
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()
    else:
        X = df.copy()
        y = None

    num_cols, cat_cols = split_features(X)

    numeric_transformers = []
    if num_impute_strategy:
        numeric_transformers.append(("imputer", SimpleImputer(strategy=num_impute_strategy)))
    if scaler_choice == "StandardScaler":
        numeric_transformers.append(("scaler", StandardScaler()))
    elif scaler_choice == "RobustScaler":
        numeric_transformers.append(("scaler", RobustScaler()))

    numeric_pipeline = Pipeline(steps=numeric_transformers) if numeric_transformers else "passthrough"
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat_impute_strategy, fill_value=cat_fill_value))
    ]) if cat_cols else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ],
        remainder="drop"
    )

    st.code(preprocessor.__repr__())

with tab4:
    st.subheader("PCA / MCA")

    st.write("Primero se aplica el **preprocesamiento** definido en la pestaña 'Transformaciones'.")
    n_components = st.slider("Componentes (PCA)", min_value=2, max_value=10, value=2, step=1)

    # PCA sobre variables numéricas solamente
    if num_cols:
        try:
            # Transform num only (más estable)
            num_pre = Pipeline([
                ("imputer", SimpleImputer(strategy=num_impute_strategy)),
                ("scaler", StandardScaler() if scaler_choice == "StandardScaler" else RobustScaler() if scaler_choice == "RobustScaler" else "passthrough")
            ])

            X_num = df[num_cols]
            X_num_tr = num_pre.fit_transform(X_num)
            pca = PCA(n_components=n_components, random_state=42)
            comps = pca.fit_transform(X_num_tr)

            explained = pca.explained_variance_ratio_
            fig1, ax1 = plt.subplots()
            ax1.plot(np.arange(1, len(explained)+1), np.cumsum(explained), marker="o")
            ax1.set_title("Varianza explicada acumulada (PCA)")
            ax1.set_xlabel("Componentes")
            ax1.set_ylabel("Proporción acumulada")
            st.pyplot(fig1)

            if n_components >= 2:
                fig2, ax2 = plt.subplots()
                ax2.scatter(comps[:,0], comps[:,1], alpha=0.6)
                ax2.set_xlabel("PC1")
                ax2.set_ylabel("PC2")
                ax2.set_title("Proyección 2D (PCA)")
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error en PCA: {e}")

    st.divider()
    st.markdown("### MCA (para categóricas)")
    if not HAS_PRINCE:
        st.info("Para usar MCA, instala `prince`: `pip install prince`")
    else:
        if len(cat_cols) >= 2:
            try:
                mca_components = st.slider("Componentes (MCA)", min_value=2, max_value=10, value=2, step=1, key="mca_comps")
                mca = prince.MCA(n_components=mca_components, random_state=42)
                X_cat = df[cat_cols].astype(str).fillna("missing")
                mca_fit = mca.fit(X_cat)
                mca_coords = mca_fit.transform(X_cat)

                fig3, ax3 = plt.subplots()
                ax3.scatter(mca_coords.iloc[:,0], mca_coords.iloc[:,1], alpha=0.5)
                ax3.set_xlabel("Dim 1")
                ax3.set_ylabel("Dim 2")
                ax3.set_title("Proyección 2D (MCA)")
                st.pyplot(fig3)
            except Exception as e:
                st.error(f"Error en MCA: {e}")
        else:
            st.info("Se requieren ≥ 2 columnas categóricas para MCA.")

with tab5:
    st.subheader("Selección de Variables (RFE)")
    if target is None or target not in df.columns:
        st.warning("Selecciona una variable objetivo en la pestaña Transformaciones.")
    else:
        X = df.drop(columns=[target])
        y = df[target]
        num_cols, cat_cols = split_features(X)

        # Simplificación: imputar y one-hot para categóricas
        X_proc = X.copy()
        for c in num_cols:
            X_proc[c] = X_proc[c].fillna(X_proc[c].median())
        for c in cat_cols:
            X_proc[c] = X_proc[c].fillna("missing")

        X_dum = pd.get_dummies(X_proc, columns=cat_cols, drop_first=True)

        n_feats = st.slider("Número de características a seleccionar", 1, min(30, X_dum.shape[1]), min(10, X_dum.shape[1]))
        estimator = LinearRegression()
        try:
            selector = RFE(estimator=estimator, n_features_to_select=n_feats)
            selector = selector.fit(X_dum, y)
            selected_mask = selector.support_
            selected_cols = X_dum.columns[selected_mask].tolist()

            st.write("**Top features seleccionadas:**")
            st.write(selected_cols)

        except Exception as e:
            st.error(f"Error en RFE: {e}")

with tab6:
    st.subheader("Modelado (Regresión)")
    if target is None or target not in df.columns:
        st.warning("Selecciona una variable objetivo en la pestaña Transformaciones.")
        st.stop()

    test_size = st.slider("Tamaño de test", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    model_name = st.selectbox("Modelo", list(model_dict(random_state).keys()))
    cv_folds = st.slider("Folds CV", 3, 10, 5, 1)
    do_learning_curve = st.checkbox("Mostrar learning curve", value=False)

    # Preprocesamiento + OneHot
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    num_cols, cat_cols = split_features(X)

    numeric_transformers = []
    if num_impute_strategy:
        numeric_transformers.append(("imputer", SimpleImputer(strategy=num_impute_strategy)))
    if scaler_choice == "StandardScaler":
        numeric_transformers.append(("scaler", StandardScaler()))
    elif scaler_choice == "RobustScaler":
        numeric_transformers.append(("scaler", RobustScaler()))

    numeric_pipeline = Pipeline(steps=numeric_transformers) if numeric_transformers else "passthrough"
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat_impute_strategy, fill_value=cat_fill_value))
    ]) if cat_cols else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ]
    )

    model = model_dict(random_state)[model_name]
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("onehot",  # OneHot via pandas.get_dummies después del ColumnTransformer imputado
            "passthrough"),
        ("model", model)
    ])

    # Truco: aplicar imputación con ColumnTransformer, luego one-hot con pandas
    # Para mantenerlo simple en Streamlit, transformamos manualmente:
    try:
        X_num = pd.DataFrame(preprocessor.named_transformers_["num"].fit_transform(X[num_cols]) if num_cols else np.empty((len(X),0)))
    except Exception:
        # Si todavía no ha sido ajustado
        if num_cols:
            X_num = pd.DataFrame(numeric_pipeline.fit_transform(X[num_cols]))
        else:
            X_num = pd.DataFrame()

    if cat_cols:
        try:
            X_cat = pd.DataFrame(preprocessor.named_transformers_["cat"].fit_transform(X[cat_cols]), columns=cat_cols)
        except Exception:
            X_cat = X[cat_cols].copy()
            X_cat = X_cat.fillna(cat_fill_value)
        X_proc = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
        X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=True)
    else:
        X_proc = X_num.copy()

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=random_state)

    # Re-ajustamos el modelo seleccionado
    model = model_dict(random_state)[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse_val = rmse(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"**R² (test):** {r2:.4f}")
    st.write(f"**RMSE (test):** {rmse_val:.4f}")
    st.write(f"**MAE (test):** {mae:.4f}")

    # CV
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_r2 = cross_val_score(model, X_proc, y, cv=kf, scoring="r2")
    cv_neg_mse = cross_val_score(model, X_proc, y, cv=kf, scoring="neg_mean_squared_error")
    st.write(f"**R² CV (media ± std):** {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    st.write(f"**RMSE CV (media):** {np.mean(np.sqrt(-cv_neg_mse)):.4f}")

    # Learning curve (opcional)
    if do_learning_curve:
        sizes, train_scores, val_scores = learning_curve(model, X_proc, y, cv=kf, scoring="r2",
                                                         train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=random_state)
        fig_lc, ax_lc = plt.subplots()
        ax_lc.plot(sizes, train_scores.mean(axis=1), marker="o", label="Train R²")
        ax_lc.plot(sizes, val_scores.mean(axis=1), marker="o", label="CV R²")
        ax_lc.set_xlabel("Tamaño de entrenamiento")
        ax_lc.set_ylabel("R²")
        ax_lc.set_title("Learning Curve")
        ax_lc.legend()
        st.pyplot(fig_lc)

    st.divider()
    st.markdown("### Post-Pruning: Árbol de Decisión (CCP)")
    st.caption("Explora el pruning por complejidad de coste si eliges DecisionTree.")
    if model_name == "DecisionTree":
        try:
            tree_full = DecisionTreeRegressor(random_state=random_state)
            tree_full.fit(X_train, y_train)
            path = tree_full.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities

            fig_ccp, ax_ccp = plt.subplots()
            ax_ccp.plot(ccp_alphas, impurities, marker="o")
            ax_ccp.set_xlabel("ccp_alpha")
            ax_ccp.set_ylabel("Impureza total hoja")
            ax_ccp.set_title("Ruta de pruning (CCP)")
            st.pyplot(fig_ccp)
        except Exception as e:
            st.info(f"No se pudo calcular CCP: {e}")

    st.divider()
    st.markdown("### Búsqueda Aleatoria (ejemplo rápido)")
    st.caption("Ejecuta una búsqueda aleatoria ligera para RandomForest o GradientBoosting.")
    do_search = st.checkbox("Ejecutar búsqueda aleatoria", value=False)
    if do_search and model_name in ["RandomForest", "GradientBoosting"]:
        try:
            if model_name == "RandomForest":
                base = RandomForestRegressor(random_state=random_state)
                param_dist = {
                    "n_estimators": [100, 200, 400],
                    "max_depth": [None, 6, 10, 14],
                    "max_features": ["sqrt", "log2", None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                }
            else:
                base = GradientBoostingRegressor(random_state=random_state)
                param_dist = {
                    "n_estimators": [100, 200, 400],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "subsample": [0.6, 0.8, 1.0],
                    "max_depth": [2, 3, 5],
                    "max_features": ["sqrt", "log2", None]
                }

            rs = RandomizedSearchCV(base, param_dist, n_iter=15, scoring="neg_root_mean_squared_error",
                                    cv=cv_folds, random_state=random_state, n_jobs=-1, verbose=0)
            rs.fit(X_train, y_train)
            st.write("**Mejores hyperparámetros:**", rs.best_params_)
            best = rs.best_estimator_
            y_pred_best = best.predict(X_test)
            st.write(f"**RMSE (test) con mejor modelo:** {rmse(y_test, y_pred_best):.4f}")
            st.write(f"**R² (test) con mejor modelo:** {r2_score(y_test, y_pred_best):.4f}")
        except Exception as e:
            st.error(f"Error en RandomizedSearchCV: {e}")

st.caption("© Adaptación automatizada desde el notebook. Ajusta parámetros según tu dataset.")

