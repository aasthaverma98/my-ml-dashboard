import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# --- Page Config & Aesthetics ---
st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide", page_icon="🚀")
st.title("🚀 End-to-End Machine Learning Pipeline")
st.markdown("Navigate through the tabs below to process your data step-by-step.")

# --- Session State Management ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'clean_df' not in st.session_state:
    st.session_state.clean_df = None

# --- Horizontal Steps ---
tabs = st.tabs([
    "1. Problem Type", "2. Data & PCA", "3. EDA", "4. Engineering & Cleaning",
    "5. Feature Selection", "6. Data Split", "7. Model Selection", 
    "8. Train & Validate", "9. Metrics & Tuning"
])

# --- Tab 1: Problem Type ---
with tabs[0]:
    st.header("1. Define the Problem")
    prob_type = st.radio("Select the type of problem to solve:", ("Classification", "Regression"))
    st.session_state.prob_type = prob_type
    st.success(f"Pipeline set for **{prob_type}** tasks.")

# --- Tab 2: Data Input & PCA ---
with tabs[1]:
    st.header("2. Data Input & Shape Analysis")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        if st.session_state.df is None or st.button("Reload Data"):
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.clean_df = st.session_state.df.copy()
            
        df = st.session_state.clean_df
        st.write("### Data Preview", df.head())
        
        target_col = st.selectbox("Select the Target Feature", df.columns, index=len(df.columns)-1)
        st.session_state.target = target_col
        
        st.markdown("---")
        st.subheader("PCA Visualization")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
            
        selected_pca_features = st.multiselect("Select features for PCA", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
        
        if len(selected_pca_features) >= 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df[selected_pca_features].dropna())
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            # Align target for coloring
            temp_target = df.dropna(subset=selected_pca_features)[target_col]
            pca_df['Target'] = temp_target.values
            
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target', title="2D PCA Plot", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least 2 numeric features for PCA.")

# --- Tab 3: Exploratory Data Analysis (EDA) ---
with tabs[2]:
    st.header("3. Exploratory Data Analysis")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Description:**")
            st.write(df.describe())
        with col2:
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())
            
        st.subheader("Correlation Matrix")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload data in Tab 2.")

# --- Tab 4: Data Engineering & Cleaning ---
with tabs[3]:
    st.header("4. Data Engineering & Cleaning")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        
        st.subheader("Missing Value Imputation")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            impute_col = st.selectbox("Select column to impute", missing_cols)
            impute_strategy = st.selectbox("Strategy", ["mean", "median", "most_frequent"])
            if st.button("Apply Imputation"):
                imputer = SimpleImputer(strategy=impute_strategy)
                df[[impute_col]] = imputer.fit_transform(df[[impute_col]])
                st.session_state.clean_df = df
                st.success(f"Imputed {impute_col} using {impute_strategy}")
                st.rerun()
        else:
            st.success("No missing values found!")

        st.markdown("---")
        st.subheader("Outlier Detection")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        outlier_method = st.selectbox("Select Outlier Method", ["None", "IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        
        if outlier_method != "None" and len(num_cols) > 0:
            outlier_indices = []
            
            if outlier_method == "IQR":
                col = st.selectbox("Select column for IQR", num_cols)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_indices = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].index
                
            elif outlier_method == "Isolation Forest":
                iso = IsolationForest(contamination=0.05, random_state=42)
                preds = iso.fit_predict(df[num_cols].fillna(0))
                outlier_indices = df[preds == -1].index
                
            elif outlier_method in ["DBSCAN", "OPTICS"]:
                model = DBSCAN(eps=3, min_samples=2) if outlier_method == "DBSCAN" else OPTICS(min_samples=2)
                preds = model.fit_predict(df[num_cols].fillna(0))
                outlier_indices = df[preds == -1].index
                
            st.write(f"Detected **{len(outlier_indices)}** outliers using {outlier_method}.")
            if len(outlier_indices) > 0 and st.button("Remove Outliers"):
                st.session_state.clean_df = df.drop(index=outlier_indices)
                st.success("Outliers removed!")
                st.rerun()
    else:
        st.info("Please upload data in Tab 2.")

# --- Tab 5: Feature Selection ---
with tabs[4]:
    st.header("5. Feature Selection")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        target = st.session_state.target
        
        st.write("Filter features based on Variance or Information Gain.")
        num_df = df.select_dtypes(include=np.number).dropna()
        X = num_df.drop(columns=[target], errors='ignore')
        
        if target in num_df.columns:
            y = num_df[target]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Variance Threshold")
                var_thresh = st.slider("Threshold", 0.0, 1.0, 0.0)
                selector = VarianceThreshold(threshold=var_thresh)
                selector.fit(X)
                var_features = X.columns[selector.get_support()]
                st.write(f"Features passing variance threshold: {len(var_features)}")
                
            with col2:
                st.subheader("Information Gain")
                if st.button("Calculate IG"):
                    if st.session_state.prob_type == "Classification":
                        ig = mutual_info_classif(X, y)
                    else:
                        ig = mutual_info_regression(X, y)
                    ig_df = pd.DataFrame({'Feature': X.columns, 'Gain': ig}).sort_values(by='Gain', ascending=False)
                    fig = px.bar(ig_df, x='Gain', y='Feature', orientation='h', title="Information Gain with Target")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Target column must be numeric for these automated selection algorithms.")
    else:
        st.info("Please upload data.")

# --- Tab 6: Data Split ---
with tabs[5]:
    st.header("6. Data Split")
    if st.session_state.clean_df is not None:
        test_size = st.slider("Test Size Proportion", 0.1, 0.5, 0.2, 0.05)
        st.session_state.test_size = test_size
        st.write(f"Data will be split: **{int((1-test_size)*100)}% Train / {int(test_size*100)}% Test**")
    else:
        st.info("Please upload data.")

# --- Tab 7: Model Selection ---
with tabs[6]:
    st.header("7. Model Selection")
    prob_type = st.session_state.get('prob_type', 'Classification')
    
    if prob_type == "Classification":
        models = ["Logistic Regression", "SVM", "Random Forest Classifier", "K-Means (Clustering)"]
    else:
        models = ["Linear Regression", "SVR", "Random Forest Regressor"]
        
    selected_model = st.selectbox("Choose a Model:", models)
    st.session_state.model_name = selected_model
    
    if selected_model in ["SVM", "SVR"]:
        kernel = st.selectbox("SVM Kernel", ["linear", "poly", "rbf", "sigmoid"])
        st.session_state.kernel = kernel

# --- Tab 8 & 9: Training, Validation, Metrics & Tuning ---
# To pass data between these logically, we group the execution here
if st.session_state.clean_df is not None:
    df = st.session_state.clean_df.select_dtypes(include=np.number).dropna() # Using clean numeric data for simplicity
    target = st.session_state.target
    if target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state.get('test_size', 0.2), random_state=42)

with tabs[7]:
    st.header("8. Model Training & K-Fold Validation")
    k_folds = st.number_input("Select K for K-Fold Cross Validation", min_value=2, max_value=20, value=5)
    
    if st.button("Initialize & Evaluate Base Model"):
        model_name = st.session_state.get('model_name', None)
        prob_type = st.session_state.get('prob_type', 'Classification')
        
        # Instantiate Model
        model = None
        if model_name == "Linear Regression": model = LinearRegression()
        elif model_name == "Logistic Regression": model = LogisticRegression()
        elif model_name == "Random Forest Classifier": model = RandomForestClassifier(random_state=42)
        elif model_name == "Random Forest Regressor": model = RandomForestRegressor(random_state=42)
        elif model_name == "SVM": model = SVC(kernel=st.session_state.get('kernel', 'rbf'))
        elif model_name == "SVR": model = SVR(kernel=st.session_state.get('kernel', 'rbf'))
        elif model_name == "K-Means (Clustering)": model = KMeans(n_clusters=len(np.unique(y)))
        
        if model:
            st.session_state.base_model = model
            # K-Fold
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            scoring = 'accuracy' if prob_type == 'Classification' and model_name != "K-Means (Clustering)" else 'neg_mean_squared_error'
            if model_name != "K-Means (Clustering)":
                cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring=scoring)
                st.write(f"**Cross Validation Scores ({k_folds} Folds):**", cv_results)
                st.write(f"**Average CV Score:** {cv_results.mean():.4f}")
            else:
                st.warning("K-Fold CV standard scoring skipped for K-Means as it is unsupervised.")
            
            # Train and fit for metrics
            model.fit(X_train, y_train)
            st.session_state.fitted_model = model
            st.success(f"{model_name} trained successfully!")

with tabs[8]:
    st.header("9. Performance Metrics & Tuning")
    if 'fitted_model' in st.session_state:
        model = st.session_state.fitted_model
        prob_type = st.session_state.prob_type
        
        if st.session_state.model_name != "K-Means (Clustering)":
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            st.subheader("Base Model Metrics")
            col1, col2 = st.columns(2)
            if prob_type == "Classification":
                train_score = accuracy_score(y_train, train_preds)
                test_score = accuracy_score(y_test, test_preds)
                col1.metric("Training Accuracy", f"{train_score:.4f}")
                col2.metric("Testing Accuracy", f"{test_score:.4f}")
            else:
                train_score = r2_score(y_train, train_preds)
                test_score = r2_score(y_test, test_preds)
                col1.metric("Training R2 Score", f"{train_score:.4f}")
                col2.metric("Testing R2 Score", f"{test_score:.4f}")
                
            # Overfitting Check
            diff = train_score - test_score
            if diff > 0.10:
                st.error(f"⚠️ Potential Overfitting detected! (Train score is significantly higher than Test score by {diff:.4f})")
            elif diff < -0.05:
                st.warning("⚠️ Potential Underfitting detected!")
            else:
                st.success("✅ Model generalized well. No severe overfitting/underfitting detected.")
                
            st.markdown("---")
            st.subheader("Hyperparameter Tuning")
            tune_method = st.radio("Select Tuning Method", ["GridSearchCV", "RandomizedSearchCV"])
            
            if st.button("Run Tuning (Random Forest & SVM Example)"):
                with st.spinner('Tuning in progress...'):
                    param_grid = {}
                    if "Random Forest" in st.session_state.model_name:
                        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
                    elif "SV" in st.session_state.model_name:
                        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
                    
                    if not param_grid:
                        st.info("Tuning logic for this specific model is not defined in this demo.")
                    else:
                        search = GridSearchCV(st.session_state.base_model, param_grid, cv=3) if tune_method == "GridSearchCV" else RandomizedSearchCV(st.session_state.base_model, param_grid, cv=3, n_iter=5)
                        search.fit(X_train, y_train)
                        
                        st.success(f"Best Parameters: {search.best_params_}")
                        tuned_preds = search.best_estimator_.predict(X_test)
                        
                        if prob_type == "Classification":
                            st.metric("Tuned Testing Accuracy", f"{accuracy_score(y_test, tuned_preds):.4f}", delta=f"{(accuracy_score(y_test, tuned_preds) - test_score):.4f}")
                        else:
                            st.metric("Tuned Testing R2", f"{r2_score(y_test, tuned_preds):.4f}", delta=f"{(r2_score(y_test, tuned_preds) - test_score):.4f}")
        else:
            st.info("Metrics and standard tuning are distinct for unsupervised clustering (K-Means).")
    else:
        st.info("Please train a model in Tab 8 first.")