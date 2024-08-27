import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
import streamlit as st
import time

# Streamlit Title
st.title('Müşteri Satın Alma Analizi')
st.write('Bu projede, müşterinin satın alma kararlarını inceleyeceğimiz bir sınıflandırma problemini ele alacağız.')
df = pd.read_csv("customer_purchase_data.csv")

# Display the dataframe if checkbox is selected
if st.checkbox('Show dataframe'):
    st.write(df)

# Display descriptive statistics if checkbox is selected
desc = df.describe().T
if st.checkbox('Show describe statistic'):
    st.write(desc)

# Dropdown menu for visualizing data
st.sidebar.title("Visualization Options")
visual_type = st.sidebar.selectbox("Select visualization type", ["Numerical", "Categorical"])

if visual_type == "Numerical":
    st.sidebar.subheader("Numerical Features")
    numerical_feature = st.sidebar.selectbox("Select numerical feature", ["Age", "AnnualIncome", "NumberOfPurchases", "TimeSpentOnWebsite", "DiscountsAvailed"])

    if numerical_feature:
        st.subheader(f"{numerical_feature} Distribution")
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(df[numerical_feature], bins=30, kde=True)
        plt.title(f'{numerical_feature} Distribution')
        st.pyplot(fig)

elif visual_type == "Categorical":
    st.sidebar.subheader("Categorical Features")
    categorical_feature = st.sidebar.selectbox("Select categorical feature", ["Gender", "ProductCategory", "LoyaltyProgram", "PurchaseStatus"])

    if categorical_feature:
        st.subheader(f"{categorical_feature} Distribution")
        fig = plt.figure(figsize=(10, 6))
        sns.countplot(x=categorical_feature, data=df)
        plt.title(f'{categorical_feature} Distribution')
        st.pyplot(fig)

# Encode categorical variables for model training
df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns.tolist(), drop_first=True)

# Prepare data for classification
X = df_encoded.drop('PurchaseStatus', axis=1)
y = df_encoded['PurchaseStatus']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, model_name):
    start_time = time.time()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    execution_time = time.time() - start_time
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Execution Time': execution_time
    }

# Initialize metrics dictionary
metrics = {}

# Evaluate all models
metrics['Random Forest'] = evaluate_model(RandomForestClassifier(random_state=42), 'Random Forest')
metrics['Support Vector Machine'] = evaluate_model(SVC(random_state=42), 'Support Vector Machine')
metrics['LightGBM'] = evaluate_model(lgb.LGBMClassifier(random_state=42), 'LightGBM')
metrics['XGBoost'] = evaluate_model(xgb.XGBClassifier(random_state=42), 'XGBoost')

# Convert metrics dictionary to DataFrame for display
metrics_df = pd.DataFrame(metrics).T
metrics_df.index.name = 'Model'

# Streamlit Title for Classification Model Evaluation
st.title('Classification Model Evaluation')

# Display metrics table
st.write('### Model Comparison Metrics')
st.table(metrics_df)

# Display confusion matrix for selected model
selected_model = st.selectbox('Select a model to view its confusion matrix', metrics.keys())
if selected_model:
    model_map = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
    
    selected_model_instance = model_map[selected_model]
    selected_model_instance.fit(x_train, y_train)
    y_pred = selected_model_instance.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    
    st.subheader(f"{selected_model} Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{selected_model} Confusion Matrix')
    st.pyplot(fig)

# Summary text
summary_text = """Bu çalışmada, müşteri satın alma davranışını tahmin etmek için dört sınıflandırma modeli değerlendirildi:
- Random Forest
- Support Vector Machine (SVM)
- LightGBM
- XGBoost

LightGBM ve XGBoost yüksek doğruluk puanlarıyla öne çıktı. Bu modeller, güvenilirlikleri ve iyi performansları nedeniyle müşteri satın alma davranışını tahmin etmek için önerilir.
"""
st.write('### Study Summary')
st.markdown(summary_text)
