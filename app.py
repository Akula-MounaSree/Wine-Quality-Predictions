import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
df=pd.read_csv(r"WineQT.csv")
st.title("Wine Quality Analysis")
st.image("https://redandwhiteshops.com/wp-content/uploads/2025/04/Sweet-Red-Wine-Types-A-Complete-Guide-for-Every-Palate.jpg")
st.markdown("This is a wine quality analysis app built using Streamlit. It allows users to explore the dataset, visualize relationships between features, and gain insights into the factors that influence wine quality.")
st.header("Dataset Overview")
st.write(df.head(10))
st.write('Dataset Shape:', df.shape)
st.write('Dataset Columns:', df.columns)
st.write('Dataset Description:', df.describe())
st.write('Missing Values:', df.isnull().sum())
#st.header("Wine Quality Distribution")
#fig, ax = plt.subplots()
#sns.histplot(df['quality'], bins=10, kde=True, ax=ax)
#st.pyplot(fig)

df['fixed acidity'] = pd.to_numeric(df['fixed acidity'], errors='coerce')
df=df.dropna(subset=['quality','Id'])
df.columns = df.columns.str.strip()
X=df.drop(['quality', 'Id'], axis=1).apply(pd.to_numeric, errors='coerce').fillna(df.mean())
y=df['quality']
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#sliders
fixed_acidity = st.slider("Fixed Acidity", float(df["fixed acidity"].min()), float(df["fixed acidity"].max()), float(df["fixed acidity"].mean()))
Volatile_acidity = st.slider("Volatile Acidity", float(df["volatile acidity"].min()), float(df["volatile acidity"].max()), float(df["volatile acidity"].mean()))
citric_acid = st.slider("Citric Acid", float(df["citric acid"].min()), float(df["citric acid"].max()), float(df["citric acid"].mean()))
residual_sugar = st.slider("Residual Sugar", float(df["residual sugar"].min()), float(df["residual sugar"].max()), float(df["residual sugar"].mean()))
chlorides = st.slider("Chlorides", float(df["chlorides"].min()), float(df["chlorides"].max()), float(df["chlorides"].mean()))
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", float(df["free sulfur dioxide"].min()), float(df["free sulfur dioxide"].max()), float(df["free sulfur dioxide"].mean()))
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", float(df["total sulfur dioxide"].min()), float(df["total sulfur dioxide"].max()), float(df["total sulfur dioxide"].mean()))
density = st.slider("Density", float(df["density"].min()), float(df["density"]. max()), float(df["density"].mean()))
pH = st.slider("pH", float(df["pH"].min()), float(df["pH"].max()), float(df["pH"].mean()))
sulphates = st.slider("Sulphates",float(df["sulphates"].min()), float(df["sulphates"].max()),float(df["sulphates"].mean()))
alcohol = st.slider("Alcohol", float(df["alcohol"].min()), float(df["alcohol"].max()), float(df["alcohol"].mean()))
input_data = np.array([[fixed_acidity, Volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
input_data_scaled = scaler.transform(input_data)

#collects input data and makes prediction
# Collect input data
input_data=np.array([[fixed_acidity, Volatile_acidity, citric_acid, residual_sugar,chlorides, free_sulfur_dioxide, total_sulfur_dioxide,density, pH, sulphates, alcohol]])

# Scale input
input_data_scaled = scaler.transform(input_data)

# Predict only when button is clicked
if st.button("Predict Quality"):
    prediction = model.predict(input_data_scaled)
    st.subheader("Predicted Wine Quality")
    st.write(f"⭐ {prediction[0]} ⭐")
#features=pd.DataFrame({"fixed acidity":[fixed_acidity], "volatile acidity":[Volatile_acidity],"citric acid":[citric_acid],
#"residual sugar":[residual_sugar],"chlorides":[chlorides],"free sulfur dioxide":[free_sulfur_dioxide],"total sulfur dioxide":[total_sulfur_dioxide],"density":[density],
#"pH":[pH],"sulphates":[sulphates],"alcohol":[alcohol]})
#if st.button("Predict Quality"):
 #   prediction = model.predict(input_data_scaled)
  #  st.write(f"Predicted Wine Quality: {prediction[0]}")    
#scaler 
#scaler_features=scaler.transform(features)
#prediction
#prediction = model.predict(scaler_features)
#st.write(f"Predicted Wine Quality: {prediction[0]}")
#st.subheader("Predicted Wine Quality")
#st.write(f"⭐ {prediction[0]} ⭐")
