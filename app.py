import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Global Development Clustering",
    layout="wide"
)

st.title("🌍 Global Development Clustering using DBSCAN")

st.markdown("""
This application clusters countries based on **development indicators**
using **DBSCAN**, selected due to the highest **Silhouette Score**.
""")

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# IMPORTANT FIX: Use ONLY numeric columns
# -------------------------------------------------
numeric_df = df.select_dtypes(include='number')

st.subheader("Numeric Features Used for Clustering")
st.write(numeric_df.columns.tolist())

# -------------------------------------------------
# Scaling
# -------------------------------------------------
scaler = RobustScaler()
scaled_data = scaler.fit_transform(numeric_df)

# -------------------------------------------------
# Sidebar – DBSCAN Parameters
# -------------------------------------------------
st.sidebar.header("DBSCAN Parameters")

eps = st.sidebar.slider(
    "EPS (Neighborhood Radius)",
    min_value=0.1,
    max_value=3.0,
    value=1.2,
    step=0.1
)

min_samples = st.sidebar.slider(
    "Minimum Samples",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)

# -------------------------------------------------
# DBSCAN Model
# -------------------------------------------------
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(scaled_data)

numeric_df["Cluster"] = clusters

# -------------------------------------------------
# Cluster Distribution
# -------------------------------------------------
st.subheader("Cluster Distribution")
st.write(numeric_df["Cluster"].value_counts())

st.markdown("""
**Note:**  
Cluster `-1` represents **noise / outliers**, which DBSCAN automatically detects.
""")

# -------------------------------------------------
# PCA Visualization
# -------------------------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(
    pca_data,
    columns=["PC1", "PC2"]
)
pca_df["Cluster"] = clusters

st.subheader("Cluster Visualization (PCA)")

fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    ax=ax
)
ax.set_title("DBSCAN Clusters (PCA Projection)")
st.pyplot(fig)

# -------------------------------------------------
# Download Results
# -------------------------------------------------
st.subheader("Download Clustered Dataset")

output_df = df.copy()
output_df["Cluster"] = clusters

csv = output_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download CSV",
    data=csv,
    file_name="clustered_countries.csv",
    mime="text/csv"
)
