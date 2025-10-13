import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Previsão de Renda", page_icon="💰", layout="wide")

st.title("💰 Previsão de Renda com Random Forest (Pipeline)")
st.markdown("App interativo com modelo completo de previsão de renda.")

# -----------------------------
# CARREGAR MODELO E DADOS
# -----------------------------
@st.cache_resource
def carregar_modelo():
    with open("modelo_pipeline.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def carregar_dados():
    return pd.read_csv("./input/previsao_de_renda.csv")

modelo = carregar_modelo()
df = carregar_dados()
df = df.drop(columns=["id_cliente", "data_ref", "Unnamed: 0"], errors="ignore")

# -----------------------------
# INTERFACE DE ENTRADA
# -----------------------------
st.sidebar.header("🧮 Parâmetros de Entrada")

inputs = {}
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# remover renda e renda_log se existirem
for alvo in ["renda", "renda_log"]:
    if alvo in num_cols: num_cols.remove(alvo)
    if alvo in cat_cols: cat_cols.remove(alvo)

int_cols = ["idade", "qt_pessoas_residencia", "qtd_filhos"]

# Campos numéricos
for col in num_cols:
    vmin, vmax = float(df[col].min()), float(df[col].max())
    vmed = float(df[col].median())

    if col in int_cols:
        valor = st.sidebar.number_input(
            col,
            min_value=int(vmin),
            max_value=int(vmax),
            value=int(vmed),
            step=1
        )
        inputs[col] = int(valor)
    else:
        valor = st.sidebar.number_input(
            col,
            min_value=vmin,
            max_value=vmax,
            value=vmed,
            step=(vmax - vmin) / 100
        )
        inputs[col] = float(valor)

# Campos categóricos
for col in cat_cols:
    valores = df[col].dropna().unique().tolist()
    inputs[col] = st.sidebar.selectbox(col, valores)

# Cria dataframe de entrada
entrada = pd.DataFrame([inputs])
entrada[int_cols] = entrada[int_cols].astype(int)

# -----------------------------
# PREVISÃO
# -----------------------------
st.subheader("📈 Previsão de Renda")

if st.button("Calcular Previsão 💵"):
    try:
        renda_prevista = modelo.predict(entrada)[0]
        st.success(f"💰 Renda estimada: **R$ {renda_prevista:,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Erro ao calcular a previsão: {e}")

# -----------------------------
# GRÁFICOS DE EXPLORAÇÃO
# -----------------------------
st.markdown("---")
st.header("📊 Exploração dos Dados")

# Renda por Tempo de Emprego com cores distintas
if "tempo_emprego" in df.columns:
    color_var = "sexo" if "sexo" in df.columns else None
    fig_te = px.scatter(
        df, x="tempo_emprego", y="renda", color=color_var,
        title="Renda por Tempo de Emprego",
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig_te, use_container_width=True)

# Renda por Qt Pessoas na Residência
if "qt_pessoas_residencia" in df.columns:
    fig_qtp = px.box(
        df, x="qt_pessoas_residencia", y="renda", title="Renda por Qt Pessoas na Residência",
        color_discrete_sequence=["#EF553B"]
    )
    st.plotly_chart(fig_qtp, use_container_width=True)

# Renda por Qtd Filhos
if "qtd_filhos" in df.columns:
    fig_qtf = px.box(
        df, x="qtd_filhos", y="renda", title="Renda por Quantidade de Filhos",
        color_discrete_sequence=["#00CC96"]
    )
    st.plotly_chart(fig_qtf, use_container_width=True)

# Renda por Idade
if "idade" in df.columns:
    fig_idade = px.scatter(
        df, x="idade", y="renda", color=color_var,
        title="Renda por Idade",
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig_idade, use_container_width=True)

# Renda por Posse de Veículo
if "possui_veiculo" in df.columns:
    fig_veiculo = px.box(
        df, x="possui_veiculo", y="renda", title="Renda por Posse de Veículo",
        color_discrete_sequence=["#FFA15A"]
    )
    st.plotly_chart(fig_veiculo, use_container_width=True)

# -----------------------------
# MAPA DE CORRELAÇÃO NUMÉRICO
# -----------------------------
st.markdown("---")
st.subheader("🧩 Correlação entre Variáveis Numéricas")

num_cols_corr = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if len(num_cols_corr) > 1:
    corr = df[num_cols_corr].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Correlação entre Variáveis Numéricas",
        labels=dict(x="Variável", y="Variável", color="Correlação")
    )
    st.plotly_chart(fig_corr, use_container_width=True)


