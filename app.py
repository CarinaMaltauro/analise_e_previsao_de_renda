import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency

# Importa as funções do módulo externo
from model_service import (
    carregar_modelo,
    carregar_dados,
    prever_renda,
    plot_renda_por_tempo_emprego,
    plot_renda_por_qt_pessoas,
    plot_renda_por_filhos,
    plot_renda_por_idade,
    plot_renda_por_veiculo,
    plot_correlacao
)

# =====================================================
# CONFIGURAÇÃO INICIAL
# =====================================================
st.set_page_config(page_title="Previsão de Renda", page_icon="💰", layout="wide")

st.title("💰 Previsão de Renda com Random Forest (Pipeline)")
st.markdown("App interativo com modelo completo de previsão de renda.")

# =====================================================
# CARREGAR MODELO E DADOS
# =====================================================
@st.cache_resource
def get_modelo():
    return carregar_modelo("modelo_pipeline.pkl")

@st.cache_data
def get_dados():
    return carregar_dados("./input/previsao_de_renda.csv")

modelo = get_modelo()
df = get_dados()

# =====================================================
# INTERFACE DE ENTRADA
# =====================================================
st.sidebar.header("🧮 Parâmetros de Entrada")

inputs = {}
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Remove colunas-alvo
for alvo in ["renda", "renda_log"]:
    if alvo in num_cols: num_cols.remove(alvo)
    if alvo in cat_cols: cat_cols.remove(alvo)

int_cols = ["idade", "qt_pessoas_residencia", "qtd_filhos"]

# Campos numéricos
# Para cada variável numérica, o app cria um campo numérico interativo com valores mínimo, máximo e mediano da base.
# Se for uma variável inteira (como idade ou número de filhos), ele usa step=1.

for col in num_cols:
    vmin, vmax = float(df[col].min()), float(df[col].max())
    vmed = float(df[col].median())

    if col in int_cols:
        valor = st.sidebar.number_input(col, min_value=int(vmin), max_value=int(vmax), value=int(vmed), step=1)
        inputs[col] = int(valor)
    else:
        valor = st.sidebar.number_input(col, min_value=vmin, max_value=vmax, value=vmed, step=(vmax - vmin) / 100)
        inputs[col] = float(valor)

# Campos categóricos
# Para as variáveis categóricas, o app cria menus suspensos (selectbox) com todas as categorias únicas encontradas no DataFrame.
for col in cat_cols:
    valores = df[col].dropna().unique().tolist()
    inputs[col] = st.sidebar.selectbox(col, valores)

entrada = pd.DataFrame([inputs])
entrada[int_cols] = entrada[int_cols].astype(int)

# =====================================================
# PREVISÃO
# =====================================================
st.subheader("📈 Previsão de Renda")

if st.button("Calcular Previsão 💵"):
    try:
        renda_prevista = prever_renda(modelo, entrada)
        st.success(f"💰 Renda estimada: **R$ {renda_prevista:,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Erro ao calcular a previsão: {e}")

# =====================================================
# GRÁFICOS DE EXPLORAÇÃO
# =====================================================
st.markdown("---")
st.header("📊 Exploração dos Dados")

if "tempo_emprego" in df.columns:
    st.plotly_chart(plot_renda_por_tempo_emprego(df), use_container_width=True)

if "qt_pessoas_residencia" in df.columns:
    st.plotly_chart(plot_renda_por_qt_pessoas(df), use_container_width=True)

if "qtd_filhos" in df.columns:
    st.plotly_chart(plot_renda_por_filhos(df), use_container_width=True)

if "idade" in df.columns:
    st.plotly_chart(plot_renda_por_idade(df), use_container_width=True)

if "posse_de_veiculo" in df.columns or "possui_veiculo" in df.columns:
    st.plotly_chart(plot_renda_por_veiculo(df), use_container_width=True)

# =====================================================
# MAPA DE CORRELAÇÃO NUMÉRICO
# =====================================================
st.markdown("---")
st.subheader("🧩 Correlação entre Variáveis Numéricas")

num_cols_corr = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if len(num_cols_corr) > 1:
    st.plotly_chart(plot_correlacao(df), use_container_width=True)
