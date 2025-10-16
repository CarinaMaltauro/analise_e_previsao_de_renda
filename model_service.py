import pickle
import pandas as pd
import plotly.express as px

# =====================================================
# CARREGAR MODELO E DADOS
# =====================================================

def carregar_modelo(caminho="modelo_pipeline.pkl"):
    """Carrega o modelo treinado (pipeline completo)."""
    with open(caminho, "rb") as f:
        modelo = pickle.load(f)
    return modelo


def carregar_dados(caminho="./input/previsao_de_renda.csv"):
    """Carrega e limpa o dataset original."""
    df = pd.read_csv(caminho)
    df = df.drop(columns=["id_cliente", "data_ref", "Unnamed: 0"], errors="ignore")
    return df


# =====================================================
# PREVISÃO
# =====================================================

def prever_renda(modelo, entrada):
    """
    Retorna o valor previsto com checagem de colunas.
    """
    # Garante que entrada é DataFrame
    if not isinstance(entrada, pd.DataFrame):
        entrada = pd.DataFrame([entrada])

    # Ajusta as colunas conforme o modelo treinado
    colunas_treinadas = modelo.feature_names_in_
    entrada = entrada.reindex(columns=colunas_treinadas)

    # Faz a previsão
    renda_prevista = modelo.predict(entrada)[0]
    return renda_prevista


# =====================================================
# GRÁFICOS DE EXPLORAÇÃO
# =====================================================

def plot_renda_por_tempo_emprego(df):
    """Renda por tempo de emprego."""
    color_var = "sexo" if "sexo" in df.columns else None
    return px.scatter(
        df,
        x="tempo_emprego",
        y="renda",
        color=color_var,
        title="Renda por Tempo de Emprego",
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Safe
    )


def plot_renda_por_qt_pessoas(df):
    """Renda por número de pessoas na residência."""
    return px.box(
        df,
        x="qt_pessoas_residencia",
        y="renda",
        title="Renda por Qt Pessoas na Residência",
        color_discrete_sequence=["#EF553B"]
    )


def plot_renda_por_filhos(df):
    """Renda por quantidade de filhos."""
    return px.box(
        df,
        x="qtd_filhos",
        y="renda",
        title="Renda por Quantidade de Filhos",
        color_discrete_sequence=["#00CC96"]
    )


def plot_renda_por_idade(df):
    """Renda por idade."""
    color_var = "sexo" if "sexo" in df.columns else None
    return px.scatter(
        df,
        x="idade",
        y="renda",
        color=color_var,
        title="Renda por Idade",
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Safe
    )


def plot_renda_por_veiculo(df):
    """Renda por posse de veículo."""
    col_veiculo = "posse_de_veiculo" if "posse_de_veiculo" in df.columns else "possui_veiculo"
    return px.box(
        df,
        x=col_veiculo,
        y="renda",
        title="Renda por Posse de Veículo",
        color_discrete_sequence=["#FFA15A"]
    )


def plot_correlacao(df):
    """Mapa de correlação entre variáveis numéricas."""
    corr = df.select_dtypes(include=["int64", "float64"]).corr()
    return px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Correlação entre Variáveis Numéricas",
        labels=dict(x="Variável", y="Variável", color="Correlação")
    )
