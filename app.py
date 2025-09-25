import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Configuração da página
st.set_page_config(
    page_title="Análise de Dados de Termo-higrômetro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do aplicativo
st.title("Projeto Integrador IV: Monitoramento de Estoque com IoT e Machine Learning")

st.markdown("""
Este aplicativo demonstra como utilizar dados de um termo-higrômetro para análise de estoque,
aplicando técnicas de machine learning para auxiliar na tomada de decisões.
""")

# --- Seção de Carregamento e Tratamento dos Dados ---
st.header("1. Carregamento e Tratamento dos Dados")

# O nome do arquivo com os dados reais (deve estar no mesmo diretório)
file_path = 'dados_pi4_13a19set.txt'

@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        # Ler o arquivo de texto, ignorando as duas primeiras linhas de metadados
        df = pd.read_csv(file_path, sep='\t', skiprows=2, encoding='utf-16')
        df.columns = ['data', 'temperatura', 'umidade', 'ponto_orvalho', 'vpd']
        df['data'] = pd.to_datetime(df['data'], format='%m/%d/%Y %H:%M')
        df['temperatura'] = df['temperatura'].str.replace(',', '.').astype(float)
        df['umidade'] = df['umidade'].str.replace(',', '.').astype(float)
        return df

    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{file_path}' não foi encontrado. Certifique-se de que ele está no mesmo diretório do script.")
        return None

df = load_and_preprocess_data(file_path)

if df is not None:
    st.write("Dados carregados com sucesso. Primeiras 5 linhas:")
    st.dataframe(df.head())
    
    st.write("Informações do DataFrame:")
    st.write(df.info())

# --- Seção de Visualização Inicial ---
st.header("2. Visualização Inicial dos Dados")
st.markdown("Veja como a temperatura e a umidade variaram ao longo do tempo.")
if st.button("Gerar Gráfico de Linhas"):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["data"], df["temperatura"], label="Temperatura (°C)")
    ax.plot(df["data"], df["umidade"], label="Umidade (%)")
    ax.legend()
    ax.set_title("Dados de temperatura e umidade no estoque")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor")
    # Usa st.pyplot para renderizar o gráfico, permitindo que ele se adapte ao layout do Streamlit
    st.pyplot(fig)

# --- Seção de Previsão de Temperatura (Análise Preditiva) ---
st.header("3. Previsão de Temperatura")
st.markdown("Um modelo de regressão linear prevê a temperatura com base no tempo.")

df["hora"] = np.arange(len(df))

X = df[["hora"]]
y = df["temperatura"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
ax_pred.plot(df["data"], y, label="Temperatura Real")
ax_pred.plot(df["data"].iloc[len(X_train):], preds, label="Previsão", color="red")
ax_pred.legend()
ax_pred.set_title("Previsão de Temperatura")
ax_pred.set_xlabel("Data")
ax_pred.set_ylabel("Valor")
st.pyplot(fig_pred)

# --- Seção de Classificação de Condições de Armazenamento ---
st.header("4. Classificação das Condições de Armazenamento")
st.markdown("""
- **Adequado**: Temperatura $< 25$ ºC e Umidade $< 70$ %
- **Inadequado**: Temperatura $> 25$ ºC ou Umidade $\\ge 70$ %
""")

def classificar_condicao(temp, umid):
    if temp < 25 and umid < 70:
        return "Adequado"
    else:
        return "Inadequado"

df['condicao_armazenamento'] = df.apply(lambda row: classificar_condicao(row['temperatura'], row['umidade']), axis=1)
st.write("Tabela com a classificação de armazenamento:")
st.dataframe(df[['data', 'temperatura', 'umidade', 'condicao_armazenamento']].head())

# --- Seção de Detecção de Anomalias ---
st.header("5. Detecção de Anomalias")

# Usando colunas para o layout responsivo, como você já fez.
col1, col2 = st.columns(2)

with col1:
    st.subheader("Anomalias de Temperatura")
    fig_temp_boxplot, ax_temp_boxplot = plt.subplots(figsize=(8, 6))
    ax_temp_boxplot.boxplot(df['temperatura'])
    ax_temp_boxplot.set_title('Boxplot da Temperatura')
    ax_temp_boxplot.set_ylabel('Temperatura (°C)')
    ax_temp_boxplot.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_temp_boxplot)

with col2:
    st.subheader("Anomalias de Umidade")
    fig_umid_boxplot, ax_umid_boxplot = plt.subplots(figsize=(8, 6))
    ax_umid_boxplot.boxplot(df['umidade'])
    ax_umid_boxplot.set_title('Boxplot da Umidade')
    ax_umid_boxplot.set_ylabel('Umidade (%)')
    ax_umid_boxplot.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_umid_boxplot)

st.header("6. Conclusão da Análise")
st.markdown("""
-   O uso de um **termo-higrômetro** fornece dados essenciais para o monitoramento de estoque.
-   Técnicas de **machine learning**, como regressão linear, permitem prever condições futuras.
-   A **classificação de dados** ajuda a identificar rapidamente períodos de risco para o armazenamento de produtos.
-   A análise de **anomalias** por meio de boxplots pode revelar falhas em equipamentos ou condições ambientais extremas.
-   A frequência de condições de risco e anomalias pode justificar a **compra de um ar condicionado** para manter a qualidade dos alimentos.
""")
