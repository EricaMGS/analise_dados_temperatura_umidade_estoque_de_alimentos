import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io
import os # Importar para manipulação de arquivos/caminhos

# --------------------------------------------------------------------------
# CONFIGURAÇÃO DE CARREGAMENTO DE DADOS (Padrão para uso local)
# Use 'dados_pi.txt' se estiver no mesmo diretório.
FILE_PATH = 'dados_pi.txt'
# --------------------------------------------------------------------------

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

@st.cache_data
def load_and_preprocess_data(file_path):
    # Tenta ler o arquivo diretamente do disco
    try:
        # Abre o arquivo com a codificação correta
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            
        # 1. PRÉ-PROCESSAMENTO: Correção de anos inválidos e limpeza de linhas.
        # Correção do ano '2925' para '2025' que causa o erro 'Out of bounds nanosecond timestamp'
        file_content = file_content.replace('2925', '2025')
        
        # Limpar linhas vazias e remover espaços/tabulações extras no início e fim de CADA linha.
        lines = file_content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        file_content_cleaned = '\n'.join(cleaned_lines)
        
        # Usar StringIO para ler o conteúdo da string como se fosse um arquivo
        data = io.StringIO(file_content_cleaned)

        # 2. Leitura com Pandas (usando engine='python' para tolerância)
        df = pd.read_csv(
            data, 
            sep='\t', 
            skiprows=1, # Ignora a linha MAC após a limpeza (que deve ser a primeira linha limpa)
            header=0,
            engine='python', # Força o parser Python (mais tolerante a irregularidades)
            skipinitialspace=True # Ignora espaços extras após o delimitador
        )
        
        # Renomear colunas para fácil acesso, removendo caracteres especiais:
        df.columns = ['data', 'temperatura', 'umidade', 'ponto_orvalho', 'vpd']
        
        # 3. Limpeza e Conversão
        df = df.dropna(how='all') 
        
        # O formato da data é Mês/Dia/Ano Hora:Minuto
        df['data'] = pd.to_datetime(df['data'], format='%m/%d/%Y %H:%M')
        
        # Converter colunas numéricas, substituindo vírgula por ponto
        for col in ['temperatura', 'umidade', 'ponto_orvalho', 'vpd']:
            if df[col].dtype == 'object':
                 df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        return df

    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{file_path}' não foi encontrado. Verifique se ele está no mesmo diretório.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar e pré-processar os dados: {e}")
        return None

# Carrega os dados usando o caminho do arquivo
df = load_and_preprocess_data(FILE_PATH)

# -----------------------------------------------------------
# CÁLCULOS DE CLASSIFICAÇÃO PARA USO NOS KPIS
# -----------------------------------------------------------
if df is not None and not df.empty:
    def classificar_condicao(temp, umid):
        if temp < 25 and umid < 70:
            return "Adequado"
        else:
            return "Inadequado"

    df['condicao_armazenamento'] = df.apply(lambda row: classificar_condicao(row['temperatura'], row['umidade']), axis=1)
    
    # Cálculo da contagem e porcentagem inadequada
    contagem_condicoes = df['condicao_armazenamento'].value_counts()
    total_pontos = len(df)
    inadequado_count = contagem_condicoes.get('Inadequado', 0)
    
    # Prevenção de divisão por zero
    perc_inadequado = (inadequado_count / total_pontos) * 100 if total_pontos > 0 else 0
    
    # Pega o último registro para KPI atual
    ultima_temp = df['temperatura'].iloc[-1]
    ultima_umid = df['umidade'].iloc[-1]
    
# -----------------------------------------------------------


# -----------------------------------------------------------
# KPIS VISUAIS NO TOPO
# -----------------------------------------------------------
if df is not None and not df.empty:
    st.header("Indicadores Chave de Desempenho (KPIs) 📈")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    # Define a cor do delta para o KPI 3
    delta_color_kpi3 = "inverse" if perc_inadequado > 10 else "normal"
    
    # KPI 1: Última Temperatura (Cor azul para temperatura)
    col_kpi1.metric(
        label="Temperatura Atual (°C) 🌡️",
        value=f"{ultima_temp:.1f}",
        delta=f"{df['temperatura'].mean():.1f} Média", # Mostrar a média como referência
        delta_color="off" 
    )
    
    # KPI 2: Última Umidade (Cor verde para umidade)
    col_kpi2.metric(
        label="Umidade Atual (%) 💧",
        value=f"{ultima_umid:.1f}",
        delta=f"{df['umidade'].mean():.1f} Média",
        delta_color="off" 
    )
    
    # KPI 3: Porcentagem de Tempo Inadequado (Cor vermelha/verde para alerta)
    col_kpi3.metric(
        label="% Tempo Inadequado (Total) ⚠️",
        value=f"{perc_inadequado:.1f} %",
        # O delta indica quantos pontos estão inadequados
        delta=f"{inadequado_count} Pontos", 
        delta_color=delta_color_kpi3
    )
    
# -----------------------------------------------------------

if df is not None and not df.empty:
    st.header("1. Carregamento e Tratamento dos Dados")
    st.write("Dados carregados com sucesso. Primeiras 5 linhas:")
    st.dataframe(df.head())
    

    # --- Seção de Visualização Inicial ---
    st.header("2. Visualização Histórica (Série Temporal) 📊")
    st.markdown("Veja como a temperatura (em azul) e a umidade (em verde) variaram ao longo do tempo. Use o zoom no gráfico interativo abaixo.")
    
    # Configurando cores para o line_chart do Streamlit
    st.line_chart(
        df, 
        x="data", 
        y=["temperatura", "umidade"],
        color=["#4A90E2", "#7ED321"] # Azul para Temp, Verde para Umidade
    )

    if st.button("Gerar Gráfico de Linhas (Matplotlib)"):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["data"], df["temperatura"], label="Temperatura (°C)", color="#4A90E2")
        ax.plot(df["data"], df["umidade"], label="Umidade (%)", color="#7ED321")
        ax.legend()
        ax.set_title("Dados de Temperatura e Umidade no Estoque (Matplotlib)", fontsize=16)
        ax.set_xlabel("Data")
        ax.set_ylabel("Valor")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # --- Seção de Previsão de Temperatura (Análise Preditiva) ---
    st.header("3. Previsão de Temperatura (Machine Learning) 🧠")
    st.markdown("Um modelo de regressão linear (simples) prevê a tendência da temperatura com base no tempo.")

    # Cria uma coluna 'hora' como índice numérico para o tempo
    df["hora"] = np.arange(len(df))

    X = df[["hora"]]
    y = df["temperatura"]

    # Dividir dados em treino e teste (mantendo a ordem temporal com shuffle=False)
    if len(df) > 10:
        test_size = max(0.2, 5 / len(df)) # Garante que o test_size seja pelo menos 5 amostras, ou 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        
        # Cálculo e exibição do RMSE
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        st.info(f"Qualidade do Modelo (RMSE): **{rmse:.4f} °C**. Quanto mais próximo de zero, melhor a previsão.")

        fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
        # Plotar a temperatura real
        ax_pred.plot(df["data"], y, label="Temperatura Real", color="#4A90E2")
        
        # Plotar a previsão apenas para o período de teste
        ax_pred.plot(df["data"].iloc[len(X_train):], preds, label="Previsão (Período de Teste)", color="#FF5733", linestyle='--')
        
        ax_pred.legend()
        ax_pred.set_title("Previsão de Temperatura (Regressão Linear)", fontsize=16)
        ax_pred.set_xlabel("Data")
        ax_pred.set_ylabel("Temperatura (°C)")
        ax_pred.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_pred)
    else:
        st.warning("Dados insuficientes para treino e previsão do modelo de regressão linear.")


    # --- Seção de Classificação de Condições de Armazenamento ---
    st.header("4. Classificação das Condições de Armazenamento 🌡️💧")
    st.markdown("""
    As condições críticas são definidas por:
    - <span style='color:green;'>**Adequado**</span>: Temperatura < 25 ºC e Umidade < 70 %
    - <span style='color:red;'>**Inadequado**</span>: Temperatura ≥ 25 ºC ou Umidade ≥ 70 %
    
    """, unsafe_allow_html=True)
    
    # A variável contagem_condicoes é uma Series e será usada diretamente no bar_chart.
    
    col_chart, col_table = st.columns(2)
    
    with col_chart:
        st.subheader("Distribuição de Condições (Contagem de Pontos)")
        
        # Cria um DataFrame temporário para usar cores específicas no gráfico
        contagem_df = contagem_condicoes.to_frame(name='Contagem')
        contagem_df.index.name = 'Condição'
        contagem_df = contagem_df.reset_index()
        
        # Define as cores (CORREÇÃO APLICADA AQUI!)
        contagem_df['Cor'] = contagem_df['Condição'].map({'Adequado': '#7ED321', 'Inadequado': '#FF5733'})
        
        st.bar_chart(
            contagem_df, 
            x='Condição', 
            y='Contagem', 
            color='Cor' # Usa a coluna 'Cor' para colorir as barras
        )
        
    with col_table:
        st.subheader("Detalhe da Tabela")
        st.dataframe(df[['data', 'temperatura', 'umidade', 'condicao_armazenamento']].tail())


    # --- Seção de Detecção de Anomalias ---
    st.header("5. Detecção de Anomalias (Boxplot) 📦")

    st.markdown("""
    Os boxplots são usados para identificar valores atípicos (anomalias), que são pontos
    que caem fora de 1.5 vezes o intervalo interquartil (IQR).
    """)

    # Usando colunas para o layout responsivo
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Anomalias de Temperatura")
        fig_temp_boxplot, ax_temp_boxplot = plt.subplots(figsize=(8, 6))
        # Define um estilo de boxplot mais agradável e vertical
        ax_temp_boxplot.boxplot(df['temperatura'], vert=True, patch_artist=True, boxprops=dict(facecolor='#4A90E2', color='darkblue'), medianprops=dict(color='red'))
        ax_temp_boxplot.set_title('Boxplot da Temperatura', fontsize=14)
        ax_temp_boxplot.set_ylabel('Temperatura (°C)')
        ax_temp_boxplot.set_xticklabels(['Temperatura']) # Rótulo fixo no eixo x
        ax_temp_boxplot.grid(True, linestyle='--', alpha=0.6, axis='y')
        st.pyplot(fig_temp_boxplot)

    with col2:
        st.subheader("Anomalias de Umidade")
        fig_umid_boxplot, ax_umid_boxplot = plt.subplots(figsize=(8, 6))
        # Define um estilo de boxplot mais agradável e vertical
        ax_umid_boxplot.boxplot(df['umidade'], vert=True, patch_artist=True, boxprops=dict(facecolor='#7ED321', color='darkgreen'), medianprops=dict(color='red'))
        ax_umid_boxplot.set_title('Boxplot da Umidade', fontsize=14)
        ax_umid_boxplot.set_ylabel('Umidade (%)')
        ax_umid_boxplot.set_xticklabels(['Umidade']) # Rótulo fixo no eixo x
        ax_umid_boxplot.grid(True, linestyle='--', alpha=0.6, axis='y')
        st.pyplot(fig_umid_boxplot)

    # --- Seção de Conclusão e Insights ---
    st.header("6. Conclusão da Análise ✨")
    st.markdown("""
    - O uso de um **termo-higrômetro** fornece dados essenciais para o monitoramento de estoque.
    - Técnicas de **machine learning**, como regressão linear, permitem prever tendências de condições futuras.
    - A **classificação de dados** ajuda a identificar rapidamente períodos de risco para o armazenamento de produtos.
    - A análise de **anomalias** por meio de boxplots pode revelar falhas em equipamentos ou condições ambientais extremas.
    - Se a contagem de períodos "Inadequado" for alta, a **compra de um ar condicionado** ou melhorias na ventilação podem ser justificadas para manter a qualidade dos produtos.
    """)


