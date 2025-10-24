import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest 
import io
import os
from datetime import datetime

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
Este aplicativo demonstra como utilizar dados de um termo-higrômetro para análise de condições do estoque, como temperatura e umidade,
aplica técnicas de machine learning para análise e auxilio na tomada de decisões. Período de coleta de dados: 09/13/2025 - 10/17/2025
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
        # Ignora a primeira linha que é o MAC, se ainda estiver presente após a limpeza
        if cleaned_lines and len(cleaned_lines) > 1:
            cleaned_lines = cleaned_lines[1:] 
        
        file_content_cleaned = '\n'.join(cleaned_lines)
        
        # Usar StringIO para ler o conteúdo da string como se fosse um arquivo
        data = io.StringIO("data\ttemperatura\tumidade\tponto_orvalho\tvpd\n" + file_content_cleaned) # Adiciona cabeçalho se o original foi removido

        # 2. Leitura com Pandas
        df = pd.read_csv(
            data, 
            sep='\t', 
            header=0,
            engine='python',
            skipinitialspace=True
        )
        
        # 3. Limpeza e Conversão
        df.columns = ['data', 'temperatura', 'umidade', 'ponto_orvalho', 'vpd'] # Garante os nomes corretos
        df = df.dropna(subset=['data']) # Remove linhas sem data
        
        # O formato da data é Mês/Dia/Ano Hora:Minuto
        df['data'] = pd.to_datetime(df['data'], format='%m/%d/%Y %H:%M', errors='coerce')
        
        # *** CORREÇÃO DO ERRO 'errors=coerce' ***
        # Converter colunas numéricas, substituindo vírgula por ponto
        for col in ['temperatura', 'umidade', 'ponto_orvalho', 'vpd']:
            # 1. Trata a coluna como string e substitui vírgula por ponto
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            
            # 2. Usa pd.to_numeric para converter para float, forçando falhas para NaN
            # Este método aceita 'errors=coerce' e resolve o problema.
            df[col] = pd.to_numeric(df[col], errors='coerce')


        # Remove linhas com valores NaN após a conversão (que não puderam ser convertidos)
        df = df.dropna(subset=['temperatura', 'umidade'])
        
        return df.sort_values(by='data').reset_index(drop=True)

    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{file_path}' não foi encontrado. Verifique se ele está no mesmo diretório.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar e pré-processar os dados: {e}")
        st.code(f"Detalhe do erro: {e}", language='text')
        return None

# Carrega os dados usando o caminho do arquivo
df_original = load_and_preprocess_data(FILE_PATH)


# --- BARRA LATERAL (SIDEBAR) PARA CONTROLES INTERATIVOS ---
st.sidebar.header("Controles de Análise")

df = None
anomalias_count = 0
perc_inadequado = 0
ultima_temp = 0
ultima_umid = 0
# Taxa de contaminação FIXADA em 5%, pois o controle foi removido da sidebar
contamination_rate = 0.05 

if df_original is not None and not df_original.empty:
    
    # 1. Slider de Filtro por Data (ÚNICO CONTROLE NA SIDEBAR)
    min_date = df_original['data'].min().date()
    max_date = df_original['data'].max().date()

    date_range = st.sidebar.slider(
        "Selecione o Intervalo de Datas:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MM/DD/YYYY"
    )

    start_datetime = datetime.combine(date_range[0], datetime.min.time())
    end_datetime = datetime.combine(date_range[1], datetime.max.time())
    
    # Filtrar o DataFrame
    df = df_original[
        (df_original['data'] >= start_datetime) & 
        (df_original['data'] <= end_datetime)
    ].copy()
    
    if df.empty:
        st.error("O intervalo de datas selecionado não contém dados.")
        df = None # Define df como None para pular as seções de cálculo
    else:
        # -----------------------------------------------------------
        # CÁLCULOS DE CLASSIFICAÇÃO E ANOMALIAS PARA USO NOS KPIS
        # -----------------------------------------------------------
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
        
        # Cálculo das Anomalias com Isolation Forest usando o valor fixo
        model_if = IsolationForest(contamination=contamination_rate, random_state=42)
        df['anomalia_score'] = model_if.fit_predict(df[['temperatura', 'umidade']])
        
        df['is_outlier'] = df['anomalia_score'].apply(lambda x: 1 if x == -1 else 0)
        anomalias_count = df['is_outlier'].sum()
        
        anomalias_df = df[df['is_outlier'] == 1].sort_values(by='data', ascending=False)
        
# -----------------------------------------------------------

# -----------------------------------------------------------
# KPIS VISUAIS NO TOPO
# -----------------------------------------------------------
if df is not None and not df.empty:
    st.header("Indicadores Chave de Desempenho (KPIs) 📈")
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4) 
    
    # Define a cor do delta para os KPIs
    delta_color_kpi3 = "inverse" if perc_inadequado > 10 else "normal"
    delta_color_kpi4 = "inverse" if anomalias_count > 0 else "normal"
    
    # KPI 1: Última Temperatura
    col_kpi1.metric(
        label="Última Temperatura (°C) 🌡️",
        value=f"{ultima_temp:.1f}",
        delta=f"{df['temperatura'].mean():.1f} Média (Filtro)", 
        delta_color="off" 
    )
    
    # KPI 2: Última Umidade
    col_kpi2.metric(
        label="Última Umidade (%) 💧",
        value=f"{ultima_umid:.1f}",
        delta=f"{df['umidade'].mean():.1f} Média (Filtro)",
        delta_color="off" 
    )
    
    # KPI 3: Porcentagem de Tempo Inadequado
    col_kpi3.metric(
        label="% Tempo Inadequado (Filtro) ⚠️",
        value=f"{perc_inadequado:.1f} %",
        delta=f"{inadequado_count} Pontos", 
        delta_color=delta_color_kpi3
    )

    # NOVO KPI 4: Contagem de Anomalias (Isolation Forest)
    col_kpi4.metric(
        label=f"Anomalias Detectadas (Taxa {contamination_rate*100:.0f}%) 🚨",
        value=f"{anomalias_count}",
        delta="Isolation Forest", 
        delta_color=delta_color_kpi4
    )
    
# -----------------------------------------------------------

# -----------------------------------------------------------
# NOVA SEÇÃO: ALERTAS
# -----------------------------------------------------------
def display_alerts(df, perc_inadequado, anomalias_count, ultima_umid, ultima_temp):
    st.header("Alertas de Status e Ações Recomendadas 🔔")
    
    alerta_ativo = False
    
    # 1. Alerta de Condição Inadequada Atual
    if df['condicao_armazenamento'].iloc[-1] == "Inadequado":
        st.error(f"""
        **ALERTA IMEDIATO:** A última condição registrada de armazenamento é **INADEQUADA**! 
        (Temp: {ultima_temp:.1f}°C, Umidade: {ultima_umid:.1f}%).
        Ação: Verifique imediatamente o sistema de ventilação/ar condicionado.
        """)
        alerta_ativo = True

    # 2. Alerta de Anomalias Detectadas
    if anomalias_count > 0:
        st.warning(f"""
        **ANOMALIA DETECTADA:** O modelo Isolation Forest encontrou **{anomalias_count}** pontos atípicos nos dados.
        Ação: Investigue os registros anômalos (Seção 5.1) e verifique a integridade do sensor.
        """)
        alerta_ativo = True

    # 3. Alerta de Histórico Crítico (Acima de 10% de Inadequação)
    if perc_inadequado > 10:
        st.warning(f"""
        **RISCO CRÍTICO:** O estoque passou {perc_inadequado:.1f}% do tempo em condições inadequadas.
        Ação: Considere a necessidade de um sistema de controle de clima mais robusto.
        """)
        alerta_ativo = True
    
    if not alerta_ativo:
        st.success("Tudo certo! As condições de armazenamento estão estáveis e sem alertas ativos.")

if df is not None and not df.empty:
    # Chama a função de alertas após os KPIs
    display_alerts(df, perc_inadequado, anomalias_count, ultima_umid, ultima_temp)
    st.markdown("---") # Linha divisória após os alertas

# -----------------------------------------------------------

if df is not None and not df.empty:
    st.header("1. Carregamento e Tratamento dos Dados")
    st.write("Dados carregados com sucesso. Primeiras 5 linhas (no período filtrado):")
    st.dataframe(df[['data', 'temperatura', 'umidade', 'condicao_armazenamento']].head())
    
    # --- Seção de Visualização Inicial ---
    st.header("2. Visualização Histórica e Anomalias (Série Temporal) 📊")
    st.markdown("Os pontos detectados como anomalias pelo Isolation Forest são destacados em vermelho.")
    
    # Cria o gráfico com Matplotlib para poder destacar os pontos
    fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
    
    # Dados Normais
    df_normal = df[df['is_outlier'] == 0]
    ax_hist.plot(df_normal["data"], df_normal["temperatura"], label="Temperatura (°C) Normal", color="#4A90E2", alpha=0.7)
    ax_hist.plot(df_normal["data"], df_normal["umidade"], label="Umidade (%) Normal", color="#7ED321", alpha=0.7)
    
    # Dados Anômalos (destaque)
    df_outlier = df[df['is_outlier'] == 1]
    if not df_outlier.empty:
        # Pontos anômalos
        ax_hist.scatter(df_outlier["data"], df_outlier["temperatura"], label="Anomalia (Temp)", color="#FF5733", s=50, zorder=5)
        ax_hist.scatter(df_outlier["data"], df_outlier["umidade"], label="Anomalia (Umid)", color="#FF5733", s=50, zorder=5, marker='x')

    ax_hist.legend()
    ax_hist.set_title("Dados de Temperatura e Umidade no Estoque (Com Anomalias)", fontsize=16)
    ax_hist.set_xlabel("Data")
    ax_hist.set_ylabel("Valor")
    ax_hist.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_hist)


    # --- Seção de Previsão de Temperatura (Análise Preditiva) ---
    st.header("3. Previsão de Temperatura (Machine Learning) 🧠")
    st.markdown("Um modelo de regressão linear (simples) prevê a tendência da temperatura com base no tempo.")

    # Cria uma coluna 'hora' como índice numérico para o tempo
    df["hora"] = np.arange(len(df))

    # Remove anomalias para um treino mais estável, se houver:
    df_trainable = df[df['is_outlier'] == 0]
    
    X = df_trainable[["hora"]]
    y = df_trainable["temperatura"]

    # Dividir dados em treino e teste (mantendo a ordem temporal com shuffle=False)
    if len(df_trainable) > 10:
        test_size = max(0.2, 5 / len(df_trainable)) # Garante que o test_size seja pelo menos 5 amostras, ou 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        
        # Cálculo e exibição do RMSE
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        st.info(f"Qualidade do Modelo (RMSE): **{rmse:.4f} °C**. Quanto mais próximo de zero, melhor a previsão.")

        fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
        # Plotar a temperatura real (de todos os dados, incluindo os removidos do treino, mas plotando o y original)
        ax_pred.plot(df["data"], df["temperatura"], label="Temperatura Real", color="#4A90E2")
        
        # Plotar a previsão apenas para o período de teste (usando o índice do df original)
        # O .loc busca as datas originais que correspondem aos índices de teste
        ax_pred.plot(df.loc[X_test.index, "data"], preds, label="Previsão (Período de Teste)", color="#FF5733", linestyle='--')
        
        ax_pred.legend()
        ax_pred.set_title("Previsão de Temperatura (Regressão Linear)", fontsize=16)
        ax_pred.set_xlabel("Data")
        ax_pred.set_ylabel("Temperatura (°C)")
        ax_pred.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_pred)
    else:
        st.warning("Dados insuficientes (após remoção de anomalias, se houver) para treino e previsão do modelo de regressão linear.")


    # --- Seção de Classificação de Condições de Armazenamento ---
    st.header("4. Classificação das Condições de Armazenamento 🌡️💧")
    st.markdown("""
    As condições críticas são definidas por:
    - <span style='color:green;'>**Adequado**</span>: Temperatura < 25 ºC e Umidade < 70 %
    - <span style='color:red;'>**Inadequado**</span>: Temperatura ≥ 25 ºC ou Umidade ≥ 70 %
    
    """, unsafe_allow_html=True)
    
    col_chart, col_table = st.columns(2)
    
    with col_chart:
        st.subheader("Distribuição de Condições (Contagem de Pontos)")
        
        # Cria um DataFrame temporário para usar cores específicas no gráfico
        contagem_df = contagem_condicoes.to_frame(name='Contagem')
        contagem_df.index.name = 'Condição'
        contagem_df = contagem_df.reset_index()
        
        # Define as cores
        contagem_df['Cor'] = contagem_df['Condição'].map({'Adequado': "#6EC70F", 'Inadequado': "#E10707"})
        
        # ------------------------------------------------------
        # CORREÇÃO PARA APLICAÇÃO DE CORES (Abordagem robusta)
        # ------------------------------------------------------
        
        # 1. Definir o mapeamento de cores
        color_map = {'Adequado': "#6EC70F", 'Inadequado': "#E10707"}
        
        # 2. Criar um novo DataFrame com a ordem forçada e todas as categorias
        contagem_df_final = pd.DataFrame({
            'Condição': list(color_map.keys()),
            'Contagem': [contagem_condicoes.get('Adequado', 0), contagem_condicoes.get('Inadequado', 0)],
            'Cor': list(color_map.values())
        })
        # Remove linhas com contagem zero 
        contagem_df_final = contagem_df_final[contagem_df_final['Contagem'] > 0]
        
        st.bar_chart(
            contagem_df_final, 
            x='Condição', 
            y='Contagem', 
            color='Cor' # Agora 'Cor' é uma coluna de dados, o que é mais robusto
        )
        
    with col_table:
        st.subheader("Detalhe da Tabela (Últimos Registros no Filtro)")
        st.dataframe(df[['data', 'temperatura', 'umidade', 'condicao_armazenamento']].tail())


    # --- Seção de Detecção de Anomalias (Isolation Forest) ---
    st.header("5. Detecção de Anomalias (Machine Learning e Estatística) 🚨")
    st.markdown(f"""
    **5.1. Detecção por Isolation Forest**
    O modelo **Isolation Forest** foi aplicado para encontrar valores atípicos, com base na taxa de contaminação de **{contamination_rate*100:.0f}%** (fixada em 5%).
    - **Total de Anomalias Detectadas:** <span style='color:red; font-size: 1.2em;'>**{anomalias_count}**</span> de {total_pontos} registros no período filtrado.
    """, unsafe_allow_html=True)

    if not anomalias_df.empty:
        st.subheader("Últimas Anomalias de Temperatura e Umidade (Isolation Forest)")
        st.markdown("Estes são os pontos mais recentes que o modelo considerou anômalos:")
        # Exibe a tabela com as 10 últimas anomalias
        st.dataframe(anomalias_df[['data', 'temperatura', 'umidade']].head(10))
    else:
        st.success(f"Nenhuma anomalia detectada pelo Isolation Forest (com taxa de contaminação de {contamination_rate*100:.0f}%).")

    # --- Scatter Plot 2D de Anomalias ---
    st.markdown("---")
    st.subheader("5.2. Visualização 2D de Anomalias (Temp vs. Umidade)")
    st.markdown("Este gráfico mostra a distribuição dos dados e destaca as anomalias detectadas pelo Isolation Forest (Pontos Vermelhos).")
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    
    # Plota dados normais em azul
    df_normal = df[df['is_outlier'] == 0]
    ax_scatter.scatter(df_normal['umidade'], df_normal['temperatura'], label='Normal', color='#4A90E2', alpha=0.6)
    
    # Plota anomalias em vermelho
    df_outlier = df[df['is_outlier'] == 1] # Redefine df_outlier pois o original foi filtrado no bloco principal
    if not df_outlier.empty:
        ax_scatter.scatter(df_outlier['umidade'], df_outlier['temperatura'], label='Anomalia (IF)', color='#FF5733', s=100, marker='X', zorder=5)
    
    ax_scatter.set_title('Anomalias no Espaço de Temperatura vs. Umidade', fontsize=16)
    ax_scatter.set_xlabel('Umidade (%)')
    ax_scatter.set_ylabel('Temperatura (°C)')
    ax_scatter.grid(True, linestyle='--', alpha=0.7)
    ax_scatter.legend()
    st.pyplot(fig_scatter)
    
    
    # --- Boxplot (Análise Estatística Visual) ---
    st.markdown("---")
    st.subheader("5.3. Visualização Estatística de Anomalias (Boxplot) 📦")
    st.markdown("""
    Os boxplots (Diagramas de Caixa) são utilizados para identificar valores atípicos (outliers estatísticos).
    """)

    # Usando colunas para o layout responsivo
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Anomalias de Temperatura (Boxplot)")
        fig_temp_boxplot, ax_temp_boxplot = plt.subplots(figsize=(8, 6))
        # Define um estilo de boxplot mais agradável e vertical
        ax_temp_boxplot.boxplot(df['temperatura'], vert=True, patch_artist=True, boxprops=dict(facecolor='#4A90E2', color='darkblue'), medianprops=dict(color='red'), flierprops=dict(marker='o', markersize=8, markeredgecolor='red', markerfacecolor='#FF5733'))
        ax_temp_boxplot.set_title('Boxplot da Temperatura', fontsize=14)
        ax_temp_boxplot.set_ylabel('Temperatura (°C)')
        ax_temp_boxplot.set_xticklabels(['Temperatura']) # Rótulo fixo no eixo x
        ax_temp_boxplot.grid(True, linestyle='--', alpha=0.6, axis='y')
        st.pyplot(fig_temp_boxplot)

    with col2:
        st.subheader("Anomalias de Umidade (Boxplot)")
        fig_umid_boxplot, ax_umid_boxplot = plt.subplots(figsize=(8, 6))
        # Define um estilo de boxplot mais agradável e vertical
        ax_umid_boxplot.boxplot(df['umidade'], vert=True, patch_artist=True, boxprops=dict(facecolor='#7ED321', color='darkgreen'), medianprops=dict(color='red'), flierprops=dict(marker='o', markersize=8, markeredgecolor='red', markerfacecolor='#FF5733'))
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
    - A **Detecção de Anomalias** (Seção 5) é crucial para revelar falhas em equipamentos ou leituras de sensores atípicas.
    - Se a contagem de períodos "Inadequado" for alta, a **compra de um ar condicionado** ou melhorias na ventilação podem ser justificadas para manter a qualidade dos produtos.
    """)
    
  
