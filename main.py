import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import os
from typing import List
from datetime import datetime, date
import locale

# Importações do framework LangChain e bibliotecas auxiliares
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from pydantic import BaseModel, Field
from langchain import hub # Biblioteca para acessar templates de prompt da comunidade

# ########################################################################################################
# NÚCLEO DE INTELIGÊNCIA: Sistema de reconhecimento automático de planilhas
# ########################################################################################################
def identificar_arquivo(nome_arquivo, arquivo_bytes):
    """Sistema inteligente que analisa estrutura das planilhas para classificação automática."""
    try:
        df_preview = pd.read_excel(arquivo_bytes, nrows=5, engine='openpyxl')
        cols = {str(col).strip().upper() for col in df_preview.columns}
        
        if 'TITULO DO CARGO' in cols and 'DESC. SITUACAO' in cols: return "ATIVOS"
        if 'DIAS DE FÉRIAS' in cols: return "FERIAS"
        if 'DATA DEMISSÃO' in cols and 'COMUNICADO DE DESLIGAMENTO' in cols: return "DESLIGADOS"
        if any('DIAS UTEIS' in col for col in cols): return "DIAS_UTEIS"
        if 'VALOR' in cols and any('ESTADO' in col for col in cols): return "VALORES"
        if 'CADASTRO' in cols and 'VALOR' in cols: return "EXTERIOR"
        
        nome_upper = nome_arquivo.upper()
        if 'APRENDIZ' in nome_upper: return "APRENDIZ"
        if 'ESTÁGIO' in nome_upper or 'ESTAGIO' in nome_upper: return "ESTAGIO"
        if 'AFASTAMENTO' in nome_upper: return "AFASTAMENTOS"

        return "DESCONHECIDO"
    except Exception:
        return "INVALIDO"

# ########################################################################################################
# TOOLKIT DE PROCESSAMENTO: Conjunto de ferramentas especializadas para automação
# ########################################################################################################

@tool
def aplicar_regras_de_exclusao(ativos_key: str, aprendiz_key: str = None, estagio_key: str = None, exterior_key: str = None, afastamentos_key: str = None) -> str:
    """
    Ferramenta que processa exclusões de colaboradores conforme políticas corporativas.
    Filtra a base de funcionários ativos removendo perfis não elegíveis ao benefício.
    """
    dfs = st.session_state.get('dfs', {})
    if ativos_key not in dfs: return "Erro: Base de dados de funcionários ativos não localizada."
    
    df_ativos = dfs[ativos_key]
    df_elegiveis = df_ativos.copy()
    matriculas_para_excluir = set()

    if aprendiz_key and dfs.get(aprendiz_key) is not None: matriculas_para_excluir.update(dfs[aprendiz_key]['MATRICULA'].tolist())
    if estagio_key and dfs.get(estagio_key) is not None: matriculas_para_excluir.update(dfs[estagio_key]['MATRICULA'].tolist())
    if exterior_key and dfs.get(exterior_key) is not None: matriculas_para_excluir.update(dfs[exterior_key]['Cadastro'].tolist())
    if afastamentos_key and dfs.get(afastamentos_key) is not None: matriculas_para_excluir.update(dfs[afastamentos_key]['MATRICULA'].tolist())

    diretores = df_elegiveis[df_elegiveis['TITULO DO CARGO'].str.contains("DIRETOR", case=False)]
    if not diretores.empty: matriculas_para_excluir.update(diretores['MATRICULA'].tolist())
        
    df_elegiveis = df_elegiveis[~df_elegiveis['MATRICULA'].isin(list(matriculas_para_excluir))]
    
    st.session_state.dfs['ELEGIVEIS'] = df_elegiveis
    return "Base de funcionários elegíveis processada e armazenada com identificador 'ELEGIVEIS'."

@tool
def calcular_dias_de_beneficio(elegiveis_key: str, dias_uteis_key: str, ferias_key: str = None, desligados_key: str = None) -> str:
    """
    Motor de cálculo que determina quantidade de dias de benefício por colaborador.
    Considera calendário útil, períodos de férias e eventuais desligamentos no período.
    """
    dfs = st.session_state.get('dfs', {})
    if elegiveis_key not in dfs or dias_uteis_key not in dfs: return "Erro: Informações essenciais para cálculo não disponíveis."

    df_elegiveis = dfs[elegiveis_key]
    df_dias_uteis = dfs[dias_uteis_key]
    df_ferias = dfs.get(ferias_key)
    df_desligados = dfs.get(desligados_key)

    df_dias_uteis_clean = df_dias_uteis.copy()
    df_dias_uteis_clean.columns = ['Sindicato', 'Dias_Uteis_Mes']
    df_calculo = pd.merge(df_elegiveis, df_dias_uteis_clean, on='Sindicato', how='left')

    if df_desligados is not None and not df_desligados.empty:
        df_desligados.columns = df_desligados.columns.str.strip()
        df_desligados['DATA DEMISSÃO'] = pd.to_datetime(df_desligados['DATA DEMISSÃO'])

    def calcular(colaborador):
        dias_a_pagar = colaborador.get('Dias_Uteis_Mes', 0)
        if df_ferias is not None:
            ferias_colaborador = df_ferias[df_ferias['MATRICULA'] == colaborador['MATRICULA']]
            if not ferias_colaborador.empty: dias_a_pagar -= ferias_colaborador['DIAS DE FÉRIAS'].iloc[0]
        if df_desligados is not None:
            desligado_colaborador = df_desligados[df_desligados['MATRICULA'] == colaborador['MATRICULA']]
            if not desligado_colaborador.empty:
                data_desligamento = desligado_colaborador['DATA DEMISSÃO'].iloc[0]
                if data_desligamento.day <= 15: return 0
                else: return np.busday_count('2025-05-01', data_desligamento.strftime('%Y-%m-%d'))
        return max(0, dias_a_pagar)

    df_calculo['Dias_A_Pagar'] = df_calculo.apply(calcular, axis=1)
    st.session_state.dfs['DIAS_CALCULADOS'] = df_calculo
    return "Processamento de dias de benefício concluído e salvo como 'DIAS_CALCULADOS'."

@tool
def calcular_valores_finais(dias_calculados_key: str, valores_key: str) -> str:
    """
    Processador final que converte dias em valores monetários aplicando tabelas regionais.
    Gera relatório completo com custos empresariais e descontos dos colaboradores.
    """
    dfs = st.session_state.get('dfs', {})
    if dias_calculados_key not in dfs or valores_key not in dfs: return "Erro: Dados para valorização não encontrados no sistema."

    df_com_dias_pagos = dfs[dias_calculados_key]
    df_valores = dfs[valores_key]
    
    df_valores_clean = df_valores.copy()
    df_valores_clean.columns = ['Estado', 'VALOR']
    df_valores_clean.dropna(inplace=True)
    
    df_final = df_com_dias_pagos.copy()
    df_final['Sigla_Estado'] = df_final['Sindicato'].str.extract(r'\b(SP|RS|RJ|PR)\b')
    mapa_estados = {'SP': 'São Paulo', 'RS': 'Rio Grande do Sul', 'RJ': 'Rio de Janeiro', 'PR': 'Paraná'}
    df_final['Estado'] = df_final['Sigla_Estado'].map(mapa_estados)
    
    df_final = pd.merge(df_final, df_valores_clean, on='Estado', how='left')
    df_final['Valor_Total_VR'] = df_final['Dias_A_Pagar'] * df_final['VALOR']
    df_final['Custo_Empresa'] = df_final['Valor_Total_VR'] * 0.80
    df_final['Desconto_Profissional'] = df_final['Valor_Total_VR'] * 0.20
    
    df_layout_final = df_final[['MATRICULA', 'EMPRESA', 'Valor_Total_VR', 'Custo_Empresa', 'Desconto_Profissional']]
    
    st.session_state.dfs['RESULTADO_FINAL'] = df_layout_final
    return "Processamento financeiro finalizado. Relatório preparado para exportação."

# ########################################################################################################
# INTERFACE PRINCIPAL: Sistema de gerenciamento e controle do processamento
# ########################################################################################################

st.set_page_config(layout="wide", page_title="🤖 Sistema Automatizado de Benefícios", page_icon="🤖")
st.title("🤖 Sistema Automatizado de Processamento de Benefícios Alimentação")
st.markdown("*Plataforma inteligente para automação completa de Vale Refeição/Alimentação*")

# Configuração do armazenamento de dados na sessão
if 'dfs' not in st.session_state: st.session_state.dfs = {}
if 'arquivos_processados_log' not in st.session_state: st.session_state.arquivos_processados_log = {}

# ########################################################################################################
# CONFIGURAÇÃO DA API: Interface para inserção de credenciais
# ########################################################################################################

st.sidebar.header("🔐 Configuração de API")
st.sidebar.markdown("*Insira sua chave de API do Google Gemini:*")

# Campo para inserção manual da chave API
GOOGLE_API_KEY = st.sidebar.text_input(
    "Chave API Google Gemini:", 
    type="password",
    placeholder="AIzaSy...",
    help="Insira sua chave de API do Google Gemini para ativar o processamento"
)

# Validação visual da chave
if GOOGLE_API_KEY:
    st.sidebar.success("✅ API configurada e pronta")
    api_ativa = True
else:
    st.sidebar.warning("⚠️ Configure a API para continuar")
    api_ativa = False

# ########################################################################################################
# MÓDULO DE IMPORTAÇÃO: Sistema de upload e processamento de arquivos
# ########################################################################################################

st.header("📂 Importação de Planilhas")
uploaded_files = st.file_uploader(
    "Carregue arquivo ZIP com as planilhas ou envie arquivos Excel individuais:", 
    type=["xlsx", "zip"], 
    accept_multiple_files=True, 
    key="file_uploader",
    help="Aceita arquivos .xlsx individuais ou arquivo .zip contendo múltiplas planilhas"
)

if uploaded_files:
    st.session_state.dfs, st.session_state.arquivos_processados_log = {}, {}
    arquivos_para_processar = []
    for file in uploaded_files:
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as z:
                for filename in z.namelist():
                    if filename.endswith('.xlsx') and not filename.startswith('__MACOSX'):
                        arquivos_para_processar.append((filename, io.BytesIO(z.read(filename))))
        elif file.name.endswith('.xlsx'):
            arquivos_para_processar.append((file.name, file))

    for nome, arquivo in arquivos_para_processar:
        tipo = identificar_arquivo(nome, arquivo)
        st.session_state.arquivos_processados_log[nome] = tipo
        if tipo not in ["DESCONHECIDO", "INVALIDO"]:
            st.session_state.dfs[tipo] = pd.read_excel(arquivo, engine='openpyxl')

# ########################################################################################################
# PAINEL DE CONTROLE: Monitor de status e diagnóstico do sistema
# ########################################################################################################

with st.expander("🔍 Painel de Diagnóstico e Monitoramento", expanded=False):
    if not st.session_state.arquivos_processados_log:
        st.info("Sistema aguardando carregamento de arquivos.")
    else:
        st.subheader("Relatório de Análise Automática:")
        for nome, tipo in st.session_state.arquivos_processados_log.items():
            if tipo == "DESCONHECIDO": 
                st.warning(f"**{nome}** → ❓ **{tipo}** - Estrutura não identificada pelo sistema")
            elif tipo == "INVALIDO": 
                st.error(f"**{nome}** → ❌ **{tipo}** - Arquivo corrompido ou inválido")
            else: 
                st.success(f"**{nome}** → ✅ **{tipo}** - Classificação bem-sucedida")

# Dashboard principal de status
st.header("📊 Dashboard de Status dos Dados")
arquivos_obrigatorios = {"ATIVOS", "DIAS_UTEIS", "VALORES"}
arquivos_opcionais = {"APRENDIZ", "ESTAGIO", "EXTERIOR", "AFASTAMENTOS", "FERIAS", "DESLIGADOS"}
obrigatorios_ok = arquivos_obrigatorios.issubset(st.session_state.dfs.keys())

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Dados Essenciais")
    for key in sorted(list(arquivos_obrigatorios)):
        if key in st.session_state.dfs: 
            st.success(f"✅ {key}: Carregado e processado")
        else: 
            st.error(f"❌ {key}: Pendente")

with col2:
    st.subheader("📝 Dados Complementares")
    for key in sorted(list(arquivos_opcionais)):
        if key in st.session_state.dfs: 
            st.success(f"✅ {key}: Disponível")
        else: 
            st.info(f"ℹ️ {key}: Opcional (não enviado)")

# ########################################################################################################
# MOTOR DE PROCESSAMENTO: Execução automatizada com inteligência artificial
# ########################################################################################################

if obrigatorios_ok and api_ativa:
    st.success("🎯 Sistema preparado! Todas as condições atendidas para processamento.")
    
    if st.button("🚀 Executar Processamento Automatizado", type="primary"):
        
        with st.spinner("🧠 Inteligência artificial processando dados..."):
            try:
                tools = [aplicar_regras_de_exclusao, calcular_dias_de_beneficio, calcular_valores_finais]
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite", 
                    temperature=0, 
                    google_api_key=GOOGLE_API_KEY, 
                    convert_system_message_to_human=True
                )
                
                # Carregamento do template de prompt otimizado
                prompt = hub.pull("hwchase17/structured-chat-agent")
                
                agent = create_structured_chat_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
                
                st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                
                chaves_disponiveis = list(st.session_state.dfs.keys())
                task = f"""
                Você é um especialista em automação de benefícios corporativos. Execute o processamento completo 
                dos dados de Vale Alimentação. Os datasets disponíveis são: {chaves_disponiveis}.
                
                Execute esta sequência de processamento:
                1. Aplique 'aplicar_regras_de_exclusao' usando 'ATIVOS' como dataset principal
                2. Execute 'calcular_dias_de_beneficio' com o resultado 'ELEGIVEIS' da etapa anterior
                3. Finalize com 'calcular_valores_finais' usando 'DIAS_CALCULADOS' da etapa anterior
                
                Confirme a conclusão de cada etapa antes de prosseguir.
                """
                
                response = agent_executor.invoke({"input": task}, {"callbacks": [st_callback]})
                
                st.success("🎉 Processamento concluído com êxito!")
                
                resultado_final_df = st.session_state.dfs.get('RESULTADO_FINAL')
                
                if resultado_final_df is not None:
                    st.subheader("📈 Relatório Final Gerado")
                    st.dataframe(resultado_final_df, use_container_width=True)
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        resultado_final_df.to_excel(writer, index=False, sheet_name='Beneficio_Final_IA')
                    agora = datetime.now()
                    data_br = agora.strftime("%d/%m/%Y")
                    st.download_button(
                        "📥 Download do Relatório Completo", 
                        output.getvalue(), 
                        f"VR MENSAL{data_br}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error("⚠️ Processamento concluído, mas relatório final não foi gerado. Verifique o log de execução.")

            except Exception as e:
                st.error(f"❌ Erro durante o processamento automatizado: {e}")
                st.info("💡 Verifique se a chave de API está correta e tente novamente.")

elif not obrigatorios_ok:
    st.info("📥 Sistema aguardando carregamento dos datasets obrigatórios.")
elif not api_ativa:
    st.info("🔐 Configure sua chave de API na barra lateral para ativar o processamento.")

# Rodapé do sistema
st.markdown("---")
st.markdown("*Sistema desenvolvido com tecnologia LangChain + Streamlit | Automação Inteligente de Processos Corporativos*")