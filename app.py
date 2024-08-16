import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.impute import SimpleImputer
import numpy as np

# Configuração da página do Streamlit
st.set_page_config(page_title="Análise Educacional", layout="wide")

# Título
st.title("Análise e Modelagem Educacional")

# Carregar o novo dataset
@st.cache_data
def load_data():
    return pd.read_csv('PEDE_PASSOS_DATASET_FIAP.csv', delimiter=';')

new_data_df = load_data()

# Exibir as colunas disponíveis no dataset
if st.checkbox("Mostrar colunas do dataset"):
    st.write(new_data_df.columns)

# Criar uma coluna fictícia de comentários
np.random.seed(42)  # Para resultados consistentes

# Comentários fictícios baseados no desempenho
comentarios = [
    "O aluno demonstrou grande progresso este ano, muito participativo e engajado.",
    "Precisa melhorar nas atividades em grupo, mas individualmente vai bem.",
    "Excelente desempenho nas provas, sempre preparado.",
    "Apresenta dificuldades em matemática, mas está melhorando.",
    "Participação abaixo do esperado, precisa se concentrar mais.",
    "Ótimo trabalho nas atividades extracurriculares, sempre dedicado.",
    "Tem dificuldades com o conteúdo, mas é esforçado.",
    "Aluno muito proativo e colaborador nas aulas.",
    "Necessita de acompanhamento adicional para manter o ritmo.",
    "Mostrou grande evolução nas últimas avaliações."
]

# Atribuir comentários fictícios aleatórios aos alunos
new_data_df['COMENTARIOS_ALUNOS'] = np.random.choice(comentarios, size=len(new_data_df))

# Simplificar o DataFrame para incluir apenas as colunas necessárias
simplified_df = new_data_df[['IDADE_ALUNO_2020', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021']]

# Remover linhas com valores nulos na coluna de desempenho (por exemplo, 'IAA_2021')
simplified_df = simplified_df.dropna(subset=['IAA_2021'])

# Visualizar a distribuição de IAA_2021
st.subheader("Distribuição de IAA_2021")
fig_iaa_dist = plt.figure(figsize=(10, 6))
sns.histplot(simplified_df['IAA_2021'], kde=True, color='blue')
plt.title('Distribuição de IAA_2021')
plt.xlabel('IAA_2021')
plt.ylabel('Frequência')
plt.grid(axis='y')
st.pyplot(fig_iaa_dist)

# Boxplot comparando as idades e o desempenho IAA_2021
st.subheader("Boxplot de Idade vs IAA_2021")
fig_boxplot = plt.figure(figsize=(12, 6))
sns.boxplot(x='IDADE_ALUNO_2020', y='IAA_2021', data=simplified_df)
plt.title('Boxplot de Idade vs IAA_2021')
plt.xlabel('Idade do Aluno em 2020')
plt.ylabel('IAA_2021')
plt.grid(axis='y')
st.pyplot(fig_boxplot)

# Realizar uma modelagem preditiva simples usando IEG_2021, IPS_2021 e IDA_2021 para prever IAA_2021
features = simplified_df[['IEG_2021', 'IPS_2021', 'IDA_2021']]
target = simplified_df['IAA_2021']

# Remover as linhas onde todos os valores em features são NaN
features = features.dropna(how='all')

# Remover valores ausentes do target
features, target = features.align(target.dropna(), join='inner', axis=0)

# Imputar valores ausentes com a média
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(features)

# Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X_imputed, target, test_size=0.3, random_state=42)

# Modelo de Regressão Linear
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
lin_reg_predictions = lin_reg_model.predict(X_test)
lin_reg_mse = mean_squared_error(y_test, lin_reg_predictions)

# Modelo de Regressão com Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)

# Exibir os resultados dos modelos
st.subheader("Erro Quadrático Médio dos Modelos")
st.write(f"Mean Squared Error (Linear Regression): {lin_reg_mse}")
st.write(f"Mean Squared Error (Random Forest): {rf_mse}")

# ------------------ Análise de Texto (NLP) ------------------

# Remover valores nulos e espaços em branco da coluna de comentários
new_data_df['COMENTARIOS_ALUNOS'] = new_data_df['COMENTARIOS_ALUNOS'].fillna('').str.strip()
new_data_df = new_data_df[new_data_df['COMENTARIOS_ALUNOS'] != '']  # Remover linhas vazias

# Calcular a polaridade dos comentários (análise de sentimentos)
new_data_df['Sentimento'] = new_data_df['COMENTARIOS_ALUNOS'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Visualizar a distribuição dos sentimentos
st.subheader("Distribuição dos Sentimentos dos Comentários")
fig_sentimentos = plt.figure(figsize=(10, 6))
sns.histplot(new_data_df['Sentimento'], kde=True, color='green')
plt.title('Distribuição dos Sentimentos dos Comentários')
plt.xlabel('Sentimento')
plt.ylabel('Frequência')
plt.grid(axis='y')
st.pyplot(fig_sentimentos)

# Gerar a nuvem de palavras dos comentários
st.subheader("Nuvem de Palavras dos Comentários Fictícios")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(new_data_df['COMENTARIOS_ALUNOS']))
fig_wordcloud = plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig_wordcloud)
