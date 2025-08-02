# demand_prediction.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. GERAÇÃO DE DADOS FICTÍCIOS ---
def generate_fictional_data():
    """
    Gera um dataset fictício de 12 meses para previsão de demanda.
    Retorna um dicionário com os dados.
    """
    print("Gerando dados de demanda fictícios...")
    
    # Gerando os dados
    meses = np.arange(1, 13)
    custo_producao = np.random.randint(100, 200, size=12) * 10
    custo_marketing = np.random.randint(50, 150, size=12) * 10
    
    # A demanda tem uma relação com os custos e um pouco de ruído aleatório
    demanda_vendas = (custo_producao * 0.5 + custo_marketing * 0.8) + np.random.normal(0, 100, size=12)
    demanda_vendas = np.round(np.abs(demanda_vendas)).astype(int)
    
    data = {
        'mes': meses,
        'custo_producao': custo_producao,
        'custo_marketing': custo_marketing,
        'demanda_vendas': demanda_vendas
    }
    
    print("Dados gerados com sucesso!")
    return data

# --- 2. MODELO DE PREVISÃO ---
def train_and_predict_model(data):
    """
    Treina um modelo de regressão e faz previsões.
    Retorna o modelo treinado, dados de teste e a previsão.
    """
    print("\nTreinando o modelo de previsão de demanda...")
    
    # Features (X) e Target (y)
    X = np.vstack((data['custo_producao'], data['custo_marketing'])).T
    y = data['demanda_vendas']
    
    # Divisão em conjuntos de treino e teste
    # Usaremos 70% para treino e 30% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Inicializa e treina o modelo de Regressão Linear
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Fazendo a previsão para o conjunto de teste
    y_pred = model.predict(X_test)
    
    print("Modelo treinado com sucesso!")
    print(f"Coeficiente R²: {r2_score(y_test, y_pred):.2f}")
    
    return model, X_test, y_test, y_pred

# --- 3. ANÁLISE DE CUSTO-BENEFÍCIO ---
def perform_cost_benefit_analysis(model, custo_futuro_prod, custo_futuro_mkt, preco_unitario=25):
    """
    Calcula a demanda prevista e analisa o custo-benefício.
    """
    print("\nRealizando análise de custo-benefício...")
    
    # Previsão da demanda para um mês futuro
    custos_futuros = np.array([[custo_futuro_prod, custo_futuro_mkt]])
    demanda_prevista = model.predict(custos_futuros)[0]
    
    # Garantir que a demanda prevista seja um número inteiro positivo
    demanda_prevista = int(max(0, round(demanda_prevista)))
    
    # Cálculos de receita e lucro
    custo_total = custo_futuro_prod + custo_futuro_mkt
    receita_prevista = demanda_prevista * preco_unitario
    lucro_previsto = receita_prevista - custo_total
    
    print("\n--- Relatório de Otimização (Mês Futuro) ---")
    print(f"Custos de Produção e Marketing: R$ {custo_total:.2f}")
    print(f"Demanda Prevista: {demanda_prevista} unidades")
    print(f"Receita Prevista: R$ {receita_prevista:.2f}")
    print(f"Lucro Previsto: R$ {lucro_previsto:.2f}")
    
    if lucro_previsto > 0:
        print("Recomendação: O plano é financeiramente viável. Avance com os custos propostos.")
    else:
        print("Recomendação: O plano pode não ser viável. Revise os custos ou o preço de venda.")

# --- 4. VISUALIZAÇÃO ---
def plot_results(data, model):
    """
    Plota a demanda real vs. a previsão para todo o período.
    """
    print("\nGerando visualização...")
    
    # Dados reais
    meses = data['mes']
    demanda_real = data['demanda_vendas']
    
    # Previsão para todos os meses (usando o modelo treinado)
    X_completo = np.vstack((data['custo_producao'], data['custo_marketing'])).T
    demanda_prevista_completa = model.predict(X_completo)
    
    plt.figure(figsize=(10, 6))
    plt.plot(meses, demanda_real, marker='o', label='Demanda Real', color='blue')
    plt.plot(meses, demanda_prevista_completa, marker='x', linestyle='--', label='Demanda Prevista', color='red')
    
    plt.title('Demanda de Vendas: Real vs. Prevista')
    plt.xlabel('Mês')
    plt.ylabel('Unidades Vendidas')
    plt.xticks(meses)
    plt.legend()
    plt.grid(True)
    plt.show()
    
# --- FLUXO PRINCIPAL DO SCRIPT ---
if __name__ == "__main__":
    # Gerar os dados
    fictional_data = generate_fictional_data()
    
    # Treinar o modelo
    trained_model, X_test, y_test, y_pred = train_and_predict_model(fictional_data)
    
    # Dados de exemplo para o mês futuro (simulado)
    novo_custo_producao = 1500
    novo_custo_marketing = 1200
    
    # Analisar o custo-benefício
    perform_cost_benefit_analysis(trained_model, novo_custo_producao, novo_custo_marketing)
    
    # Visualizar os resultados
    plot_results(fictional_data, trained_model)
