import pandas as pd
import numpy as np
import glob

# Mapeamento dos gestos para valores numéricos
gestures = {
    "cima": 0,
    "baixo": 1,
    "esquerda": 2,
    "direita": 3,
    "parar": 4
}

# Carregar todos os arquivos CSV
data = []
labels = []

for gesto, arquivo in gestures.items():
    df = pd.read_csv(f"gesto_{gesto}.csv")  # Ajuste o nome se necessário
    df = df.dropna()  # Remove linhas vazias
    data.append(df.iloc[:, :-1].values)  # Pega todas as colunas, exceto a label
    labels.append(np.full((df.shape[0],), gestures[gesto]))  # Cria rótulos numéricos

# Converter para numpy arrays
X = np.vstack(data)  # Dados de entrada
y = np.concatenate(labels)  # Rótulos

print(f"Total de amostras: {X.shape[0]}, Dimensão dos dados: {X.shape[1]}")
