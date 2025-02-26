import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar a rede neural
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(gestures), activation='softmax')  # Saída para 5 classes
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Salvar o modelo treinado
model.save("modelo_gestos.h5")
