"""
Rede Neural Artificial do zero (NumPy) para classificar o dataset Iris.

Arquitetura: 4 entradas -> 8 neurônios (ReLU) -> 3 saídas (Softmax)
Treino: Gradiente Descendente com Cross-Entropy Loss e Backpropagation manual.

Scikit-Learn é usado ESTRITAMENTE para carregar e preparar os dados.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ----------------------------------------------------------------------
# Hiperparâmetros
# ----------------------------------------------------------------------
SEMENTE = 42
N_ENTRADAS = 4
N_ESCONDIDOS = 8
N_SAIDAS = 3
EPOCAS = 1000
TAXA_APRENDIZADO = 0.1


# ----------------------------------------------------------------------
# 1. Pré-processamento dos dados
# ----------------------------------------------------------------------
def carregar_e_preparar_dados():
    """Carrega o Iris, divide (80/20), padroniza e aplica One-Hot nos rótulos."""
    iris = load_iris()
    X, y = iris.data, iris.target  # X: (150, 4) | y: (150,)

    # Divisão 80% treino / 20% teste (estratificada para manter as proporções)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=SEMENTE, stratify=y
    )

    # Padronização (média 0, desvio padrão 1).
    # O scaler é ajustado SOMENTE no treino para evitar vazamento de dados.
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)

    # One-Hot Encoding dos rótulos (vetores binários de tamanho 3)
    encoder = OneHotEncoder(sparse_output=False)
    Y_treino = encoder.fit_transform(y_treino.reshape(-1, 1))
    Y_teste = encoder.transform(y_teste.reshape(-1, 1))

    return X_treino, X_teste, Y_treino, Y_teste


# ----------------------------------------------------------------------
# 2. Funções de ativação e perda
# ----------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def relu_derivada(z):
    return (z > 0).astype(float)


def softmax(z):
    # Subtrai o máximo de cada linha para estabilidade numérica
    z_estavel = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_estavel)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(Y_real, Y_prev):
    """Entropia cruzada média sobre todas as amostras."""
    epsilon = 1e-12  # evita log(0)
    return -np.mean(np.sum(Y_real * np.log(Y_prev + epsilon), axis=1))


# ----------------------------------------------------------------------
# 3. Rede Neural
# ----------------------------------------------------------------------
def inicializar_parametros(rng):
    """Inicializa pesos aleatoriamente (He, adequado para ReLU) e vieses com zeros."""
    return {
        "W1": rng.standard_normal((N_ENTRADAS, N_ESCONDIDOS)) * np.sqrt(2.0 / N_ENTRADAS),
        "b1": np.zeros((1, N_ESCONDIDOS)),
        "W2": rng.standard_normal((N_ESCONDIDOS, N_SAIDAS)) * np.sqrt(2.0 / N_ESCONDIDOS),
        "b2": np.zeros((1, N_SAIDAS)),
    }


def forward(X, p):
    """Forward Pass: entrada -> escondida (ReLU) -> saída (Softmax)."""
    Z1 = X @ p["W1"] + p["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = softmax(Z2)
    return Z1, A1, A2


def backward(X, Y, Z1, A1, A2, p):
    """Backpropagation: gradientes via regra da cadeia."""
    m = X.shape[0]

    # Softmax + Cross-Entropy => gradiente simplificado: (A2 - Y)
    dZ2 = (A2 - Y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Propaga o erro para a camada escondida (aplicando a derivada da ReLU)
    dA1 = dZ2 @ p["W2"].T
    dZ1 = dA1 * relu_derivada(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def atualizar_parametros(p, grads, taxa):
    """Gradiente Descendente: parâmetro = parâmetro - taxa * gradiente."""
    for chave in p:
        p[chave] -= taxa * grads[chave]


def acuracia(Y_real, Y_prev):
    return np.mean(np.argmax(Y_real, axis=1) == np.argmax(Y_prev, axis=1))


# ----------------------------------------------------------------------
# Execução principal
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEMENTE)

    X_treino, X_teste, Y_treino, Y_teste = carregar_e_preparar_dados()
    print(f"Treino: {X_treino.shape[0]} amostras | Teste: {X_teste.shape[0]} amostras\n")

    params = inicializar_parametros(rng)

    # Loop de treinamento
    for epoca in range(1, EPOCAS + 1):
        Z1, A1, A2 = forward(X_treino, params)
        perda = cross_entropy(Y_treino, A2)
        grads = backward(X_treino, Y_treino, Z1, A1, A2, params)
        atualizar_parametros(params, grads, TAXA_APRENDIZADO)

        if epoca == 1 or epoca % 100 == 0:
            acc = acuracia(Y_treino, A2)
            print(f"Época {epoca:4d}/{EPOCAS} | Loss: {perda:.4f} | Acurácia treino: {acc:.2%}")

    # Avaliação no conjunto de teste
    _, _, A2_teste = forward(X_teste, params)
    perda_teste = cross_entropy(Y_teste, A2_teste)
    acc_teste = acuracia(Y_teste, A2_teste)

    print("\n" + "=" * 50)
    print(f"Loss no teste:     {perda_teste:.4f}")
    print(f"Acurácia no teste: {acc_teste:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
