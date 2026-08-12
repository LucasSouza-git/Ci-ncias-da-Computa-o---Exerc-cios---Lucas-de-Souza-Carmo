import numpy as np
import pandas as pd

class Perceptron:
    """
    Implementação do modelo Perceptron de camada única do zero usando NumPy.
    
    Parâmetros:
    -----------
    lr : float
        Taxa de aprendizado (learning rate), controla a intensidade do ajuste dos pesos.
    epochs : int
        Número de passagens completas pelo conjunto de dados de treinamento.
    """
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Treina o Perceptron ajustando os pesos e o bias iterativamente.
        
        Parâmetros:
        -----------
        X : np.ndarray de forma (n_amostras, n_atributos)
            Vetor de características de entrada.
        y : np.ndarray de forma (n_amostras,)
            Rótulos ou saídas desejadas (0 ou 1).
        """
        n_samples, n_features = X.shape
        
        # Inicializa os pesos e o bias com zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        print("Iniciando o treinamento do Perceptron...\n")

        for epoch in range(self.epochs):
            errors_in_epoch = 0
            
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                # Erro = Valor real - Valor previsto
                error = target - prediction
                
                # Regra de atualização dos Pesos e Bias: Δw = η * (y - ŷ) * x
                update = self.lr * error
                self.weights += update * xi
                self.bias += update
                
                if error != 0:
                    errors_in_epoch += 1
                    
            print(f"Época {epoch + 1:02d}/{self.epochs:02d} | Erros: {errors_in_epoch} | Pesos: {self.weights} | Bias: {self.bias:.2f}")

    def predict(self, X):
        """
        Calcula a previsão binária para as entradas fornecidas.
        
        Combinação Linear: z = (W · X) + b
        Função de Ativação Degrau: ŷ = 1 se z >= 0, senão 0
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 1.0, 1.0, 0)
    