# Rede Neural do Zero — Classificação do Dataset Iris

Implementação de uma **Rede Neural Artificial com uma camada escondida**, construída do zero com **NumPy**, para classificar as espécies de flores do conjunto de dados **Iris**.

> Atividade da disciplina de **Inteligência Artificial** — Ciência da Computação  
> **UNIPAC Barbacena** · Prof. Felipe Roncalli de Paula Carneiro

---

## Sobre o projeto

O **Scikit-Learn** é usado *estritamente* para carregar e preparar os dados. Toda a rede neural (forward pass, função de perda, backpropagation e gradiente descendente) é implementada manualmente com **NumPy**.

### Arquitetura

| Camada    | Neurônios | Ativação |
|-----------|-----------|----------|
| Entrada   | 4         | —        |
| Escondida | 8         | ReLU     |
| Saída     | 3         | Softmax  |

```
Entrada (4) ──W1,b1──► ReLU (8) ──W2,b2──► Softmax (3)
```

### Etapas implementadas

1. **Pré-processamento**
   - Carregamento com `load_iris`
   - Divisão em treino (80%) e teste (20%)
   - Padronização (média 0, desvio padrão 1)
   - One-Hot Encoding dos rótulos
2. **Rede neural**
   - Pesos (`W1`, `W2`) inicializados aleatoriamente e vieses (`b1`, `b2`)
   - ReLU e sua derivada; Softmax na saída
3. **Treinamento**
   - Forward pass
   - Cross-Entropy Loss
   - Backpropagation (regra da cadeia)
   - Atualização por Gradiente Descendente
4. **Avaliação**
   - Inferência no conjunto de teste e cálculo da acurácia

---

## Fundamentos matemáticos

**Forward pass**

```
Z1 = X·W1 + b1        A1 = ReLU(Z1)
Z2 = A1·W2 + b2       A2 = Softmax(Z2)
```

**Cross-Entropy Loss**

```
L = -(1/m) · Σ Σ  y · log(ŷ)
```

**Backpropagation** (a combinação Softmax + Cross-Entropy simplifica o gradiente)

```
dZ2 = (A2 - Y) / m
dW2 = A1ᵀ · dZ2          db2 = Σ dZ2
dZ1 = (dZ2 · W2ᵀ) ⊙ ReLU'(Z1)
dW1 = Xᵀ · dZ1           db1 = Σ dZ1
```

**Atualização dos pesos**

```
θ = θ - α · ∇θ
```

---

## Como executar

### Pré-requisitos
- Python 3.9 ou superior

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/rede-neural-iris.git
cd rede-neural-iris

# 2. (Opcional) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o script
python rede_neural_iris.py
```

---

## Exemplo de saída

```
Treino: 120 amostras | Teste: 30 amostras

Época    1/1000 | Loss: 1.4316 | Acurácia treino: 41.67%
Época  100/1000 | Loss: 0.2668 | Acurácia treino: 92.50%
...
Época 1000/1000 | Loss: 0.0495 | Acurácia treino: 97.50%

==================================================
Loss no teste:     0.0799
Acurácia no teste: 96.67%
==================================================
```

---

## Hiperparâmetros

Ajustáveis no topo do arquivo `rede_neural_iris.py`:

| Parâmetro          | Valor padrão |
|--------------------|--------------|
| `EPOCAS`           | 1000         |
| `TAXA_APRENDIZADO` | 0.1          |
| `N_ESCONDIDOS`     | 8            |
| `SEMENTE`          | 42           |

---

## 📝 Decisões de implementação

- **Padronização sem vazamento de dados:** o `StandardScaler` é ajustado apenas no conjunto de treino e depois aplicado ao teste.
- **Divisão estratificada:** mantém a proporção das 3 classes em treino e teste.
- **Inicialização He:** pesos aleatórios escalados por `√(2/n)`, adequada para ReLU. Os vieses começam em zero.
- **Estabilidade numérica:** a Softmax subtrai o valor máximo de cada linha e a Cross-Entropy soma um `epsilon` dentro do `log`.

---

## Estrutura do repositório

```
rede-neural-iris/
├── rede_neural_iris.py   # Script principal
├── requirements.txt      # Dependências
├── .gitignore
└── README.md
```

## Tecnologias

- [Python](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [Scikit-Learn](https://scikit-learn.org/) (apenas para dados)

---

## Autor

**Nome:** Lucas de Souza Carmo
**Curso:** Ciência da Computação — UNIPAC Barbacena
