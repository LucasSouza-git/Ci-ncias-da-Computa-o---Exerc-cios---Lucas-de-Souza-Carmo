import pandas as pd
from perceptron import Perceptron

dados = pd.read_csv("dados/massa_qualidade_100.csv")
   
X = dados[
     [
        "desvio_diametro_mm",
        "desvio_peso_g",
     ]
   ].values
   
y = dados["peca_aprovada"].values

model = Perceptron(lr=0.1, epochs=19)

model.fit(X, y)

print("=" * 60)
print("DESAFIO PRÁTICO: TREINAMENTO DO PERCEPTRON - ")
print("=" * 60 + "\n")

    # Instancia e treina o Perceptron


print("\n" + "=" * 60)
print("VALIÇÃO E TESTE DAS PREVISÕES")
print("=" * 60)

all_correct = True
for xi, target in zip(X, y):
        pred = model.predict(xi)
        is_correct = (pred == target)
        if not is_correct:
            all_correct = False
        status = "✓ CORRETO" if is_correct else "✗ ERRO"
        print(f"Entrada: {xi} | Saída Esperada: {target} | Previsto: {pred} | Status: {status}")

print("\nResultados finais:")
if all_correct:
        print("-> Sucesso! O Perceptron aprendeu perfeitamente a lógica de controle de qualidade industrial.")
else:
        print("-> Atenção: O modelo não convergiu completamente. Tente aumentar o número de épocas.")