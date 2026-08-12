import pandas as pd
from perceptron import Perceptron

dados = pd.read_csv("dados/massa_saude_100.csv")
   
X = dados[
     [
        "glicemia_relativa",
        "imc_relativo",
        "historico_familiar"
     ]
   ].values
   
y = dados["prioritario"].values

model = Perceptron(lr=1.0, epochs=19)

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
        print("-> Sucesso! O Perceptron aprendeu perfeitamente a lógica de triagem medica.")
else:
        print("-> Atenção: O modelo não convergiu completamente. Tente aumentar o número de épocas.")