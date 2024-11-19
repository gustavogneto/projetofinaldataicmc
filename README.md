# Projeto final datai ICMC



**Resultados do treinamento inicial** 

Iniciando o pipeline de treinamento...
Dataset carregado com sucesso. Shape: (150, 5)

Primeiras 5 linhas do dataset:
   SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0            5.1           3.5            1.4           0.2  Iris-setosa
1            4.9           3.0            1.4           0.2  Iris-setosa
2            4.7           3.2            1.3           0.2  Iris-setosa
3            4.6           3.1            1.5           0.2  Iris-setosa
4            5.0           3.6            1.4           0.2  Iris-setosa

Verificando valores ausentes:
SepalLengthCm    0
SepalWidthCm     0
PetalLengthCm    0
PetalWidthCm     0
Species          0
dtype: int64

Dimensões após divisão:
X_train: (120, 4)
X_test: (30, 4)

Resultados da Validação Cruzada - SVM:
Accuracy média: 0.9500 (+/- 0.1225)

Resultados do SVM:
Acurácia: 1.0

Relatório de Classificação - SVM:
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      1.00      1.00         9
 Iris-virginica       1.00      1.00      1.00        11

       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg       1.00      1.00      1.00        30


Pipeline de treinamento concluído!