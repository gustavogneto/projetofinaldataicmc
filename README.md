# Projeto de Classificação de Flores Iris

## 1. Objetivo
O objetivo principal deste trabalho é projetar e analisar modelos de aprendizado de máquina para identificar corretamente espécies de flores Iris com base nas características observadas da sépala e da pétala. O projeto envolve:
- Pré-processamento dos dados
- Experimentação com diferentes métodos de classificação
- Avaliação do desempenho dos modelos gerados

## 2. Dataset
O dataset Iris é um dos conjuntos de dados mais conhecidos em machine learning e estatística, coletado por Ronald A. Fisher em 1936. Contém informações sobre três espécies de flores:
- Iris setosa
- Iris versicolor
- Iris virginica

### Estrutura do Dataset
- **Total de amostras**: 150 (50 por espécie)
- **Características** (medidas em centímetros):
  - SepalLengthCm: Comprimento da sépala
  - SepalWidthCm: Largura da sépala
  - PetalLengthCm: Comprimento da pétala
  - PetalWidthCm: Largura da pétala
- **Coluna alvo**: Species (espécie da flor)

## 3. Estrutura do Projeto

### Arquivos Principais
1. `main.py`: Código base do projeto que deve ser completado
2. `iris.data`: Dataset em formato CSV

### Classe Modelo
A classe `Modelo` no `main.py` contém os seguintes métodos para implementação:

#### CarregarDataset
- Função para carregar o dataset Iris
- Utiliza o parâmetro `path` para especificar o caminho do arquivo

#### TratamentoDeDados
Responsável pelo pré-processamento dos dados:
- Visualização inicial com `self.df.head()`
- Verificação e tratamento de dados faltantes
- Análise e seleção de características relevantes

#### Treinamento
Implementa o treinamento do modelo:
- Divisão dos dados em conjuntos de treino e teste
- Implementação de diferentes modelos (ex: SVM, Regressão Linear)
- Possibilidade de usar técnicas como validação cruzada

#### Teste
Responsável pela avaliação do modelo treinado

### Requisitos de Implementação
1. **Múltiplos Modelos**
   - Implementação de no mínimo dois modelos diferentes
   - Comparação entre os resultados obtidos

2. **Análise e Documentação**
   - Análise da acurácia de cada modelo
   - Documentação das diferenças entre os modelos

3. **Avaliação**
   - Utilização do método `Teste` para avaliação de desempenho
   - Comparação dos resultados entre diferentes modelos

# Visão geral e documentação basica

## Instalações necessárias

```bash

pip install pandas
pip install scikit-learn

```

## imports necessários

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

```

1. **Carregamento de Dados**:
   - Implementei o carregamento do dataset com feedback visual
   - Adicionei exibição das primeiras linhas para verificação

2. **Tratamento de Dados**:
   - Adicionei verificação de valores ausentes
   - Implementei codificação da variável target usando LabelEncoder
   - Separei features e target
   - Dividi os dados em conjuntos de treino e teste (80/20)

3. **Treinamento**:
   - Implementei dois modelos diferentes:
     - SVM (Support Vector Machine)
     - Regressão Logística (substitui a Regressão Linear por ser mais apropriada para classificação)
   - Adicionei validação cruzada (cross-validation) para ambos os modelos
   - Incluí feedback sobre o desempenho durante o treinamento

4. **Teste**:
   - Implementei avaliação completa dos modelos
   - Adicionei métricas de performance (acurácia e relatório de classificação)
   - Incluí comparação entre os dois modelos

5. **Melhorias Gerais**:
   - Adicionei docstrings detalhadas
   - Implementei feedback em cada etapa do processo
   - Incluí tratamento de dados mais robusto
   - Adicionei verificações e prints informativos

Para usar o código, basta instanciar a classe e chamar o método Train():

```python
modelo = Modelo()
modelo.Train()
```

Este código atende a todos os requisitos especificados no PDF:
- Implementa dois modelos diferentes
- Inclui pré-processamento adequado dos dados
- Possui métodos de avaliação e comparação
- Está bem documentado e organizado



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

# testes realizando Regressão Logistica

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
Acuracia média: 0.9500 (+/- 0.1225)

Resultados da Validação Cruzada - Regressão Logística:
Acuracia média: 0.9667 (+/- 0.0972)

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


Resultados da Regressão Logística:
Acurácia: 1.0

Relatório de Classificação - Regressão Logística:
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      1.00      1.00         9
 Iris-virginica       1.00      1.00      1.00        11

       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg       1.00      1.00      1.00        30


Pipeline de treinamento concluído!