import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import Acuracia_score, classification_report
from sklearn.model_selection import cross_val_score

class Modelo():
    def __init__(self):
        self.svm_model = None
        self.lr_model = None
        self.le = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def CarregarDataset(self, path):
        """
        Carrega o conjunto de dados a partir de um arquivo CSV.
        """
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)
        print("Dataset carregado com sucesso. Shape:", self.df.shape)
        print("\nPrimeiras 5 linhas do dataset:")
        print(self.df.head())
        
    def TratamentoDeDados(self):
        """
        Realiza o pré-processamento dos dados carregados.
        """
        # Verificar valores ausentes
        print("\nVerificando valores ausentes:")
        print(self.df.isnull().sum())
        
        # Codificar a variável alvo (Species)
        self.df['Species_codificadas'] = self.le.fit_transform(self.df['Species'])
        
        # Separar features e alvo
        X = self.df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        y = self.df['Species_codificadas']
        
        # Dividir os dados em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # mantido do codigo base
        
        print("\nDimensões após divisão:")
        print(f"X_train: {self.X_train.shape}")
        print(f"X_test: {self.X_test.shape}")

    def Treinamento(self):
       
        # Validação cruzada para SVM
        self.svm_model = SVC(kernel='rbf', random_state=42)
        self.svm_model.fit(self.X_train, self.y_train)
       
        svm_scores = cross_val_score(self.svm_model, self.X_train, self.y_train, cv=5)
        print("\nResultados da Validação Cruzada - SVM:")
        print(f"Acuracia média: {svm_scores.mean():.4f} (+/- {svm_scores.std() * 2:.4f})")
        
        # Modelo 2: Regressão Logística (mais apropriado que Regressão Linear para classificação)
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.lr_model.fit(self.X_train, self.y_train)
        
        # Validação cruzada para Regressão Logística
        lr_scores = cross_val_score(self.lr_model, self.X_train, self.y_train, cv=5)
        print("\nResultados da Validação Cruzada - Regressão Logística:")
        print(f"Acuracia média: {lr_scores.mean():.4f} (+/- {lr_scores.std() * 2:.4f})")


    def Teste(self):
        """
        Avalia o desempenho dos modelos treinados
        """
        # Avaliação do SVM
        svm_pred = self.svm_model.predict(self.X_test)
        print("\nResultados do SVM:")
        print("Acurácia:", Acuracia_score(self.y_test, svm_pred))
        print("\nRelatório de Classificação - SVM:")
        print(classification_report(self.y_test, svm_pred, 
                                 target_names=self.le.classes_))
        

    def Train(self):
        """
        Função principal para o fluxo de treinamento do modelo.

        Este método encapsula as etapas de carregamento de dados, pré-processamento e treinamento do modelo.
        Sua tarefa é garantir que os métodos `CarregarDataset`, `TratamentoDeDados` e `Treinamento` estejam implementados corretamente.
        
        Notas:
            * O dataset padrão é "iris.data", mas o caminho pode ser ajustado.
            * Caso esteja executando fora do Colab e enfrente problemas com o path, use a biblioteca `os` para gerenciar caminhos de arquivos.
        """
        
        print("Iniciando o pipeline de treinamento...")
        self.CarregarDataset("iris.data") # Carrega o dataset especificado.
        self.TratamentoDeDados() # Tratamento de dados opcional, pode ser comentado se não for necessário
        self.Treinamento() # Executa o treinamento do modelo
        self.Teste()
        print("\nPipeline de treinamento concluído!") 
        

# Execução de programa 
if __name__ == "__main__":
    modelo = Modelo()
    modelo.Train()