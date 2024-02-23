import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

max_sequence_length = 514

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

data = 'Imbd_steeming_and_lemmatization.csv'

def encode_labels(data, label_column):
    # Cria um dicionário de mapeamento para os rótulos
    label_map = {"positive": 1, "negative": 0}

    # Aplica a função de mapeamento para criar uma nova coluna com os valores codificados
    data['encoded_labels'] = data[label_column].map(label_map)

    return data


def tokenize_with_roberta(csv_file, column_name, max_length=128):
    data = pd.read_csv(csv_file)
    data.head()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenized = data[column_name].apply(lambda x: tokenizer.encode(x, max_length=max_length,padding='max_length', truncation=True))
    data['tokens'] = tokenized
    return data


tokenized_data = tokenize_with_roberta(data,'review',max_length = 512)
tokenized_data = encode_labels(tokenized_data,"sentiment")
tokenized_data.head()


# roberto
from sklearn.model_selection import train_test_split
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset

train_data, test_data = train_test_split(tokenized_data, test_size=0.2)

train_inputs = torch.tensor(train_data['tokens'].tolist())
train_labels = torch.tensor(train_data['encoded_labels'].tolist())

test_inputs = torch.tensor(test_data['tokens'].tolist())
test_labels = torch.tensor(test_data['encoded_labels'].tolist())

# Crie datasets do PyTorch
train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)

# Crie dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Carregue o modelo pre-treinado para fine-tuning
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configure o otimizador
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Treinamento
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0  # Variável para armazenar a perda total da época
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  # Adiciona a perda do batch à perda total da época
    epoch_loss /= len(train_dataloader)  # Calcula a média da perda da época
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

# Avaliação no conjunto de teste
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        outputs = model(inputs)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Acurácia no conjunto de teste: {accuracy}")



#Xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Dividindo os dados em features e rótulos
X = np.array(tokenized_data['tokens'].tolist())
y = np.array(tokenized_data['encoded_labels'])  # Substitua 'rotulo' pelo nome da coluna de rótulos

# Dividindo em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertendo para objetos DMatrix do XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Definindo parâmetros do modelo
params = {
    'max_depth': 10,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Treinando o modelo
num_round = 100
model = xgb.train(params, dtrain, num_round)

# Fazendo previsões no conjunto de teste
predictions_proba = model.predict(dtest)
predictions = [1 if pred > 0.5 else 0 for pred in predictions_proba]

# Calculando a acurácia do modelo
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')
