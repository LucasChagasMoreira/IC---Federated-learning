import pandas as pd
import torch
from torch import nn
from torch import optim
import os
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def carregar_dadosv2(train_data, test_data, test_size=0.2, random_state=42):
    # Load data from CSV file
    scaler = StandardScaler()

    train_data = pd.read_csv(train_data)
    val_data = pd.read_csv(test_data)

    # Split data into features (X) and targets (y) ("hysteresis")
    X = train_data.drop(['hysteresis', 'joule'], axis=1)
    y = train_data[['hysteresis', 'joule']]

    # Perform train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_test = val_data.drop(['hysteresis', 'joule'], axis=1)
    y_test = val_data[['hysteresis', 'joule']]

    X_train = scaler.fit_transform(X_train.values)
    X_val = scaler.transform(X_val.values)
    X_test = scaler.transform(X_test.values)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)


    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor

import matplotlib.pyplot as plt
import torch

def plot_comparacao(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    plt.figure(figsize=(8, 6))

    plt.scatter(y_true, y_pred, color='blue', alpha=0.5)

    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Linha de Referência')

    plt.xlabel('Valores previstos')
    plt.ylabel('valores Reais')
    plt.title('Comparação entre Valores Reais e Valores Previstos')
    plt.legend()


    plt.show()


class TwentyLayerNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(TwentyLayerNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Primeira camada
        self.layers.append(nn.Linear(input_size, 30))
        self.layers.append(nn.BatchNorm1d(30))

        # 19 camadas internas
        for _ in range(19):
            self.layers.append(nn.Linear(30, 30))
            self.layers.append(nn.BatchNorm1d(30))

        # Camada de saída
        self.layers.append(nn.Linear(30, output_size))

        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], nn.Linear):
                x = self.layers[i](x)
            else:
                x = self.relu(self.layers[i](x))
        x = self.layers[-1](x)  # Camada de saída sem ReLU
        return x

from torch import tensor
from torchmetrics.regression import MeanAbsolutePercentageError
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau



def load_model(filename, model_class, input_size, output_size):
    print("Carregando modelo...")
    checkpoint = torch.load(filename)

    model = model_class(input_size, output_size)

    model.load_state_dict(checkpoint['state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer

def save_model(state,filename="Saved_Model.pth"):
  print("salvando modelo...")
  torch.save(state,filename)

def train(model,num_epochs,optimizer,train_set,target):
  criterion = nn.MSELoss()
  scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
  for epoch in range(num_epochs):
      optimizer.zero_grad()
      outputs = model(train_set)
      loss = criterion(outputs, target)
      loss.backward()
      optimizer.step()
      if (epoch+1) % 100 == 0:
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate(model,val_set,val_target):
    criterion = nn.MSELoss()
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for validation
        val_outputs = model(val_set)
        val_loss = criterion(val_outputs, val_target)
        print(f'Val_Loss: {val_loss}')


def test(model,data):
  model.eval()
  criterion_MSE = nn.MSELoss()
  criterion_MAE = nn.L1Loss()
  criterion_MAPE = MeanAbsolutePercentageError()
  with torch.no_grad():
      predicted = model(data[0])
      test_loss_MSE = criterion_MSE(predicted, data[1])
      test_loss_MAE = criterion_MAE(predicted, data[1])
      test_loss_MAPE = criterion_MAPE(predicted, data[1])
      print(f'MSE Test Loss: {test_loss_MSE.item():.4f}')
      print(f'MAE Test Loss: {test_loss_MAE.item():.4f}')
      print(f'MAPE Test Loss: {test_loss_MAPE.item():.4f}')
  return predicted

folder_path = '..\FederatedLearning-main\Data\IPMSM_datasets\dataset_for_iron_losses_of_IPMSMs\V'
os.chdir(folder_path)

X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = carregar_dadosv2('dados_de_treino.csv','dados_de_teste.csv')


X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

histeresis_train_tensor = [pair[0].item() for pair in y_train_tensor]
histeresis_train_tensor = torch.tensor(histeresis_train_tensor, dtype=torch.float32).to(device)
histeresis_train_tensor = histeresis_train_tensor.unsqueeze(1)

joule_train_tensor = [pair[1].item() for pair in y_train_tensor]
joule_train_tensor = torch.tensor(joule_train_tensor, dtype=torch.float32).to(device)
joule_train_tensor = joule_train_tensor.unsqueeze(1)

histeresis_val_tensor = [pair[0].item() for pair in y_val_tensor]
histeresis_val_tensor = torch.tensor(histeresis_val_tensor, dtype=torch.float32).to(device)
histeresis_val_tensor = histeresis_val_tensor.unsqueeze(1)

joule_val_tensor = [pair[1].item() for pair in y_val_tensor]
joule_val_tensor = torch.tensor(joule_val_tensor, dtype=torch.float32).to(device)
joule_val_tensor = joule_val_tensor.unsqueeze(1)

histeresis_test_tensor = [pair[0].item() for pair in y_test_tensor]
histeresis_test_tensor = torch.tensor(histeresis_test_tensor, dtype=torch.float32).to(device)
histeresis_test_tensor = histeresis_test_tensor.unsqueeze(1)

joule_test_tensor = [pair[1].item() for pair in y_test_tensor]
joule_test_tensor = torch.tensor(joule_test_tensor, dtype=torch.float32).to(device)
joule_test_tensor = joule_test_tensor.unsqueeze(1)

input_size = X_train_tensor.shape[1]
print(input_size)

modelV_hysteresis = TwentyLayerNetwork(input_size, 1)
modelV_joule = TwentyLayerNetwork(input_size, 1)
modelV_total = TwentyLayerNetwork(input_size, 2)

optimizer_h = optim.Adam(modelV_hysteresis.parameters(), lr=0.01)
optimizer_j = optim.Adam(modelV_joule.parameters(), lr=0.01)
optimizer_t = optim.Adam(modelV_total.parameters(), lr=0.01)

train(modelV_hysteresis,5000,optimizer_h,X_train_tensor, histeresis_train_tensor)
evaluate(modelV_hysteresis,X_val_tensor,histeresis_val_tensor)

train(modelV_joule,5000,optimizer_j,X_train_tensor, joule_train_tensor)
evaluate(modelV_joule,X_val_tensor,joule_val_tensor)

train(modelV_total,5000,optimizer_t,X_train_tensor, y_train_tensor)
evaluate(modelV_total,X_val_tensor,y_val_tensor)

previstos_h = test(modelV_hysteresis,[X_test_tensor,histeresis_test_tensor])

previstos_j = test(modelV_joule,[X_test_tensor,joule_test_tensor])

previstos_t = test(modelV_total,[X_test_tensor,y_test_tensor])

plot_comparacao(histeresis_test_tensor,previstos_h)

plot_comparacao(joule_test_tensor,previstos_j)

plot_comparacao(y_test_tensor,previstos_t)

mse_h = mean_squared_error(histeresis_test_tensor, previstos_h)
mae_h = mean_absolute_error(histeresis_test_tensor, previstos_h)
histeresis_test_np = histeresis_test_tensor.numpy()
previstos_np_h = previstos_h.numpy()
mape_h = np.mean(np.abs((histeresis_test_np - previstos_np_h) / histeresis_test_np)) * 100

mse_j = mean_squared_error(joule_test_tensor, previstos_j)
mae_j = mean_absolute_error(joule_test_tensor, previstos_j)
joule_test_np = joule_test_tensor.numpy()
previstos_np_j = previstos_j.numpy()
mape_j = np.mean(np.abs((joule_test_np - previstos_np_j) / joule_test_np)) * 100

mse_t = mean_squared_error(y_test_tensor, previstos_t)
mae_t = mean_absolute_error(y_test_tensor, previstos_t)
total_test_np = y_test_tensor.numpy()
previstos_np_t = previstos_t.numpy()
mape_t = np.mean(np.abs((total_test_np - previstos_np_t) / total_test_np)) * 100

print(f'V histeresis model:')
print(f'Test MSE: {mse_h}')
print(f'Test MAE: {mae_h}')
print(f'Test MAPE: {mape_h}%')

print(f'V joule model:')
print(f'Test MSE: {mse_j}')
print(f'Test MAE: {mae_j}')
print(f'Test MAPE: {mape_j}%')

print(f'V total model:')
print(f'Test MSE: {mse_t}')
print(f'Test MAE: {mae_t}')
print(f'Test MAPE: {mape_t}%')