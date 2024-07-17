import pandas as pd
import torch
from torch import nn
from torch import optim
import os
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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

class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RegressionModel, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim if _ == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
from torch import tensor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna


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


def create_train_model(input_dim, output_dim, X_train, y_train, X_val, y_val):
    def train_model(trial):

        hidden_dim = trial.suggest_int('hidden_dim', 16, 64)
        num_layers = trial.suggest_int('num_layers', 10, 40)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 2e-4, log=True)
        batch_size = trial.suggest_int('batch_size', 512, 512)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2)

        model = RegressionModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

        X_train_device = X_train.to(device)
        y_train_device = y_train.to(device)
        X_val_device = X_val.to(device)
        y_val_device = y_val.to(device)

        model.train()
        for epoch in range(200):
            permutation = torch.randperm(X_train.size()[0]).to(device)
            for i in range(0, X_train.size()[0], batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = mean_squared_error(y_val.cpu().numpy(), val_outputs.cpu().numpy())


        return val_loss

    return train_model

def evaluate(model,val_set,val_target):
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_set)
        val_loss = criterion(val_outputs, val_target)
        print(f'Val_Loss: {val_loss}')


def test(model,data):
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = [d.to(device) for d in data]
        predicted = model(data[0]).to(device)
    return predicted

folder_path = '/media/work/lucasmoreira/IC---Federated-learning/FederatedLearning-main/Data/IPMSM_datasets/dataset_for_iron_losses_of_IPMSMs/V'
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

study1 = optuna.create_study(direction='minimize')
train_model = create_train_model(input_size,1,X_train_tensor, histeresis_train_tensor,X_val_tensor,histeresis_val_tensor)
study1.optimize(train_model, n_trials=500)

trial = study1.best_trial

best_model_h = trial.params

best_model = RegressionModel(X_train_tensor.shape[1], best_model_h['hidden_dim'], 1, best_model_h['num_layers']).to(device)

optimizer = optim.Adam(best_model.parameters(), lr=best_model_h['learning_rate'])

for name, param in best_model.named_parameters():
    print(f"{name}: {param.size()}")

print(best_model)

def fine_tune(params,model,num_epochs,optimizer,train_set,target,val_set,val_target):
  criterion = nn.MSELoss()
  model.train()
  for epoch in range(num_epochs):
      permutation = torch.randperm(train_set.size()[0]).to(device)
      for i in range(0, train_set.size()[0], params['batch_size']):
          optimizer.zero_grad()
          indices = permutation[i:i + params['batch_size']]
          batch_x, batch_y = train_set[indices], target[indices]

          outputs = model(batch_x)
          loss = criterion(outputs, batch_y)
          loss.backward()
          optimizer.step()

  model.eval()
  with torch.no_grad():
      val_outputs = model(val_set)
      val_loss = mean_squared_error(val_target.cpu().numpy(), val_outputs.cpu().numpy())

fine_tune(best_model_h,best_model,5000,optimizer,X_train_tensor,histeresis_train_tensor,X_val_tensor,histeresis_val_tensor)
previstos_h = test(best_model.to(device), [X_test_tensor.to(device), histeresis_test_tensor.to(device)])
print(previstos_h)

print('Best trial:')
trial = study1.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

input_size = X_train_tensor.shape[1]
print(input_size)

study2 = optuna.create_study(direction='minimize')
train_model = create_train_model(input_size,1,X_train_tensor, joule_train_tensor,X_val_tensor,joule_val_tensor)
study2.optimize(train_model, n_trials=20)

trial = study2.best_trial

best_model_j = trial.params

best_model = RegressionModel(X_train_tensor.shape[1], best_model_j['hidden_dim'], 1, best_model_j['num_layers'])
criterion = nn.MSELoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_model_j['learning_rate'])

previstos_j = test(best_model,[X_test_tensor,joule_test_tensor])

for name, param in best_model.named_parameters():
    print(f"{name}: {param.size()}")

print(best_model)

fine_tune(best_model_j, best_model,5000,optimizer,X_train_tensor,joule_train_tensor,X_val_tensor,joule_val_tensor)
previstos_j = test(best_model, [X_test_tensor, joule_test_tensor])
print(previstos_j)

print('Best trial:')
trial = study2.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

input_size = X_train_tensor.shape[1]
print(input_size)

study3 = optuna.create_study(direction='minimize')
train_model = create_train_model(input_size,2,X_train_tensor, y_train_tensor,X_val_tensor,y_val_tensor)
study3.optimize(train_model, n_trials=20)

trial = study3.best_trial

best_model_t = trial.params

best_model = RegressionModel(X_train_tensor.shape[1], best_model_t['hidden_dim'], 2, best_model_t['num_layers'])
criterion = nn.MSELoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_model_t['learning_rate'])

previstos_t = test(best_model,[X_test_tensor,y_test_tensor])

for name, param in best_model.named_parameters():
    print(f"{name}: {param.size()}")

print(best_model)

fine_tune(best_model_t,best_model,5000,optimizer,X_train_tensor,y_train_tensor,X_val_tensor,y_val_tensor)
previstos_t = test(best_model, [X_test_tensor, y_test_tensor])
print(previstos_t)

print('Best trial:')
trial = study3.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

mse_h = mean_squared_error(histeresis_test_tensor.cpu(), previstos_h.cpu())
mae_h = mean_absolute_error(histeresis_test_tensor.cpu(), previstos_h.cpu())
histeresis_test_np = histeresis_test_tensor.cpu().numpy()
previstos_np_h = previstos_h.cpu().numpy()
mape_h = np.mean(np.abs((histeresis_test_np - previstos_np_h) / histeresis_test_np)) * 100

mse_j = mean_squared_error(joule_test_tensor.cpu(), previstos_j.cpu())
mae_j = mean_absolute_error(joule_test_tensor.cpu(), previstos_j.cpu())
joule_test_np = joule_test_tensor.cpu().numpy()
previstos_np_j = previstos_j.cpu().numpy()
mape_j = np.mean(np.abs((joule_test_np - previstos_np_j) / joule_test_np)) * 100

mse_t = mean_squared_error(y_test_tensor.cpu(), previstos_t.cpu())
mae_t = mean_absolute_error(y_test_tensor.cpu(), previstos_t.cpu())
total_test_np = y_test_tensor.cpu().numpy()
previstos_np_t = previstos_t.cpu().numpy()
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