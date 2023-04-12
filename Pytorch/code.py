import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define the neural network
class HousingNN(nn.Module):
    def __init__(self, input_size):
        super(HousingNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Load and preprocess the dataset
data = pd.read_csv("housing.csv")

# Separate the target column
target = data.pop('median_house_value')

# Handle missing values
data.fillna(data.mean(), inplace=True)

# One-hot encode the 'ocean_proximity' column
encoder = OneHotEncoder(sparse=False)
proximity_encoded = encoder.fit_transform(data['ocean_proximity'].values.reshape(-1, 1))
data_encoded = data.drop(columns=['ocean_proximity'])
data_encoded = pd.concat([data_encoded, pd.DataFrame(proximity_encoded, columns=encoder.get_feature_names_out(['OP']))], axis=1)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(data_encoded)
y = target.values

# Perform k-fold cross-validation
num_epochs = 100
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Instantiate the model, loss function, and optimizer
    model = HousingNN(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        test_targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        print(f'Fold {fold + 1}/{k_folds}, Test Loss: {test_loss.item():.4f}')
torch.save(model.state_dict(), "housing_model.pt")
