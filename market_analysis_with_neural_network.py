import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import datasetTrain, datasetTest
from sklearn.metrics import mean_squared_error
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data path
folder_path = 'C:/Users/USER/Desktop/minjung_last'
checkpoint_path = 'C/Users/USER/Desktop/minjung_last/save/model-8420.pth'


# Hyper-parameters 
input_size = 4
num_epochs = 50000
batch_size = 4096*5
start_save_epoch = 100

learning_rate = 0.001



# Train CSV dataset
train_dataset = datasetTrain(folder_path)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
# Test CSV dataset
test_dataset = datasetTest(folder_path)
# Data loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    '''
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048, bias=True) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 2, bias=True)  
    
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    '''
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024, bias=True) 
        self.fc2 = nn.Linear(1024, 2048, bias=True) 
        self.fc3 = nn.Linear(2048, 4096, bias=True)
        self.fc4_1 = nn.Linear(4096, 2048, bias=True)
        self.fc4_2 = nn.Linear(4096, 2048, bias=True)
        self.fc5_1 = nn.Linear(2048, 1, bias=True)
        self.fc5_2 = nn.Linear(2048, 1, bias=True)
        self.relu = nn.ReLU()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4_1.weight)
        torch.nn.init.xavier_uniform_(self.fc4_2.weight)
        torch.nn.init.xavier_uniform_(self.fc5_1.weight)
        torch.nn.init.xavier_uniform_(self.fc5_2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out_1 = self.fc4_1(out)
        out_1 = self.relu(out_1)
        out_1 = self.fc5_1(out_1)

        out_2 = self.fc4_2(out)
        out_2 = self.relu(out_2)
        out_2 = self.fc5_2(out_2)

        out = torch.cat([out_1, out_2], dim = 1)

        return out


model = NeuralNet(input_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-3)  

# Train the model

current_loss = 0
total_step = len(train_loader)
for epoch in range(num_epochs):
    total_loss = 0
    for i, (datas, targets) in enumerate(train_loader):  
        # Move tensors to the configured device
        datas = datas.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(datas)
        loss = criterion(outputs, targets)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        print ('[train]Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    # In test phase, we don't need to compute gradients
    with torch.no_grad():
        total_weekly_diff = 0
        total_weekend_diff = 0
        target_size = 0

        for datas, targets in test_loader:
            datas = datas.to(device)
            targets = targets.to(device)
            outputs = model(datas)

            np_targets = targets.cpu().numpy()
            np_outputs = outputs.cpu().numpy()
            
            targets_weekly = np_targets.T[0]
            outputs_weekly = np_outputs.T[0]
            targets_weekeed = np_targets.T[1]
            outputs_weekeed = np_outputs.T[1]

            weekly_diff = mean_squared_error(targets_weekly, outputs_weekly)
            weekend_diff =  mean_squared_error(targets_weekeed, outputs_weekeed)
            target_size += targets.size(0)
            total_weekly_diff += weekly_diff
            total_weekend_diff += weekend_diff

        print('one of the real targets value(targets_weekly)',targets_weekly[0])
        print('one of the real outputs value(outputs_weekly)',outputs_weekly[0])
        print('weekly difference of the network on the dataset: {:.8f}'.format(total_weekly_diff / target_size))
        print('weekend difference of the network on the dataset: {:.8f}'.format(total_weekend_diff / target_size))
        print('total training loss: {:.8f}'.format(total_loss / len(train_loader)))
        # Save the model checkpoint
        if epoch == 0:
            current_loss = total_loss / len(train_loader)

        if (epoch > start_save_epoch) and ((total_loss / len(train_loader)) < current_loss):
            current_loss = total_loss / len(train_loader)
            model_name = checkpoint_path + '/model-' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_name)


