#import math
import torch
import torch.nn as nn
import pandas as pd
#import torchvision
import torchvision.transforms as transforms
from dataset import datasetTest
#from sklearn.metrics import mean_squared_error
#Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data path
folder_path = 'C:/Users/USER/Desktop/minjung_last'
checkpoint_model = 'C:/Users/USER/Desktop/minjung_last/save/model-8420.pth'

# Hyper-parameters 
input_size = 4
batch_size = 4096*4


# Test CSV dataset
test_dataset = datasetTest(folder_path)
# Data loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
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
    def __init__(self,input_size):
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

#model = NeuralNet(input_size)
model = NeuralNet(input_size)
model.load_state_dict(torch.load(checkpoint_model, map_location ='cpu')) #trained model load

# Test the model
# In test phase, we don't need to compute gradients
with torch.no_grad():
    quarter=int(input("분기 : "))
    area=input("상권 : ")
    detail_area=int(input("상세상권 : "))
    sector=input("업종 : ")

    quarter = (quarter-1)/(4-1)

    if area=='A':
        area=1
    elif area=='D':
        area=2
    elif area=='R':
        area=3
    elif area=='U':
        area=4

    area=(area-1)/(4-1)
    
    detail_area = (detail_area-1000001)/(1001496-1000001)
   
    sector = sector.strip("CS") #서비스 코드 앞 CS제거
    sector =pd.to_numeric(sector) #str -> int

    if sector>=100001:
        if sector<=100010:
            sector=sector-100000

    if sector>=200001:
        if sector<=300000:
            sector=sector-200000+10

    if sector>=300001:
        if sector<=400000:
            sector=sector-300000+55

    sector = (sector-1)/(98-1)

    datas = torch.tensor([[quarter, area, detail_area, sector]], dtype=torch.float32)
    #print("스케일링 후 : ",datas[0],datas[1],datas[2],datas[3] )

    outputs = model(datas[0:4])

        # np_targets = targets.cpu().numpy()
    np_outputs = outputs.cpu().numpy()
            
        # targets_weekly = np_targets.T[0]
    outputs_weekly = np_outputs.T[0]*10000000 #예측한 주중 매출
        # targets_weekeed = np_targets.T[1]
    outputs_weekeed = np_outputs.T[1]*10000000 #예측한 주말 매출

        
    print('예상 주중매출: ', outputs_weekly)
    print('예상 주말매출: ', outputs_weekeed)