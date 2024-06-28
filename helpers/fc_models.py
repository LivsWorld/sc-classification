import torch.nn as nn

FC_NAMES = ['lin1', 'lin2', 'lin3', 'lin4']
# 1 layer
class Lin1(nn.Module):
    def __init__(self, in_features, output):
        super().__init__()
        self.fc1 = nn.Linear(in_features, output)
    
    def forward(self, x):
        return self.fc1(x)
    
# 2 layers
class Lin2(nn.Module):
    def __init__(self, in_features, fc2_in, output, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features, fc2_in)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc2_in, output)
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.dropout1(x)
        return self.fc2(x)

# 3 layers
class Lin3(nn.Module):
    def __init__(self, in_features, fc2_in, fc3_in, output, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features, fc2_in)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc2_in, fc3_in)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(fc3_in, output)
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return self.fc3(x)
