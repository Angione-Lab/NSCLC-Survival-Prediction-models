import torch
import torch.nn as nn

class CoxModel(nn.Module):
    def __init__(self, In_Nodes, Clinical_Nodes, Hidden_Node_1, Hidden_Node_2, Out_Nodes):
        super(CoxModel,self).__init__()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.sc1 = nn.Linear(In_Nodes, Hidden_Node_1)
        self.BN = nn.BatchNorm1d(Hidden_Node_2)
        self.sc2 = nn.Linear(Hidden_Node_1, Hidden_Node_2)
        self.sc3 = nn.Linear(Hidden_Node_2, Out_Nodes)
        self.sc4 = nn.Linear(Out_Nodes + Clinical_Nodes, Out_Nodes)
        self.sc5 = nn.Linear(Out_Nodes, 1, bias=False)
        self.sc5.weight.data.uniform_(-0.001, 0.001)
        
        
    def forward(self,x_1,x_2):
        x_1 = self.tanh(self.sc1(x_1))
        x_1 = self.dropout(x_1)
        x_1 = self.tanh(self.sc2(x_1))
        x_1 = self.BN(x_1)
        x_1 = self.tanh(self.sc3(x_1))
        x_cat = torch.cat((x_1, x_2),1)
        integr_layer = self.sc4(x_cat)
        cox_pred = self.sc5(integr_layer)
        
        return cox_pred
        
        
        