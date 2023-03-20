import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#建立資料
n_data = torch.ones(100, 2)#資料的型態
x0 = torch.normal(2*n_data, 1)  #標籤為0的x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)           #標籤為0的y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data, 1) #標籤為1的x data (tensor), shape=(100, 1)
y1 = torch.ones(100)            #標籤為1的y data (tensor), shape=(100, )

# 用torch.cat將資料合併
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  #設定為FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    #設定為LongTensor = 64-bit integer

#畫出資料分布，用y來決定點的顏色
#plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy())
#plt.show()

#建立網路模型，方法一
class Cmodel(nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        #用super函數調用初始父類
        super(Cmodel,self).__init__()
        self.hidden=nn.Linear(n_features,n_hidden)
        self.out=nn.Linear(n_hidden,n_output)
    
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.out(x)
        return x

#快速搭建，方法二
'''
model=nn.Sequential(
    nn.Linear(2,10),
    nn.ReLU(),
    nn.Linear(10,2),
)
'''

model=Cmodel(2,10,2)

#優化參數，設定學習率為0.1，設定loss function為CrossEntropyLoss
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
loss_func=nn.CrossEntropyLoss()


for i in range(1000):
    out=model(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()#清空上一步的更新參數
    loss.backward()#反向傳遞，計算新的參數
    optimizer.step()#優化梯度，將更新的參數加到model的parameters

#準確率計算

predictions=torch.max(out,1)[1]
pred_y=predictions.data.numpy().squeeze()
target_y=y.data.numpy()
accuracy=sum(pred_y==target_y)/len(target_y)

#畫圖
#plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy())#實際分布
plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y)#預測分布
plt.text(x=1.5,y=-4,s=("accurancy=%.2f" %accuracy))
plt.show()


