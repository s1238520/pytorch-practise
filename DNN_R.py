import torch
import torch.utils.data as TData
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)#建立範例資料
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x),Variable(y)

LR=0.1          #學習率
BATCH_SIZE=32   #分開資料的數量
EPOCH=12        #訓練次數

torch_dataset=TData.TensorDataset(x,y)
loader=TData.DataLoader(dataset=torch_dataset,
                       batch_size=BATCH_SIZE,
                       shuffle=False,
                       num_workers=0)

#畫出x,y的資料分布
#plt.scatter(x.data.numpy(),y.data.numpy())
#plt.show()

#建立網路模型，方法一
class Rmodel(nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        #用super函數調用初始父類
        super(Rmodel,self).__init__()
        self.hidden=nn.Linear(n_features,n_hidden)
        self.predict=nn.Linear(n_hidden,n_output)
    
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

#快速搭建，方法二
'''
model=nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,1),
)
'''

model=Rmodel(1,10,1)

#優化參數，設定學習率，設定loss function為MSE
optimizer=torch.optim.SGD(model.parameters(),lr=LR)
loss_func=nn.MSELoss()
d_l1=[]

for epoch in range(EPOCH):
    #print('Epoch',epoch)
    for step, (b_x,b_y) in enumerate(loader):
        #for i in range(1000):
        prediction=model(b_x)
        loss=loss_func(prediction,b_y)
        optimizer.zero_grad()#清空上一步的更新參數
        loss.backward()#反向傳遞，計算新的參數
        optimizer.step()#優化梯度，將更新的參數加到model的parameters
        d_l1.append(loss.data.numpy())

d_l2=[]
#畫圖
for i in d_l1:
    d_l2.append(i)
plt.ylabel("loss")
plt.plot(d_l2)
plt.show()


