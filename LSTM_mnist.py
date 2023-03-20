import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision              #下載數據用
import matplotlib.pyplot as plt

EPOCH=1
BATCH_SIZE=64
INPUT_SIZE=28 #rnn的每一步輸入值/圖片每一行的像素
TIME_STEP=28  #rnn的時間步數/圖片高度
LR=0.01
DOWNLOAD_MNIST=False

train_data =torchvision.datasets.MNIST(
    root='mnist',                               #下載手寫辨識
    train=True,                                 #下載訓練資料
    transform=torchvision.transforms.ToTensor(),#將下載的資料轉換成tensor格式
    download=DOWNLOAD_MNIST
)

train_loader=Data.DataLoader(dataset=train_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=0
                             )

test_data=torchvision.datasets.MNIST(
    root='mnist',
    train=False
)
test_x=test_data.data.type(torch.FloatTensor)[:2000]/255.
test_y=test_data.targets[:2000]

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,         #隱藏層的個數
            num_layers=1,           #層數
            batch_first=True        #主要是在處理資料時，batch_size是否放在第一個
        )
        self.out=nn.Linear(64,10)   #輸出層

    def forward(self,x):
        #LSTM 有兩個 hidden states, h_n是分線, h_c是主線
        # h_n->(n_layers, batch, hidden_size)   
        # h_c->(n_layers, batch, hidden_size)
        r_out,(h_n,h_c)=self.lstm(x,None) # None表示hidden state會用全0的state
        out=self.out(r_out[:,-1,:])       #這邊取最後一個值，因為最後輸出圖片只有一張
        return out

model=LSTM()
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
#print(test_x)

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x,b_y=x.view(-1,28,28),y
       
        output=model(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50==0:
            test_output=model(test_x)
            pred_y=torch.max(test_output,1)[1].data.squeeze()
            accuracy=sum(pred_y==test_y)/test_y.size(0)
            print("EPOCH",epoch,'train_loss:%4f'%loss.item(),'test_accuracy:%4f'%accuracy)

test_output=model(test_x[:10].view(-1,28,28))
pred_y=torch.max(test_output,1)[1].data.squeeze()
print(pred_y.numpy())
print(test_y[:10].numpy())


