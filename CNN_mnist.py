import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision              #下載數據用
import matplotlib.pyplot as plt

EPOCH=1
BATCH_SIZE=50
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

test_x=Variable(torch.unsqueeze(test_data.data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y=test_data.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(          #圖片格式為(1, 28, 28)->input shape (1, 28, 28)
                in_channels=1,  #高度
                out_channels=16,#filters的數量，將生成16張特徵圖形
                kernel_size=5,  #filters的size
                stride=1,       #filters的移動
                padding=2       #當filter移動到最邊邊時，可能會超過圖片大小，這時候補padding，將圖片圍上0，若要圖片大小不變則 
                                #if kernel=1,padding=(kernel_size-1)/2
            ),#圖片格式變為(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#圖片格式變為(16,14,14)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),#圖片格式變為(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)#圖片格式變為(32,7,7)
        )
        self.out=nn.Linear(32*7*7,10)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1) #攤平多维的捲積圖
        output=self.out(x)
        return output
    
model=CNN()
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x,b_y=Variable(x),Variable(y)
       
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

test_output=model(test_x[:10])
pred_y=torch.max(test_output,1)[1].data.squeeze()
print(pred_y.numpy())
print(test_y[:10].numpy())










