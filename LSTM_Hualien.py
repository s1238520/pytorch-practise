# 導入所需的套件
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# 設定隨機種子
torch.manual_seed(0)

# 假設上面資料是一個numpy陣列
df=pd.read_csv("花東線近十年搭車人數數據3.csv",encoding='utf-8')
data = np.array(df)
device=torch.device("cuda")
# 將資料分成訓練集和測試集

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train_data = data[:train_size]
test_data = data[train_size:]
#print(train_data)


# 定義一個函數來創建序列資料
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 設定序列長度為12
seq_length = 2

# 創建訓練集和測試集的序列資料
train_x, train_y = create_sequences(train_data, seq_length)
test_x, test_y = create_sequences(test_data, seq_length)
#print(train_y[0])


# 將numpy陣列轉換成pytorch tensor
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()


# 定義lstm模型的類別
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.input_size = input_size # 輸入特徵維度
        self.hidden_size = hidden_size # 隱藏層特徵維度
        self.num_layers = num_layers # lstm層數
        
        # 定義lstm層
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True) # 設定batch在第一維
        
        # 定義全連接層（輸出層）
        self.fc1 = nn.Linear(hidden_size, 1) # 預測下一個值
    
    def forward(self, x):
        
        # 初始化隱藏狀態和記憶單元（h_0 和 c_0）
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # 通過lstm層得到輸出（out）和最終隱藏狀態（h_n 和 c_n）
        out, (h_n,c_n) = self.lstm(x,(h_0,c_0))
        
        # 取最後一步
        out = out[:, -1, :]
        
        # 通過全連接層得到預測值
        out = self.fc1(out)
        
        return out


# 設定模型的參數
input_size = 1 # 輸入特徵維度為1（人數）
hidden_size = 32 # 隱藏層特徵維度為32
num_layers = 2 # lstm層數為2

# 創建模型實例
model = LSTMModel(input_size, hidden_size, num_layers)

# 定義優化器和損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # 使用Adam優化器，學習率為0.1
criterion = nn.MSELoss() # 使用均方誤差作為損失函數

# 設定訓練的迭代次數（epoch）
num_epochs = 1000

# 開始訓練模型
for epoch in range(num_epochs):
    # 將模型設定成訓練模式（啟用dropout等）
    model.train()
    
    # 將輸入和目標轉換成適合模型的形狀（batch_size, seq_length, input_size）
    x = train_x.reshape(-1, seq_length, input_size)
    y = train_y.reshape(-1)
    
    # 清空梯度
    optimizer.zero_grad()
    
    # 前向傳播，得到預測值
    y_pred = model(x)
    
    # 計算損失值
    loss = criterion(y_pred, y)
    
    # 反向傳播，計算梯度
    loss.backward()
    
    # 更新參數
    optimizer.step()

    # 每100個epoch打印一次損失值
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        
# 訓練完成後，將模型設定成評估模式（關閉dropout等）
model.eval()

# 將測試集的輸入轉換成適合模型的形狀（batch_size, seq_length, input_size）
x_test = test_x.reshape(-1, seq_length, input_size)

# 得到測試集的預測值
y_test_pred = model(x_test)

# 計算測試集的損失值
test_loss = criterion(y_test_pred, test_y)
print(f'Test Loss: {test_loss.item():.4f}')

# 將預測值和真實值轉換成numpy陣列，方便畫圖
y_test_pred = y_test_pred.detach().numpy()
y_test = test_y.numpy()

# 畫出預測值和真實值的折線圖，比較差異
import matplotlib.pyplot as plt
plt.plot(y_test_pred, label='Predicted')
plt.plot(y_test, label='Actual')
plt.legend()
plt.show()
