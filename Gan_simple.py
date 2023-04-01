import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

BATCH_SIZE =64

LR_G=0.0001
LR_D=0.0001

N_IDEAS=5           #idea數量
ART_COMPONENTS=15   #產出ART_COMPONENTS數量

#線條生成，可以換成其他資料
PAINT_POINTS=np.vstack([np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE)])
def artist_works():
    a=np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
    paintings=a*np.power(PAINT_POINTS,2)+(a-1)
    paintings=torch.from_numpy(paintings).float()
    return paintings

#用5個idea創造15個art_components
G=nn.Sequential(
    nn.Linear(N_IDEAS,128),
    nn.ReLU(),
    nn.Linear(128,ART_COMPONENTS)
)
#將創造出來的15個art_components做辨別，用sigmoid轉成百分比
D=nn.Sequential(
    nn.Linear(ART_COMPONENTS,128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid()
)

G_opt=torch.optim.Adam(G.parameters(),lr=LR_G)
D_opt=torch.optim.Adam(D.parameters(),lr=LR_D)

for step in range(3000):
    artist_paintings=artist_works()
    G_ideas=torch.randn(BATCH_SIZE,N_IDEAS)
    G_paintings=G(G_ideas)
    prob_artist1=D(G_paintings)
    G_loss=torch.mean(torch.log(1.-prob_artist1))
    G_opt.zero_grad()
    G_loss.backward()
    G_opt.step()
    
    prob_artist0=D(artist_paintings)
    prob_artist1=D(G_paintings.detach())
    D_loss=-torch.mean(torch.log(prob_artist0)+torch.log(1.-prob_artist1))
    D_opt.zero_grad()
    D_loss.backward()
    D_opt.step()

    #畫圖，可以等圖畫完
    """
    if step % 50 == 0:  
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0],label='Generate')
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1,label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0,label='lower bound')
        plt.text(-.5,2.3,'D accuracy=%.2f' % prob_artist0.data.numpy().mean())
        plt.text(-.5,2,'D score= %.2f' % -D_loss.data.numpy())
        plt.ylim((0, 3))
        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(0.01)
plt.ioff()
plt.show()
"""