import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def plot_vectors(df, stride):
    rx = []
    ry = []
    ru = []
    rv = []
    ru1 =[]
    max = 0
    xalt = -1
    yalt = -1
    skipx = 0
    skipy = 0
    for x,y,u,v,u1 in zip(df[0], df[1], df[3], df[4], df[5]):
        plot = True
        if xalt != x:
            xalt = x
            skipx += 1
            if skipx < stride:
                plot = False
            else:
                skipx = 0
        if yalt != y:
            yalt = y
            skipy += 1
            if skipy < stride:
                plot = False
            else:
                skipy = 0
        if plot:
            rx.append(x)
            ry.append(y)
            ru.append(u)
            rv.append(v)
            ru1.append(u1)
            l = np.sqrt(u*u+v*v)
            if l > max:
                max = l
    return rx, ry, ru, rv, ru1, max
    
    
droppedFile = sys.argv[1].strip().replace('\\', '/')
# droppedFile = 'D:/Data/inpho/IS-8654/Lambert93-Korsika-H0_d10_vec.txt'
df = pd.read_csv(droppedFile, delim_whitespace=True, header=None, dtype=np.float64, skiprows=1)

out = df.pivot_table(2, pd.cut(df[1], bins=17), pd.cut(df[0], bins=37), aggfunc='mean')

fig, ax = plt.subplots(figsize=(10,5))

x, y, u, v, u1, max = plot_vectors(df, 2)

maxi = (1000*max*180/np.pi)
maxf = maxi/1000
lv = lambda v: 0
qu = ax.quiver(x, y, u, np.zeros(shape=len(u)), angles='xy', width = 0.001, color='r')
qu1 = ax.quiver(x, y, u1, v, angles='xy', width = 0.002)
qu1._init()
scale = qu1.scale
scali = int(20000*scale)
scalef = scali/10000
ax.quiverkey(qu1, 1.1, -0.12, scalef*np.pi/180, str(scalef) + ' deg')
ax.quiverkey(qu, 0.85, -0.12, maxf*np.pi/180, str(maxf) + ' deg mer.conv.')

cmap = sns.color_palette("coolwarm", as_cmap=True)
im = ax.pcolormesh(np.linspace(0, 360, 37), np.linspace(-80, 80, 17), out, alpha=0.4, cmap=cmap, vmin=0.998, vmax = 1.002)

# show scale at 10, 90, 170 deg vertical angle
for v in [-80,0,80]:
    d2 = df[(df[0]==0) & (df[1]==v)]
    val = d2.iloc[0][2]
    ax.text(0, v, str(val))

fig.colorbar(im, ax=ax)
ax.set_ylabel('vertical angle')
ax.set_xlabel('horizontal angle (0->N, 90->E)')
ax.set_title(droppedFile)

plt.tight_layout()
plt.show()

print("")