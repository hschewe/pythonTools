# import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def vectors(df, stride, meridian):
    rx = []
    ry = []
    rs = []
    ru = []
    rv = []
    ru1 =[]
    max = 0
    xalt = -1
    yalt = -1
    skipx = 0
    skipy = 0
    for x,y, u,v,w, u1,v1,w1 in df:
        plot = True
        if y == -90 or y == 90:
            plot = False
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
            ang_xy = np.arctan2(u, v)
            ang_xy1 = np.arctan2(u1, v1)
            if abs(v) > 1.e-15 and abs(v1) > 1.e-15:
                dxy = ang_xy1-ang_xy + meridian
            else:
                dxy = 0
            if dxy > np.pi:
                dxy -= 2*np.pi
            if dxy < -np.pi:
                dxy += 2*np.pi
            e = np.sqrt(u*u+v*v)
            e1 = np.sqrt(u1*u1+v1*v1)
            ang_ez = np.arctan2(w, e)
            ang_ez1 = np.arctan2(w1, e1)
            if abs(e) > 1.e-15 and abs(e1) > 1.e-15:
                dez = ang_ez1-ang_ez
            else:
                dez = 0
            if dez > np.pi:
                dez -= 2*np.pi
            if dez < -np.pi:
                dez += 2*np.pi
            rx.append(x)
            ry.append(y)
            ru.append(dxy)
            rv.append(dez)
            ll = np.sqrt(dxy*dxy+dez*dez)
            if ll > max:
                max = ll
    return rx, ry, ru, rv, max
    
def scales(df):
    sc = np.ndarray(shape=(df.shape[0], 3))
    for i, (x,y, u,v,w, u1,v1,w1) in enumerate(df):
        l = np.sqrt(u*u+v*v+w*w)
        l1 = np.sqrt(u1*u1+v1*v1+w1*w1)  
        scale = l1/l
        sc[i]=[x,y,scale]
    return sc

def filterValues(xx, yy, sc):
    s = {}
    for ss in sc:
        if ss[0] in xx and ss[1] in yy:
            s[(ss[0],ss[1])] =ss[2]
    x, y, z = [], [], []
    for k1, k2 in s:
        x.append(k1)
        y.append(k2)
        z.append(s[(k1,k2)])
    return x, y, z, s

if len(sys.argv) > 1:    
    droppedFile = sys.argv[1].strip().replace('\\', '/')
else:
    droppedFile = 'D:/Data/inpho/IS-8654/GK-45-9-H0_d10_vec.txt'

#data
f = open(droppedFile, "r")
title = f.readline().strip('\n').strip('"')
meridian = float(f.readline())
f.close()

nd = np.genfromtxt(droppedFile, skip_header=2)

# out = df.pivot_table(2, pd.cut(df[1], bins=17), pd.cut(df[0], bins=37), aggfunc='mean')

fig, ax = plt.subplots(figsize=(10,5))

sc = scales(nd)
x, y, u, v, max = vectors(nd, 2, meridian)

qu1 = ax.quiver(x, y, u, v, angles='xy', width = 0.002)
qu1._init()
scale = qu1.scale
scali = int(20000*scale)
scalef = scali/10000
ax.quiverkey(qu1, 1.1, -0.12, scalef*np.pi/180, str(scalef) + ' deg')

xm = [180]
ym = [0]
um = [meridian*180/np.pi]
maxi = (1000*meridian*180/np.pi)
maxf = maxi/1000
qu = ax.quiver(xm, ym, um, np.zeros(shape=len(um)), angles='xy', width = 0.002, color='r', )
qu._init()
ax.text(xm[0]+meridian*qu.scale, ym[0], s="meridian convergence: {:.3f}".format(maxf), color="r")
# ax.quiverkey(qu, 0.7, 0.45, maxf*np.pi/180, "{:.3f}".format(maxf) + ' deg mer.conv.')

cmap = sns.color_palette("coolwarm", as_cmap=True)
xg = np.linspace(0, 360, 37)
yg = np.linspace(-80, 80, 17)
xm, ym = np.meshgrid(xg, yg)
x, y, z, s = filterValues(xg, yg, sc)
xx = np.array(x).reshape(37,17)
yy = np.array(y).reshape(37,17)
zz = np.array(z).reshape(37,17)

im = ax.pcolormesh(xx, yy, zz, alpha=0.4, cmap=cmap, vmin=0.998, vmax = 1.002)

# show scale at 10, 90, 170 deg vertical angle
for v in [-80,0,80]:
    val = s[(0,v)]
    ax.text(0, v, "{:.5f}".format(val))

fig.colorbar(im, ax=ax)
ax.set_ylabel('vertical angle')
ax.set_xlabel('azimuth (0->N, 90->E)')
ax.set_title(droppedFile)

plt.tight_layout()
plt.show()

print("")