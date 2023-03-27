# import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def vectors(df, stride, meridian, xmin, xmax, ymin, ymax):
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
        if x <xmin or x>xmax:
            plot = False
        if y<ymin or y>ymax:
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
            # if abs(v) > 1.e-15 and abs(v1) > 1.e-15:
            dxy = ang_xy1-ang_xy + meridian
            # else:
            #     dxy = 0
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


#################################### main ####################################

if len(sys.argv) > 1:    
    droppedFile = sys.argv[1].strip().replace('\\', '/')
else:
    droppedFile = 'D:/Data/inpho/IS-8654/Lambert93-Korsika-H0_d10_vec.txt'

#data
f = open(droppedFile, "r")
title = f.readline().strip('\n').strip('"')
meridian = float(f.readline())
f.close()

nd = np.genfromtxt(droppedFile, skip_header=2)
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(211)
axn = fig.add_subplot(223, projection='polar')
axz = fig.add_subplot(224, projection='polar')

axn.set_theta_zero_location('N')
axn.set_theta_direction('clockwise')
axn.set_rmin(-90)
axn.set_rmax(-10)
axn.set_rticks([-80,-60,-40,-20])
axn.set_thetagrids((0,90,180,270), labels=('N', 'E','S','W'))
axn.set_title("Nadir view (looking downward)")

axz.set_theta_zero_location('S')
axz.set_rmin(90)
axz.set_rmax(10)
axz.set_rticks([80,60,40,20])
axz.set_thetagrids((0,90,180,270), labels=('N', 'E','S','W'))
axz.set_title("Zenith view (looking upward)")


sc = scales(nd)

####################

x, y, u, v, max = vectors(nd, 2, meridian, 0, 360, -80, 80)

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

# show scale at -80, 0, 80 deg vertical angle
for v in [-80,0,80]:
    val = s[(0,v)]
    ax.text(0, v, "{:.5f}".format(val))

fig.colorbar(im, ax=ax)
ax.set_ylabel('vertical angle')
ax.set_xlabel('azimuth (0->N, 90->E)')
ax.set_title(droppedFile)

#############

xg = np.linspace(0, 360, 37)
yg = np.linspace(-90, 0, 10)
x, y, z, s = filterValues(xg, yg, sc)
xx = np.array(x).reshape(37,10)
yy = np.array(y).reshape(37,10)
zz = np.array(z).reshape(37,10)

rad = np.linspace(-90, 0, 10)
azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)

im = axn.pcolormesh(th, r, zz, alpha=0.4, cmap=cmap, vmin = 0.998, vmax = 1.002)
# show scale at 0, -90
for v in [-90,0]:
    val = s[(0,v)]
    axn.text(0, v, "{:.5f}".format(val))

x, y, u, v, max = vectors(nd, 0, meridian, 0, 360, -90, 0)
xv = []
for a in x:
    xv.append(a/180*np.pi)

qu1 = axn.quiver(xv, y, u, v, angles='xy', width = 0.003)
qu1._init()
qu1.scale = scale

xm = [45, 135, 225, 315]
ym = [0, 0, 0, 0]
um, vm = [], []
for i, a in enumerate(xm):
    xm[i] = xm[i]/180*np.pi
for x, y in zip(xm, ym):
    um.append(meridian*180/np.pi)
    vm.append(0)

maxi = (1000*meridian*180/np.pi)
maxf = maxi/1000
qu = axn.quiver(xm, ym, um, np.zeros(shape=len(um)), angles='xy', width = 0.004, color='r', pivot='mid')
qu._init()
axn.text(xm[0], ym[0], s="meridian convergence: {:.3f}".format(maxf), color="r")

####################

xg = np.linspace(0, 360, 37)
yg = np.linspace(0, 90, 10)
x, y, z, s = filterValues(xg, yg, sc)
xx = np.array(x).reshape(37,10)
yy = np.array(y).reshape(37,10)
zz = np.array(z).reshape(37,10)

rad = np.linspace(0, 90, 10)
azm = np.linspace(0, 2 * np.pi, 37)
r, th = np.meshgrid(rad, azm)

im = axz.pcolormesh(th, r, zz, alpha=0.4, cmap=cmap, vmin = 0.998, vmax = 1.002)
# show scale at 0, -90
for v in [90,0]:
    val = s[(0,v)]
    axz.text(0, v, "{:.5f}".format(val))

x, y, u, v, max = vectors(nd, 0, meridian, 0, 360, 0, 90)
xv = []
for a in x:
    xv.append(a/180*np.pi)

qu1 = axz.quiver(xv, y, u, v, angles='xy', width = 0.003)
qu1._init()
qu1.scale = scale

xm = [45, 135, 225, 315]
ym = [0, 0, 0, 0]
um, vm = [], []
for i, a in enumerate(xm):
    xm[i] = xm[i]/180*np.pi
for x, y in zip(xm, ym):
    um.append(meridian*180/np.pi)
    vm.append(0)

maxi = (1000*meridian*180/np.pi)
maxf = maxi/1000
qu = axz.quiver(xm, ym, um, np.zeros(shape=len(um)), angles='xy', width = 0.004, color='r', pivot='mid')
qu._init()
# axz.text(xm[0], ym[0], s="meridian convergence: {:.3f}".format(maxf), color="r")



plt.tight_layout()
plt.show()

print("")