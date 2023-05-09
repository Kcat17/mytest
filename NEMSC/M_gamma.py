# -*- coding:utf-8 -*-

from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy
import matplotlib.pyplot as plt  # 导入 matplotlib
from PIL import ImageGrab, Image
import matplotlib
matplotlib.rcParams['font.family']='SimHei'

theta = 0
beta = 0.44
delta = 0.1
#phi = 0.02
gamma = 0.4
epsilon1 = 0.07
epsilon2 = 0.07
epsilon3 = 0.07
epsilon4 = 0.07
epsilon5 = 0.07
omega = 0  # 0.005
alpha = 0.8
mu = 0.01
iota = 0.01
eta=0 #促进因子
kappa=0.01
list = []
save_path = []
i = 0
color = ['black', 'green', 'blue', 'purple', 'pink']
lineclass = [',-', ':', '--', '-.', '-']


def dyNEMSC(y, t, alpha, theta, beta, delta, phi, gamma, epsilon2,epsilon3,epsilon4,epsilon5,omega, iota,kappa):  # SEIR 模型，导数函
    n, e, m, s, c = y  # youcans
    dn_dt = -(alpha-theta) * n * (e + m + s) - (iota+eta) * n + (epsilon5-theta) *c +(epsilon2+eta)*e# ds/dt = -lamda*s*i
    de_dt = (alpha-theta) * n * (e+ m +s) - (epsilon2+eta)*e  - (beta-theta) *e -(kappa+eta)*e+(epsilon3+eta)*m# de/dt = lamda*s*i - delta*e
    dm_dt =  (beta-theta) * e - (mu+eta) * m -(gamma-theta)*m-(epsilon3+eta)*m+(epsilon4+eta)*s # di/dt = delta*e - mu*i
    ds_dt = (gamma-theta) * m - (delta+eta)* s-(epsilon4+eta)*s
    dc_dt = (mu+eta) * m+ (kappa+eta) * e + (iota+eta) * n + (delta+eta)* s - ( epsilon5-theta) * c
    return np.array([dn_dt, de_dt, dm_dt, ds_dt, dc_dt])

tEnd = 100 # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)

# e0 = 1 #潜伏者比例的初
# m0 = 1 #恐慌者比例的初
# s0 = 1
# c0 = 1 #理性者比例的初
N = 1
e0 = 0.2 * N  # 潜伏者比例的初
m0 = 0.2 * N  # 恐慌者比例的初
c0 = 0.12 * N  # 理性者比例的初
n0 = N - e0 - m0 - c0  # 无知者比例的初

N = 1
e1 = 0.1 * N  # 潜伏者比例的初
m1 = 0.1 * N  # 恐慌者比例的初
s1 = 0.1* N
c1 = 0.2 * N  # 理性者比例的初
n1 = N - e1 - m1 - s1 - c1  # 无知者比例的初

Y0 = (n0, e0, m0, c0)  # 微分方程组的初
Y1 = (n1, e1, m1, s1, c1)  # 微分方程组的初
# yNEMC = odeint(dyNEMC, Y0, t, args=(alpha, theta, beta, delta, phi, gamma, epsilon, omega))  # NEMC 模型

# IEPRI 模型

i=0
#for i in range(5):
    # plt.cla()
print(i)
    # yNEMSC_N = np.empty((0, len(yNEMSC[:,0])), float)
    # yNEMSC_E = np.empty((0, len(yNEMSC[:,1])), float)
    # yNEMSC_M = np.empty((0, len(yNEMSC[:,2])), float)
    # yNEMSC_S = np.empty((0, len(yNEMSC[:,3])), float)
    # yNEMSC_C = np.empty((0, len(yNEMSC[:,4])), float)
    # yNEMSC_N =  np.vstack((yNEMSC_N, yNEMSC[:,0]))
    # yNEMSC_E =  np.vstack((yNEMSC_E, yNEMSC[:,1]))
    # yNEMSC_M =  np.vstack((yNEMSC_M, yNEMSC[:,2]))
    # yNEMSC_S =  np.vstack((yNEMSC_S, yNEMSC[:,3]))
    # yNEMSC_C =  np.vstack((yNEMSC_C, yNEMSC[:,4]))
theta = 0
beta = 0.6
delta = 0.1
#phi = 0.02
gamma = 0.4
epsilon1 = 0.07
epsilon2 = 0.07
epsilon3 = 0.07
epsilon4 = 0.07
epsilon5 = 0.07
# omega = 0  # 0.005
alpha = 0.55
mu = 0.01
iota = 0.01
eta=0 #促进因子
kappa=0.01
phi=0
saveP="altha="+str(alpha)+" "+"beta"+" "+str(beta)+"delta"+" "+str(delta)+"gamma"+" "+str(gamma)+"theta"+" "+str(theta)+"phi"+" "+str(phi)+"epsilon"+" "+str(epsilon1)+"eta"+" "+str(eta)

# yNEMSC = odeint(dyNEMSC, Y1, t, args=(alpha, theta, beta, delta, phi, gamma, epsilon2,epsilon3,epsilon4,epsilon5, omega, iota,kappa))
# plt.plot(t, yNEMSC[:, 0], lineclass[1], color=color[0], label="N" )
# plt.plot(t, yNEMSC[:, 1], lineclass[1], color=color[1], label="E")
# plt.plot(t, yNEMSC[:, 2], lineclass[1], color=color[2], label="M" )
# plt.plot(t, yNEMSC[:, 3], lineclass[1], color=color[3], label="S" )
# plt.plot(t, yNEMSC[:, 4], lineclass[1], color=color[4], label="C" )
for i in range(5):

    yNEMSC = odeint(dyNEMSC, Y1, t, args=(alpha, theta, beta, delta, phi, gamma, epsilon2,epsilon3,epsilon4,epsilon5, omega, iota,kappa))
    plt.plot(t, yNEMSC[:, 2], lineclass[i], color=color[i], label="gamma=" +str('{:.2f}'.format(gamma)))
    gamma-=0.02
# plt.plot(t, yNEMSC[:, 1], lineclass[1], color=color[1], label="E")
# plt.plot(t, yNEMSC[:, 2], lineclass[1], color=color[2], label="M" )

# plt.plot(t, yNEMSC[:, 3], lineclass[1], color=color[3], label="S" )
# plt.plot(t, yNEMSC[:, 4], lineclass[1], color=color[4], label="C" )

# plt.plot(t, yNEMSC_C[0,:], '-.', color='darkviolet', label='alpha=0.5')
plt.legend(loc='upper right')

# plt.title("M_alpha", fontsize=24)
plt.xlabel('迭代次数m',fontsize=14)
plt.ylabel('轻度恐慌人群(M)比例(%)',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right',fontsize=14)
plt.savefig("D:\NEMSC\results\M_gamma"+saveP+".png",dpi=300)
plt.show()

# for i in range(100):
# plt.cla()
# list.append(str(i))
# save_path.append("D:/shot/shotimage"+list[i]+".png")
# yNEMSC = odeint(dyNEMSC, Y1, t,args=(alpha, theta, beta, delta, phi, gamma, epsilon, omega,iota))  # IEPRI 模型
# print(alpha, theta, beta, delta, phi, gamma, epsilon, omega,mu,iota)
# plt.plot(t, yNEMSC[:,0], '-', color='darkviolet', label='n')
# plt.plot(t, yNEMSC[:,1], '-', color='orchid', label='e')
# plt.plot(t, yNEMSC[:,2], '-', color='red', label='m')
# plt.plot(t, yNEMSC[:,3], '-', color='blue', label='s')
# plt.plot(t, yNEMSC[:,4], '-', color='green', label='c')
# plt.legend(loc='right') # youcans
# plt.title("alpha=0.88"+""+"beta=0.14"+""+"theta=0.03"+""+"mu="+str('{:.2f}'.format(mu+0.01)),y=0,loc='right')
# plt.savefig(save_path[i],dpi=300)

# alpha=0.88#初始确定
# #beta =float('%.2f'%(beta+0.01))
# beta = 0.14
# #theta =float('%.2f'%(theta+0.01))
# theta = 0.03
# mu =float('%.2f'%(mu+0.01))
# print("i的值是"+str(i))
# for i in range(10):


# yNEMSC_N = np.empty((0, len(yNEMSC[:,0])), int)
# yNEMSC_E = np.empty((0, len(yNEMSC[:,1])), int)
# yNEMSC_M = np.empty((0, len(yNEMSC[:,2])), int)
# yNEMSC_S = np.empty((0, len(yNEMSC[:,3])), int)
# yNEMSC_C = np.empty((0, len(yNEMSC[:,4])), int)
# yNEMSC_N =  np.vstack((yNEMSC_N, yNEMSC[:,0]))
# alpha = 0.5


# yNEMSC = odeint(dyNEMSC, Y1, t,args=(alpha, theta, beta, delta, phi, gamma, epsilon, omega,iota))  # IEPRI 模型
# yNEMSC_N =  np.vstack((yNEMSC_N, yNEMSC[:,0]))
# yNEMSC_E =  np.vstack((yNEMSC_E, yNEMSC[:,1]))
# yNEMSC_M =  np.vstack((yNEMSC_M, yNEMSC[:,2]))
# yNEMSC_S =  np.vstack((yNEMSC_S, yNEMSC[:,3]))
# yNEMSC_C =  np.vstack((yNEMSC_C, yNEMSC[:,4]))
# alpha = 0.4
# yNEMSC = odeint(dyNEMSC, Y1, t,args=(alpha, theta, beta, delta, phi, gamma, epsilon, omega,iota))  # IEPRI 模型
# yNEMSC_N =  np.vstack((yNEMSC_N, yNEMSC[:,0]))
# yNEMSC_E =  np.vstack((yNEMSC_E, yNEMSC[:,1]))
# yNEMSC_M =  np.vstack((yNEMSC_M, yNEMSC[:,2]))
# yNEMSC_S =  np.vstack((yNEMSC_S, yNEMSC[:,3]))
# yNEMSC_C =  np.vstack((yNEMSC_C, yNEMSC[:,4]))
# alpha = 0.3
# yNEMSC = odeint(dyNEMSC, Y1, t,args=(alpha, theta, beta, delta, phi, gamma, epsilon, omega,iota))  # IEPRI 模型
# yNEMSC_N =  np.vstack((yNEMSC_N, yNEMSC[:,0]))
# yNEMSC_E =  np.vstack((yNEMSC_E, yNEMSC[:,1]))
# yNEMSC_M =  np.vstack((yNEMSC_M, yNEMSC[:,2]))
# yNEMSC_S =  np.vstack((yNEMSC_S, yNEMSC[:,3]))
# yNEMSC_C =  np.vstack((yNEMSC_C, yNEMSC[:,4]))

# print(t)
# plt.plot(t, yNEMSC_N[0,:], '-.', color='darkviolet', label='alpha=0.6')

# plt.plot(t, yNEMSC_N[0,:], '-.', color='orchid', label='alpha=0.5')
# plt.plot(t, yNEMSC_E[0,:], '-.', color='darkviolet', label='alpha=0.5')
# plt.plot(t, yNEMSC_M[0,:], '-.', color='darkviolet', label='alpha=0.5')
# plt.plot(t, yNEMSC_S[0,:], '-.', color='darkviolet', label='alpha=0.5')
# plt.plot(t, yNEMSC_C[0,:], '-.', color='darkviolet', label='alpha=0.5')


# plt.plot(t, yNEMSC_N[1,:], '-.', color='#111111', label='alpha=0.4')
# plt.plot(t, yNEMSC_E[1,:], '-.', color='darkviolet', label='alpha=0.4')
# plt.plot(t, yNEMSC_M[1,:], '-.', color='darkviolet', label='alpha=0.4')
# plt.plot(t, yNEMSC_S[1,:], '-.', color='darkviolet', label='alpha=0.4')
# plt.plot(t, yNEMSC_C[1,:], '-.', color='darkviolet', label='alpha=0.4')


# plt.plot(t, yNEMSC_N[1,:], '-.', color='#999999', label='alpha=0.3')
# plt.plot(t, yNEMSC_N[2,:], '-.', color='#FF0000', label='alpha=0.'+list[0])
# plt.plot(t, yNEMSC_N[3,:], '-.', color='#00FF00', label='alpha=0.3')


#   box = (500,200 ,1300,1300)
# img_region = ImageGrab.grab(box)
# img_region_pil = Image.frombytes("RGB",img_region.size,img_region.tobytes())
# img_region_pil.save(save_path)