#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FormatStrFormatter

font_size=18
data_name = sys.argv[1]
lrate=sys.argv[2]
thread_count=sys.argv[3]
decay='1.'
rmse_SGD=[]
er_SGD=[]
grad_norm_SGD=[]
time_SGD=[]

rmse_ASGD=[]
er_ASGD=[]
grad_norm_ASGD=[]
time_ASGD=[]

rmse_IS_ASGD_rnd=[]
er_IS_ASGD_rnd=[]
grad_norm_IS_ASGD_rnd=[]
time_IS_ASGD_rnd=[]

rmse_IS_ASGD_sh=[]
er_IS_ASGD_sh=[]
grad_norm_IS_ASGD_sh=[]
time_IS_ASGD_sh=[]

rmse_SVRG_ASGD=[]
er_SVRG_ASGD=[]
grad_norm_SVRG_ASGD=[]
time_SVRG_ASGD=[]

found_SGD=1
found_ASGD=1
found_ISASGD_rnd=1
found_ISASGD_sh=1
found_SVRG=1

def get_improved(x,y):
    best=100
    improved_x=[]
    improved_y=[]
    for i,yi in enumerate(y):
        if yi<best:
            improved_y.append(yi)
            improved_x.append(x[i])
            best=yi
    return improved_x,improved_y

########################################
try:
    print('open ',data_name+'_rmse_1_lr_'+lrate+'_ld_'+decay)
    with open(data_name+'_rmse_1_lr_'+lrate+'_ld_'+decay) as f:
        for line in f:
            rmse_SGD.append(float(line.strip('\n').split()[0]))
            er_SGD.append(float(line.strip('\n').split()[1]))
            grad_norm_SGD.append(float(line.strip('\n').split()[2]))
            time_SGD.append(float(line.strip('\n').split()[3]))
    f.close()
    print(len(rmse_SGD))
except:
    pass
#######################################
#print('open ',data_name+'_rmse_'+str(thread_count)+'_lr_'+lrate+'_ld_'+lr)
with open(data_name+'_rmse_'+str(thread_count)+'_lr_'+lrate+'_ld_'+decay) as f:
    for line in f:
        rmse_ASGD.append(float(line.strip('\n').split()[0]))
        er_ASGD.append(float(line.strip('\n').split()[1]))
        grad_norm_ASGD.append(float(line.strip('\n').split()[2]))
        time_ASGD.append(float(line.strip('\n').split()[3]))
f.close()
#print(len(rmse_ASGD))

#print('open ',data_name+'_IS_random_Dis_rmse_'+str(thread_count)+'_lr_'+lrate+'_ld_'+lr)
with open(data_name+'_IS_random_Dis_rmse_'+str(thread_count)+'_lr_'+lrate+'_ld_'+decay) as f:
    for line in f:
        rmse_IS_ASGD_rnd.append(float(line.strip('\n').split()[0]))
        er_IS_ASGD_rnd.append(float(line.strip('\n').split()[1]))
        grad_norm_IS_ASGD_rnd.append(float(line.strip('\n').split()[2]))
        time_IS_ASGD_rnd.append(float(line.strip('\n').split()[3]))
f.close()
#print(len(rmse_IS_ASGD_rnd))

#print(len(rmse_IS_ASGD_sh))
#print('open ',data_name+'_SVRG_rmse_'+str(thread_count)+'_lr_'+lrate+'_lr_'+lr)
try:
    with open(data_name+'_SVRG_rmse_'+str(thread_count)+'_lr_'+lrate+'_lr_'+decay) as f:
        for line in f:
            rmse_SVRG_ASGD.append(float(line.strip('\n').split()[0]))
            er_SVRG_ASGD.append(float(line.strip('\n').split()[1]))
            grad_norm_SVRG_ASGD.append(float(line.strip('\n').split()[2]))
            time_SVRG_ASGD.append(float(line.strip('\n').split()[3]))
    f.close()
except:
    found_SVRG=0

ax=plt.subplot(121)
plt.plot(time_ASGD[1:],rmse_ASGD[1:],'r--',label='RMSE ASGD')
plt.plot(time_ASGD[1:],rmse_IS_ASGD_rnd[1:],'b-.',label='RMSE IS-ASGD')
if found_SVRG:
    time_SVRG_ASGD=np.asarray(time_SVRG_ASGD)
    time_SVRG_ASGD=time_SVRG_ASGD*0.1
    plt.plot(time_SVRG_ASGD[1:],rmse_SVRG_ASGD[1:],'y-.',label='RMSE SVRG-ASGD')
plt.legend(fancybox=True,frameon = False,fontsize = font_size)
ax=plt.subplot(122)
x,y=get_improved(time_ASGD,er_ASGD)
plt.plot(x,y,'r--',label='Error Rate ASGD')
x,y=get_improved(time_ASGD,er_IS_ASGD_rnd)
plt.plot(x,y,'b-.',label='Error Rate IS-ASGD')
if found_SVRG:
    x,y=get_improved(time_SVRG_ASGD,er_SVRG_ASGD)
    plt.plot(x,y,'y-.',label='Error Rate SVRG-ASGD')
plt.legend(fancybox=True,frameon = False,fontsize = font_size)
plt.show()
