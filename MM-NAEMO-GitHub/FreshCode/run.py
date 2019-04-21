import time
import numpy as np 
import os
import scipy.io as sio
from MMF import *
from gen_func import *
from MMNAEMO import *
import matplotlib.pyplot as plt
import math
pi = math.pi
a = math.e
import simplejson
savedata = 'PS_PF_data'

# Parameter Settings
problem = "MMF1"  # problem
n_obj = 2  # number of objectives
n_var = 2  # number of decision variables
xl = np.asarray([1, -1])
xu = np.asarray([3, 1])
repoint=[1.1, 1.1]


L_hard = 200 # lower limit of the size of global archive (equal to number of reference line), specific to n_obj
L_soft = 300  # upper limit of the size of global archive, specific to n_obj
maxgen = 93 # maximum generation (problem-specific)
div = 100  # number of division for defining reference lines (one layered)
mut_prob = 0.75   # mutation probability for mutation switching scheme
flag1 = True  # problem specific parameter (for mutation switching scheme)
flag2 = False  # problem specific parameter (for mutation switching scheme)


# load reference PS and PF

func = problem + "_Reference_PSPF_data.mat"
reference_ps_pf = sio.loadmat(func)

PS = reference_ps_pf['PS'] 
PF = reference_ps_pf['PF']

HYP, IGDX, IGDF, CR_, PSP_ = [],[],[],[],[]
runtimes = 21
for run in range(0, runtimes):
    optimizer = MMNAEMO()
    optimizer.define_exp(maxgen, L_hard, L_soft, div, mut_prob, problem, n_var, n_obj, xl, xu, flag1, flag2)
    optimizer.optimize()

    obtained_ps, obtained_pf = [], []
    for i in range(0, len(optimizer.history)):
        for j in range(0, len(optimizer.history[i])):
            obtained_ps.append(optimizer.history[i][j][0])
            obtained_pf.append(optimizer.history[i][j][1])
    ps_pf_ = {}
    ps_pf_['ps'] = obtained_ps
    ps_pf_['pf'] = obtained_pf

    sio.savemat(os.path.join(savedata, problem) +'_%d'%run, ps_pf_)
    obtained_ps_pf = sio.loadmat(os.path.join(savedata, problem) +'_%d'%run)

    ps = obtained_ps_pf['ps']
    pf = obtained_ps_pf['pf']

    hyp = Hypervolume_Calculation(pf, repoint)
    IGDx = IGD_calculation(ps, PS)
    IGDf = IGD_calculation(pf, PF)
    CR = CR_calculation(ps, PS)
    PSP = CR/IGDx

    HYP.append(round(1.0/hyp, 4))
    IGDX.append(round(IGDx, 4))
    IGDF.append(round(IGDf, 4))
    CR_.append(round(CR, 4))
    PSP_.append(round(1.0/PSP, 4))



min_hyp = np.min(HYP)
min_IGDX = np.min(IGDX)
min_IGDF = np.min(IGDF)
min_CR_ = np.min(CR_)
min_PSP_ = np.min(PSP_) 
    
max_hyp = np.max(HYP)
max_IGDX = np.max(IGDX)
max_IGDF = np.max(IGDF)
max_CR_ = np.max(CR_)
max_PSP_ = np.max(PSP_) 

mean_hyp = np.mean(HYP)
mean_IGDX = np.mean(IGDX)
mean_IGDF = np.mean(IGDF)
mean_CR_ = np.mean(CR_)
mean_PSP_ = np.mean(PSP_) 

med_hyp = np.median(HYP)
med_IGDX = np.median(IGDX)
med_IGDF = np.median(IGDF)
med_CR_ = np.median(CR_)
med_PSP_ = np.median(PSP_) 


std_hyp = np.std(HYP)
std_IGDX = np.std(IGDX)
std_IGDF = np.std(IGDF)
std_CR_ = np.std(CR_)
std_PSP_ = np.std(PSP_) 

HYP.append(min_hyp)
HYP.append(max_hyp)
HYP.append(round(mean_hyp,4))
HYP.append(med_hyp)
HYP.append(round(std_hyp,4))

IGDX.append(min_IGDX)
IGDX.append(max_IGDX)
IGDX.append(round(mean_IGDX, 4))
IGDX.append(med_IGDX)
IGDX.append(round(std_IGDX, 4))

IGDF.append(min_IGDF)
IGDF.append(max_IGDF)
IGDF.append(round(mean_IGDF, 4))
IGDF.append(med_IGDF)
IGDF.append(round(std_IGDF, 4))

CR_.append(min_CR_)
CR_.append(max_CR_)
CR_.append(round(mean_CR_,4))
CR_.append(med_CR_)
CR_.append(round(std_CR_,4))

PSP_.append(min_PSP_)
PSP_.append(max_PSP_)
PSP_.append(round(mean_PSP_,4))
PSP_.append(med_PSP_)
PSP_.append(round(std_PSP_,4))

