import numpy as np 
from gen_func import *
from MMF import *
from MMNAEMO import *

# Parameter Settings
problem = "MMF2"  # problem
n_obj = 2  # number of objectives
n_var = 2  # number of decision variables
xl = np.asarray([0, 0])
xu = np.asarray([1, 2])
repoint=[1.1, 1.1]


L_hard = 350 # lower limit of the size of global archive (equal to number of reference line), specific to n_obj
L_soft = 400  # upper limit of the size of global archive, specific to n_obj
maxgen = 490 # maximum generation (problem-specific)
div = 100  # number of division for defining reference lines (one layered)
mut_prob = 0.75   # mutation probability for mutation switching scheme
flag1 = True  # problem specific parameter (for mutation switching scheme)
flag2 = False  # problem specific parameter (for mutation switching scheme)


optimizer = MMNAEMO()
optimizer.define_exp(maxgen, L_hard, L_soft, div, mut_prob, problem, n_var, n_obj, xl, xu, flag1, flag2)
optimizer.optimize()

obtained_ps, obtained_pf = [], []
for i in range(0, len(optimizer.history)):
    for j in range(0, len(optimizer.history[i])):
        obtained_ps.append(optimizer.history[i][j][0])
        obtained_pf.append(optimizer.history[i][j][1])

d_x,d_y,o_x,o_y = [], [], [], []
for i in range(0, len(obtained_ps)):
    d_x.append(obtained_ps[i][0])
    d_y.append(obtained_ps[i][1])
for i in range(0,len(obtained_pf)):
    o_x.append(obtained_pf[i][0])
    o_y.append(obtained_pf[i][1])

import matplotlib.pyplot as plt
plt.scatter(d_x,d_y)
plt.show()

plt.scatter(o_x,o_y)
plt.show()