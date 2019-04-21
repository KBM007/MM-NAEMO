import numpy as np 
from gen_func import *
from MMF import *
from sklearn.mixture import GaussianMixture
import time

class MMNAEMO(object):
    def define_exp(self,itrr,hard,soft,div,frc,fun,d_in,d_out,x_lo,x_up,flg1,flg2,e_m=20):
        self.flag1 = flg1
        self.flag2 = flg2
        self.itr = itrr
        self.theta = 5.0
        self.hard_l = hard
        self.soft_l = soft
        self.divisions = div
        self.gamma = 1
        self.mut1 = "SBX"
        self.mut2 = "diff"
        self.frac = frc
        self.F_m = 0.5
        self.CR_m = 0.2
        self.eta_c_m = 30.0/50.0
        self.F = 0.5
        self.CR = 0.2
        self.eta_c = 30.0
        self.eta_m_m = e_m
        self.eta_m = e_m
        self.func = fun
        self.dim_in = d_in
        self.dim_out = d_out
        self.mut_prob = 1.0/self.dim_in
        self.cross_prob = 1.0
        self.min_x = x_lo
        self.max_x = x_up
        

        factory = form_ref_pts(self.dim_out, div-1)
        factory.form()
        self.ref_pts = copy.deepcopy(factory.points)

        self.nbr_size = 0.2
        self.k = int(self.nbr_size*len(self.ref_pts))

        self.nbrs = form_ref_tab(self.ref_pts, len(self.ref_pts) -1)

        self.history = []
        for i in range(0, len(self.ref_pts)):
            self.history.append([])

        for i in range(0, self.gamma*self.soft_l):
            new = self.create_random_pt()
            data = associate_pt(new[1], self.ref_pts)
            d1 = LA.norm(new[1])
            g = d1 + (self.theta*data[1])
            self.history[data[0]].append([new[0], new[1], g, data[1]])
  
    def create_random_pt(self):
        decision = np.zeros(self.dim_in)
        for i in range(0, self.dim_in):
            r = self.min_x[i] + (random.random())*(self.max_x[i] -self.min_x[i])
            decision[i] = r
        objective = evaluate(self.func, decision, self.dim_out)
        
        return [decision, objective]

    def g(self, objective, i):
        d_2 = d2(objective, self.ref_pts[i])
        d_1 = LA.norm(objective)
        g = d_1 + (self.theta*d_2)
        return g

    def perturb(self, pt, mut, nbr_arr):
        decision = np.zeros(self.dim_in)
        if (mut == "poly_mut"):
            decision = polynomial_mutate(pt[0], self.mut_prob, self.eta_m, self.min_x, self.max_x)
        elif (mut == "SBX"):
            decision = SBX_mut(self.history, nbr_arr, pt[0], self.eta_c, self.cross_prob, self.min_x, self.max_x)
        else:
            decision = diff_mut(self.history, nbr_arr, pt[0], self.F, self.CR, self.min_x, self.max_x)

        objective = evaluate(self.func, decision, self.dim_out)
        
        return [decision, objective]


    def dom_amt_archive(self, indv):
        l = []
        for i in range(0, len(self.history)):
            points_to_remove = []
            for j in range(0, len(self.history[i])):
                ch = check_dominate(self.history[i][j][1], indv[1])
                if ch == -1:
                    points_to_remove.append(j)
            l.append(points_to_remove)
        
        return l

    def remove_from_archive_g(self, l):
            non_empty = [i for i,x in enumerate(l) if x != []]
            for item in non_empty:
                    if len(self.history[item]) > 2:
                            X_train = []
                            for j in range(0, len(self.history[item])):
                                    X_train.append(copy.deepcopy(self.history[item][j][0]))
                            gmm = GaussianMixture(n_components=2)
                            gmm.fit(X_train)
                            clusters = gmm.predict(X_train)
                        

                            n_1 = sum(clusters)
                            n_0 = len(X_train) - n_1
                            n = [n_0, n_1]
                            for j in range(0, len(clusters)):
                                    self.history[item][j].append(clusters[j])
                        
                        
                            for k in range(0, len(l[item])):
                                    ind = l[item][len(l[item]) -k -1]
                                    if n[int(clusters[ind])] > 1:
                                            self.history[item].pop(ind)
                                            n[int(clusters[ind])] -= 1
                        

    def cluster_g(self, limit):
            pop_arr = []
            for i in range(0, len(self.history)):
                if len(self.history[i]) > 2:
                    X_train = []
                    for j in range(0, len(self.history[i])):
                        X_train.append(copy.deepcopy(self.history[i][j][0]))
                    gmm = GaussianMixture(n_components=2)
                    gmm.fit(X_train)
                    clusters = gmm.predict(X_train)
                    for j in range(0, len(clusters)):
                        self.history[i][j].append(clusters[j])
                pop_arr.append(len(self.history[i]))
            l_sum = sum(pop_arr)
            while l_sum > limit:
                ind = pop_arr.index(max(pop_arr))
                cluster_0 = []
                cluster_1 = []

                for j in range(0, len(self.history[ind])):
                    if self.history[ind][j][-1] == 0:
                        cluster_0.append(copy.deepcopy(self.history[ind][j]))
                    else:
                        cluster_1.append(copy.deepcopy(self.history[ind][j]))

                cluster_0.sort(key = lambda x:x[2])
                cluster_1.sort(key = lambda x:x[2])

                if len(cluster_0) > len(cluster_1):
                    cluster_0.pop()
                elif len(cluster_1) > len(cluster_0):
                    cluster_1.pop()
                else:
                    if random.random() < 0.5:
                        cluster_0.pop()
                    else:
                        cluster_1.pop()

                self.history[ind] = cluster_0 + cluster_1
                pop_arr[ind] -= 1
                l_sum -= 1
      

    
    def iterrate(self, itr, gen):
            cur_nbrs = []
            cnt = 0
            j = 0

            k_val = self.k
            #k_val = ((float(self.itr - gen)/float(self.itr))*self.k) + 2


            while cnt < k_val and j < len(self.nbrs[itr]):
                    if len(self.history[self.nbrs[itr][j]]) > 0:
                            cur_nbrs.append(self.nbrs[itr][j])
                            cnt = cnt +1 
                    j = j+1

            ch_ind = itr
            while len(self.history[ch_ind]) == 0:
                    a = random.randint(0, len(cur_nbrs) -1)
                    ch_ind = cur_nbrs[a]

            r1 = random.randint(1, len(self.history[ch_ind]))
            current_pt = copy.deepcopy(self.history[ch_ind][r1-1])
  
            self.eta_c, self.F, self.CR = generate(self.eta_c_m, 0.1, self.F_m, 0.1, self.CR_m, 0.1)
            self.eta_c = 50.0*self.eta_c

            new_pt = []
            if random.random() > self.frac:
                    new_pt = self.perturb(current_pt, self.mut2, cur_nbrs)
                    if (self.flag2 == True):
                            new_pt = self.perturb(new_pt, "poly_mut", cur_nbrs)
                        
            else:
                    new_pt = self.perturb(current_pt, self.mut1, cur_nbrs)
                    if (self.flag1 == True):
                            new_pt = self.perturb(new_pt, "poly_mut", cur_nbrs)
            
            dom_stat = check_dominate(current_pt[1], new_pt[1]) != 1
            ret = 0
            if dom_stat == 1:
                    ret = 1
                    l = self.dom_amt_archive(new_pt)
                
                    data = associate_pt(new_pt[1],self.ref_pts)
                    g_val = self.g(new_pt[1], data[0])
                    new_pt.append(g_val)
                    new_pt.append(data[1])
                    self.history[data[0]].append(new_pt)

                    tot_l = 0
                    tot = 0

                
                    for i in range(0, len(l)):
                            tot_l = tot_l + len(l[i])
                            tot = tot + len(self.history[i])
                    if (tot_l == 0):
                            if tot > self.soft_l:
                                    self.cluster_g(self.hard_l)     
                    else:

                            self.remove_from_archive_g(l) 
            return ret



    def optimize(self):
    

            self.cluster_g(self.hard_l)
            for i in range(0, self.itr):
                eta_s = []
                F_s = []
                CR_s = []

                for j in range(0, len(self.ref_pts)):
                    a = self.iterrate(j, i)
                    if a == 1:
                        eta_s.append(self.eta_c)
                        F_s.append(self.F)
                        CR_s.append(self.CR)

                if len(eta_s) != 0:
                    eta_s = np.array(eta_s) /50.0
                    self.eta_c_m = lehmer_mean(eta_s, 1)

                if len(F_s) != 0:
                    F_s = np.array(F_s)
                    self.F_m = lehmer_mean(F_s, 1)

                if len(CR_s) != 0:
                    CR_s = np.array(CR_s)
                    self.CR_m = lehmer_mean(CR_s, 1)

                print('Progress: %0.2f%%' % (i*100/self.itr), end = '\r', flush = True)
                
            self.cluster_g(self.hard_l)
