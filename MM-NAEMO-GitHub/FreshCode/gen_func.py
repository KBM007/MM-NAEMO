import numpy as np
import copy
from math import *
import random
import time
import operator
import csv
from numpy import linalg as LA
from random import shuffle 
import math

EPS = 1.0e-14

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)



def SBX_mut(history,nbr,p1,eta_c,cross_prob,min_x,max_x):
	size = len(p1)
	
	a = random.sample(range(0,len(nbr)),1)
	r1 = random.randint(0,len(history[nbr[a[0]]])-1)		
	p2 = copy.deepcopy(history[nbr[a[0]]][r1][0])
	
	c1,c2 = SBX(p1,p2,eta_c,cross_prob,min_x,max_x)
	v_new = []
	rnd = random.random()
	if (rnd > 0.5):
		v_new = c1
	else:
		v_new = c2
	return v_new


def SBX_mutate_2(history,nbr,p1,eta_c,cross_prob,min_x,max_x):
	size = len(p1)
	p2 = []
	if random.random() < 0.0:
		tot_len = 0
		for i in range(0,len(history)):
			tot_len += len(history[i])
		r1 = random.randint(1,tot_len)
		ch_ind = 0
		while r1 > len(history[ch_ind]):
			r1 = r1 - len(history[ch_ind])
			ch_ind += 1
			
		p2 = copy.deepcopy(history[ch_ind][r1-1][0])
	else:
		a = random.sample(range(0,len(nbr)),1)
		r1 = random.randint(0,len(history[nbr[a[0]]])-1)
		p2 = copy.deepcopy(history[nbr[a[0]]][r1][0])

	c1,c2 = SBX(p1,p2,eta_c,cross_prob,min_x,max_x)
	v_new = []
	rnd = random.random()
	if (rnd > 0.5):
		v_new = c1
	else:
		v_new = c2
	return v_new


def diff_mut(history,nbr,v,F,CR,min_x,max_x):
	size = len(v)
	
	a = random.sample(range(0,len(nbr)),3) # without replacement
	r1 = random.randint(0,len(history[nbr[a[0]]])-1)
	p1 = copy.deepcopy(history[nbr[a[0]]][r1][0])
	r2 = random.randint(0,len(history[nbr[a[1]]])-1)
	p2 = copy.deepcopy(history[nbr[a[1]]][r2][0])
	r3 = random.randint(0,len(history[nbr[a[2]]])-1)
	p3 = copy.deepcopy(history[nbr[a[2]]][r3][0])
	
	k_rand = random.randint(0,size-1)
	u = copy.deepcopy(v)
	for k in range(0,size):
		r = random.random()
		if((r<CR) or (k==k_rand)):
			u[k] = min(max_x[k],max(p1[k] + F*(p2[k] - p3[k]),min_x[k]))
	return u

def diff_mutate(archive,v,F,CR,min_x,max_x):
	size = len(v)
	a = random.sample(range(0,len(archive)),3)
	p1 = copy.deepcopy(archive[a[0]][0])
	p2 = copy.deepcopy(archive[a[1]][0])
	p3 = copy.deepcopy(archive[a[2]][0])
	
	k_rand = random.randint(0,size-1)
	u = copy.deepcopy(v)
	for k in range(0,size):
		r = random.random()
		if((r<CR) or (k==k_rand)):
			u[k] = min(max_x[k],max(p1[k] + F*(p2[k] - p3[k]),min_x[k]))
	return u


def diff_mutate_2(history,nbr,v,F,CR,min_x,max_x):
	size = len(v)
	p1 = []
	p2 = []
	p3 = []
	if random.random() < 0.0:
		tot_len = 0
		for i in range(0,len(history)):
			tot_len += len(history[i])
		a = random.sample(range(1,tot_len+1),3)

		ch_ind1 = 0
		while a[0] > len(history[ch_ind1]):
			a[0] = a[0] - len(history[ch_ind1])
			ch_ind1 += 1

		ch_ind2 = 0
		while a[1] > len(history[ch_ind2]):
			a[1] = a[1] - len(history[ch_ind2])
			ch_ind2 += 1

		ch_ind3 = 0
		while a[2] > len(history[ch_ind3]):
			a[2] = a[2] - len(history[ch_ind3])
			ch_ind3 += 1

		p1 = copy.deepcopy(history[ch_ind1][a[0]-1][0])
		p2 = copy.deepcopy(history[ch_ind2][a[1]-1][0])
		p3 = copy.deepcopy(history[ch_ind3][a[2]-1][0])
	else:
		a = random.sample(range(0,len(nbr)),3)
		r1 = random.randint(0,len(history[nbr[a[0]]])-1)
		p1 = copy.deepcopy(history[nbr[a[0]]][r1][0])
		r2 = random.randint(0,len(history[nbr[a[1]]])-1)
		p2 = copy.deepcopy(history[nbr[a[1]]][r2][0])
		r3 = random.randint(0,len(history[nbr[a[2]]])-1)
		p3 = copy.deepcopy(history[nbr[a[2]]][r3][0])
	
	k_rand = random.randint(0,size-1)
	u = copy.deepcopy(v)
	for k in range(0,size):
		r = random.random()
		if((r<CR) or (k==k_rand)):
			u[k] = min(max_x[k],max(p1[k] + F*(p2[k] - p3[k]),min_x[k]))
	return u



def real_mutate(v,b,min_x,max_x):
	v_new = copy.deepcopy(v)
	r = random.randint(0,len(v)-1)
	y = v_new[r]
	d_rnd = random.random() - 0.5
	d_rnd_lap = 0.0
	
	if( d_rnd < 0 ):
		d_rnd_lap = b*log(1.0 - 2.0*abs(d_rnd))
	else:
		d_rnd_lap = -1.0*b*log(1.0 - 2.0*abs(d_rnd))
	y = y + d_rnd_lap
	i_count = 0
	while(((y < min_x[r]) or (y > max_x[r])) and (i_count < 20)):
		y = v_new[r]
		d_rnd = random.random() - 0.5
		d_rnd_lap = 0.0
	
		if( d_rnd < 0.0):
			d_rnd_lap = b*log(1.0 - (2.0*abs(d_rnd)))
		else:
			d_rnd_lap = -1.0*b*log(1.0 - (2.0*abs(d_rnd)))
		y = y + d_rnd_lap
		i_count = i_count + 1
	v_new[r] = y
	if ( i_count == 20 ):
		if(v_new[r] < min_x[r]):
			v_new[r] = min_x[r]
		elif(v_new[r] > max_x[r]):
			v_new[r] = max_x[r]
	return v_new

def form_ref_tab(ref_pts,nbr_size):
	table = []
	for i in range(0,len(ref_pts)):
		dist = []
		ind = []
		for j in range(0,len(ref_pts)):
			d = LA.norm(ref_pts[i]-ref_pts[j])
			dist.append(d)
			ind.append(j)
		ind = [x for (y,x) in sorted(zip(dist,ind))]
		table.append(ind[0:nbr_size])
	return table

def check_dominate(v1,v2):
	s = (v1<v2).sum()
	if s == len(v1):
		return 1
	elif s == 0:
		return -1
	else :
		return 0

def check_dominate_(x1,x2):
	flag1 = 0
	flag2 = 0
	for i in range(0,len(x1)):
		if(x1[i] < x2[i]):
			flag1 = 1
		if(x1[i] > x2[i]):
			flag2 = 1
	if ( flag1 == 1 and flag2 == 0):
		return 1					#	x1 dominates x2 (for min)
	elif (flag2 == 1 and flag1 == 0):
		return -1					#	x2 dominates x1	(for min)
	else:
		return 0					#	non-dominated


def d2(data,point):
	size = len(data)
	n1 = LA.norm(point)
	point = (np.dot(data,point)/(n1**2))*point
	return LA.norm(data - point)	

def IGD(ref_points,archive):
	d = 0.0
	for i in range(0,len(ref_points)):
		min_d = LA.norm(ref_points[i]-archive[0][1])
		for j in range(1,len(archive)):
			dd = LA.norm(ref_points[i]-archive[j][1]) 
			if dd < min_d :
				min_d = dd
		d = d + min_d
	d = d/len(ref_points)
	return d


def polynomial_mutate(v,mut_prob,eta_m,min_x,max_x):
	v_new = copy.deepcopy(v)
	for i in range(0,len(v)):
		r = random.random()
		if (r < mut_prob):
			y = v_new[i]
			yl = min_x[i]
			yu = max_x[i]
			
			delta1 = (y-yl)/(yu-yl)
			delta2 = (yu-y)/(yu-yl)

			mut_pow = 1.0/(eta_m + 1.0)
			deltaq = 0.0

			rnd = random.random()

			if(rnd <= 0.5):
				xy = 1.0 - delta1
				val = (2.0*rnd)+((1.0-(2.0*rnd))*(xy**(eta_m+1.0)))
				deltaq = (val**mut_pow) - 1.0
			else:
				xy = 1.0 - delta2
				val = (2.0*(1.0-rnd))+(2.0*(rnd-0.5)*(xy**(eta_m+1.0)))
				deltaq = 1.0 - (val**mut_pow)
			
			y = y + deltaq*(yu-yl)
			y = min(yu, max(yl, y))
			v_new[i] = y
	return v_new

def SBX(parent1,parent2,eta_c,cross_prob,min_x,max_x):
	child1 = copy.deepcopy(parent1)
	child2 = copy.deepcopy(parent2)
	r = random.random()
	if(r < cross_prob):
		for i in range(0,len(parent1)):
			r = random.random()
			if(r <= 0.5 ):
				if(abs(parent1[i]-parent2[i]) > EPS):
					y1 = min(parent1[i],parent2[i])
					y2 = max(parent1[i],parent2[i])
					yl = min_x[i] 
					yu = max_x[i]
					r = random.random()
					beta = 1.0 + (2.0*(y1-yl)/(y2-y1))
					alpha = 2.0 - (beta**(-1.0*(1.0 + eta_c)))
					betaq = 0.0
					if (r <= (1.0/alpha)):
						betaq = (r*alpha)**(1.0/(eta_c + 1.0))
					else:
						betaq = (1.0/(2.0 - (r*alpha)))**(1.0/(eta_c + 1.0))
					child1[i] = 0.5*(y1+y2-(betaq*(y2-y1)))

					beta = 1.0 + (2.0*(yu-y2)/(y2-y1))
					alpha = 2.0 - (beta**(-1.0*(1.0 + eta_c)))
					if (r <= (1.0/alpha)) :
						betaq = (r*alpha)**(1.0/(eta_c + 1.0))
					else:
						betaq = (1.0/(2.0 - (r*alpha)))**(1.0/(eta_c + 1.0))
					child2[i] = 0.5*((y1+y2)+(betaq*(y2-y1)))

					child1[i] = min(yu, max(yl, child1[i]))
					child2[i] = min(yu, max(yl, child2[i]))
					
					r = random.random()
					if (r <= 0.5):
						sw = child1[i]
						child1[i] = child2[i]
						child2[i] = sw
					
	return child1,child2

def lehmer_mean(f,p):
	return sum(f**p)/sum(f**(p-1))

def generate(m1,v1,m2,v2,m3,v3):
	F1 = np.random.normal(m1,v1)
	if F1 > 1:
		F1 = 1.0
	if F1 < 0:
		F1 = 0.000001
	F2 = np.random.normal(m2,v2)
	if F2 > 1:
		F2 = 1.0
	if F2 < 0:
		F2 = 0.000001
	CR = np.random.normal(m3,v3)
	if CR > 1:
		CR = 1.0
	if CR < 0:
		CR = 0.000001
	return F1,F2,CR
	
def crowd_sort(data,mx,mn):
		inf = 0
		for i in range(0,len(data)):
			data[i][1] = np.append(data[i][1],0)
		d = len(data[0][1])-1
		for j in range(0,d-1):
			data = sorted(data,key = lambda x : x[1][j] )#############
			data[0][1][d] = -1
			data[len(data)-1][1][d] = -1			#	assign largest distance to boundary solutions
			for i in range(1,len(data)-1):
				if data[i][1][d] != -1:
					data[i][1][d] = data[i][1][d] + ((data[i+1][1][j] - data[i-1][1][j])/(mx[j] - mn[j]))	#	distance = half of perimeter of bounding rectangle
					if data[i][1][d] > inf:		#	for finding max so far
						inf = data[i][1][d]
		for i in range(0,len(data)):
			if data[i][1][d] == -1:
				data[i][1][d] = inf + 1			#	assigning the infinity as the number greater than the max
		data = sorted(data,key = lambda x : x[1][d],reverse=True)	#	sorting in reverse
		for i in range(0,len(data)):
			data[i][1] = np.delete(data[i][1],len(data[i][1])-1)
		return data

def nds(data,population_size):
	n = []
	S = []
	P = [[]]
	P_d = [[]]
	for i in range(0,len(data)):
		n.append(0)
		S_ = []
		for j in range(0,len(data)):
			if (j != i):
				d = check_dominate_(data[i][1],data[j][1])
				if (d == 1):
					S_.append(j)
				if (d == -1):
					n[i] = n[i] + 1
		S.append(S_)
		if(n[i] == 0):
			P[0].append(i)
			P_d[0].append(data[i])
	k = 0
	l = len(P[k])
	while (l<population_size):
		Q = []
		Q_d = []
		for p in P[k]:
			for q in S[p]:
				n[q] = n[q] - 1
				if (n[q] == 0):
					Q.append(q)
					Q_d.append(data[q])
		k = k+1
		l = l + len(Q)
		P.append(Q)
		P_d.append(Q_d)

	return P_d,l

def nds_th(arch,norm,archive_ptr,theta,l_ref):
	ref_sets = []
	l = 0
	for i in range(0,l_ref):
		rf = []
		for j in range(0,len(archive_ptr)):
			if(archive_ptr[j][0] == i) :
				fitness = sqrt((LA.norm(norm[j])**2) - (archive_ptr[j][1]**2)) + (theta*archive_ptr[j][1])
				mem = [arch[j],fitness]
				rf.append(mem)
		rf.sort(key=lambda x: x[1])
		if len(rf) > l :
			l = len(rf)
		ref_sets.append(rf)
	
	final = []
	for i in range(0,l):
		rank = []
		for j in range(0,len(ref_sets)):
			if len(ref_sets[j]) > i :
				rank.append(ref_sets[j][i])
		final.append(rank)
	for i in range(0,len(final)):
		for j in range(0,len(final[i])):
			final[i][j] = final[i][j][0]
	return final




def associate_pt(pt,ref_pts):
	data_pair = []
	data_pair.append(0)
	data_pair.append(d2(pt,ref_pts[0]))
	for i in range(1,len(ref_pts)):
		d_val = d2(pt,ref_pts[i])
		if d_val < data_pair[1] : 
			data_pair[1] = d_val
			data_pair[0] = i
	return data_pair

def associate(norm,ref_pts):
	pop_arr = np.zeros(len(ref_pts))
	archive_pointer = []
	for i in range(0,len(norm)):
		data_pair = associate_pt(norm[i],ref_pts)
		archive_pointer.append(data_pair)
		pop_arr[int(data_pair[0])] += 1
	return pop_arr,archive_pointer



def form_ref_table(ref_pts,nbr_size):
	table = []
	for i in range(0,len(ref_pts)):
		dist = []
		ind = []
		for j in range(0,len(ref_pts)):
			d = LA.norm(ref_pts[i]-ref_pts[j])
			dist.append(d)
			ind.append(j)
		ind = [x for (y,x) in sorted(zip(dist,ind))]
		table.append(ind[1:nbr_size])
	return table

class form_ref_pts(object):
	def __init__(self,m,divisions):
		self.M = m-1
		self.div = divisions
		self.points = []

	def recursive(self,arr,d,l):
		arr_c = copy.deepcopy(arr)
		if d == self.M-1:
			self.points.append(arr_c)
		else:
			for i in range(0,l):
				node_val = float(i)/float(self.div)
				arr_next = copy.deepcopy(arr_c)
				arr_next.append(node_val)
				self.recursive(arr_next,d+1,l-i)

	def form(self):
		layer = []
		for i in range(0,self.div+1):
			layer.append(float(i)/float(self.div))
		for i in range(0,len(layer)):
			l1 = []
			l1.append(layer[i])
			self.recursive(l1,0,len(layer)-i)
		for i in range(0,len(self.points)):
			s = sum(self.points[i])
			self.points[i].append(1.0-s)
		self.points = np.asarray(self.points)
	

def IGD_calculation(ps, PS):
	n_ref = len(PS)
	obtained_to_ref = []
	for i in range(0, n_ref):
		ref_m = np.matlib.repmat(PS[i,:], len(ps), 1)
		d = ps -ref_m
		D = np.power(np.power(abs(d), 2).sum(axis =1), 0.5)
		obtained_to_ref.append(min(D))
	IGD = sum(obtained_to_ref)/n_ref

	return IGD


def Hypervolume_Calculation(pf, repoint):

	popsize = len(pf)
	temp_index = np.argsort(pf[:,0])
	sorted_pf = pf[temp_index,:]
	pointset = np.append([repoint], sorted_pf, axis =0)
	hyp = 0
	for i in range(0, popsize):
		cubei = (pointset[0,0] -pointset[i+1,0])*(pointset[i,1] -pointset[i+1,1])
		hyp = hyp + cubei
	
	return hyp


def CR_calculation(ps, PS):

	n_var = PS.shape[1]
	obtained_min = np.min(ps, axis=0)
	obtained_max = np.max(ps, axis=0)

	reference_min = np.min(PS, axis=0)
	reference_max = np.max(PS, axis=0)

	kesi = [0.0]*n_var
	for i in range(0, n_var):
		if reference_max[i] == reference_min[i]:
			kesi[i] = 1
		elif obtained_min[i] >= reference_max[i] or reference_min[i] >= obtained_max[i]:
			kesi[i] = 0
		else:
			kesi[i] = ((min(obtained_max[i],reference_max[i]) -max(obtained_min[i],reference_min[i]))/(reference_max[i]-reference_min[i]))**2

	CR = pow(np.prod(kesi), 1/(2*n_var))
	return CR



def NDSort(data, population_size):
    n = []
    S = []
    P = [[]]
    P_d = [[]]



    for i in range(0, len(data)):
        n.append(0)
        S_ = []
        for j in range(0, len(data)):
            if (j != i):
                d = check_dominate_(data[i][1], data[j][1])
                if (d == 1):
                    S_.append(j)
                if (d == -1):
                    n[i] = n[i] +1
        S.append(S_)
        if (n[i] == 0):
            P[0].append(i)
            P_d[0].append(data[i])
    k = 0
    l = len(P[k])

    while (l<len(data)):
            Q = []
            Q_d = []
            for p in P[k]: # list of indexes of nondominated points
                    for q in S[p]:
                            n[q] = n[q] -1
                            if (n[q] == 0):
                                    Q.append(q)
                                    Q_d.append(data[q])
                    k = k+1
                    l = l + len(Q)
                    P.append(Q)
                    P_d.append(Q_d)

    return P_d, l