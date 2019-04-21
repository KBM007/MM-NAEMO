import numpy as np
import math


def MMF1(input_array):
        x1 = abs((input_array[0]-2))
        f1 = x1
        f2 = 1.0 - math.sqrt(x1) + 2.0*(input_array[1] - math.sin(6*math.pi*x1 + math.pi))**2
        return np.asarray([f1,f2]) 

"""
def MMF2(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        f1 = x1
        if (1 < x2 <= 2):
                x2 = x2-1
        f2 = (1-math.sqrt(x1)) + 2*(4*(x2-math.sqrt(x1))**2 -2*math.cos((20*(x2 -math.sqrt(x1))*math.pi)/math.sqrt(2)) +2)
        return np.asarray([f1,f2]) 

"""
def MMF2(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        if x2>1:
                x2 = x2 -1
        f1 = x1
        y2 = x2 -x1**0.5
        f2 = 1.0 - math.sqrt(x1) + 2*((4*y2**2) -2*math.cos(20*y2*math.pi/math.sqrt(2))+2)
        return np.asarray([f1, f2])

def MMF3(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        f1 = x1
        if (0 <= x2 <= 0.5):
                y2 = x2 - x1**0.5
        if (0 <= x1 < 0.25) and (0.5 < x2 < 1):
                y2 = x2 - 0.5 - x1**0.5
        if (0.5 < x2 < 1) and (0.25 < x1 <= 1):
                y2 = x2 - x1**0.5
        if (1 <= x2 <= 1.5):
                y2 = x2 - 0.5 - x1**0.5
        
        f2 = 1.0 - (x1)**0.5 + 2*((4*y2**2) - 2*math.cos(20*y2*math.pi/math.sqrt(2)) + 2)
        return np.asarray([f1,f2])



def MMF4(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        f1 = abs(x1)
        if (0 <= x2 <1):
                f2 = (1-x1**2) +2*(x2 -math.sin(math.pi*abs(x1)))**2
        else:
                f2 = (1-x1**2) +2*(x2 -1 -math.sin(math.pi*abs(x1)))**2
        return np.asarray([f1,f2])


def MMF5(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        f1 = abs(x1 -2)
        if (1 <x2 <= 3):
                x2 = x2 -2
        else:
                x2 = x2
        f2 = 1 -math.sqrt(abs(x1 -2)) + 2*(x2 -math.sin(6*math.pi*abs(x1 -2) +math.pi))**2
        return np.asarray([f1,f2])


def MMF6(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        f1 = abs(x1 -2)
        if x2 >1:
                x2 = x2 -1
        f2 = 1 -math.sqrt(abs(x1 -2)) + 2*(x2 -math.sin(6*math.pi*abs(x1 -2) +math.pi))**2
        return np.asarray([f1,f2])



"""
def MMF6(input_array):
        x1 = input_array[0]
        x2 = input_array[1]
        if (x2 >-1 and x2 <=0) and (((x1 >7/6 and x1<=8/6)) or (x1>9/6 and x1 <=10/6) or (x1 >11/6 and x1<=2)):
                x2=x2
        elif (x2 > -1 and x2 <=0) and ((x1 >2 and x1 <=13/6) or (x1 > 14/6 and x1 <= 15/6) or (x1 > 16/6 and x1 <= 17/6)):
                x2=x2
        elif (x2 > 1 and x2 <=2) and ((x1 >1 and x1 <=7/6) or (x1 >4/3 and x1 <=3/2) or (x1 >5/3 or x1 <= 11/6)):  
                x2=x2-1
        elif (x2 >1 and x2 <=2) and ((x1 >13/6 and x1 <=14/6) or (x1 >15/6 and x1 <=16/6) or (x1 >17/6 and x1 <=3)):
                x2 = x2-1
        elif (x2 >0 and x2 <= 1) and ((x1 > 1 and x1 <= 7/6) or (x1 > 4/3 and x1 <= 3/2) or (x1 >5/3 and x1 <= 11/6) or (x1 > 13/6 and x1 <= 14/6) or (x1 >15/6 and x1 <= 16/6) or (x1 >17/6 and x1 <=3)):
                x2 =x2
        elif (x2 >0 and x2 <= 1) and ((x1 > 7/6 and x1 <= 8/6) or (x1 > 9/6 and x1 <= 10/6) or (x1 > 11/6 and x1 <= 2) or (x1 > 2 and x1 <= 13/6) or (x1 >14/6 and x1 <= 15/6) or (x1 > 16/6 and x1 <= 17/6)):
                x2 = x2 -1
        f1 = abs(x1-2)             
        f2 = 1.0 - math.sqrt(abs(x1-2)) + 2.0*(x2- math.sin(6*math.pi* abs(x1 -2)+ math.pi))**2 

        return np.asarray([f1, f2])
"""

def MMF7(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        f1 = abs(x1 -2)
        f2 = 1 -math.sqrt(abs(x1 -2)) +(x2 - ((np.float(0.3)*(abs(x1 -2))**2)*math.cos(24*math.pi*abs(x1 -2) +4*math.pi) +np.float(0.6)*abs(x1 -2))*math.sin(6*math.pi*abs(x1-2) +math.pi))**2
        return np.asarray([f1, f2]) 


def MMF8(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        f1 = math.sin(abs(x1))
        if x2 > 4:
                x2 = x2 -4
        else:
                x2 = x2
        f2 = math.sqrt(1 -(math.sin(abs(x1)))**2) + 2*(x2 - math.sin(abs(x1)) -abs(x1))**2
        return np.asarray([f1, f2])


def MMF9(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        num_of_peak = 2
        temp1 = (math.sin(num_of_peak*math.pi*x2))**6
        g = 2 -temp1

        f1 = x1
        f2 = g/x1
        return np.asarray([f1, f2])


def MMF10(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        g = 2 -math.exp(-((x2 -0.2)/0.004)**2) -0.8*math.exp(-((x2 -0.6)/0.4)**2)
        
        f1 = x1
        f2 = g/x1
        return np.asarray([f1, f2])


def MMF11(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        num_of_peak = 2
        temp1 = (math.sin(num_of_peak*math.pi*x2))**6
        temp2 = math.exp(-2*math.log10(2)*((x2-0.1)/0.8)**2)
        g = 2 -temp2*temp1

        f1 = x1
        f2 = g/x1
        return np.asarray([f1, f2])


def MMF12(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        q = 4
        alpha = 2
        num_of_peak = 2

        f1 = x1

        g = 2 -(math.exp(-2*math.log10(2)*((x2 -0.1)/0.8)**2))*((math.sin(num_of_peak*math.pi*x2))**6)
        h = 1-(f1/g)**alpha -(f1/g)*math.sin(2*math.pi*q*f1)

        f2 = g*h
        return np.asarray([f1, f2])

        
def MMF13(input_array):
        x1 = input_array[0]
        x2 = input_array[1]
        x3 = input_array[2]

        num_of_peak = 2
        t = x2 +math.sqrt(x3)

        f1 = x1

        g = 2 - (math.exp(-2*math.log10(2)*((t -0.1)/0.8)**2))*((math.sin(num_of_peak*math.pi*t))**6)

        f2 = g/x1
        return np.asarray([f1, f2])


def MMF14(input_array):
        """
        varargin = args
        nargin = 1 + len(varargin)

        if nargin <2 :
                m = 3
        if nargin <3:
                num_of_peak =2
        """
        m = 3
        num_of_peak =2
        n = len(input_array)

        k = n-(m-1)

        g = 2 - (math.sin(num_of_peak*math.pi*input_array[(m-1+k)-1]))**2

        cosvalList = []
        sinvalList = []
        for i in range(0, m-1):
                cosval =1.0
                for j in range(0, m-1):
                        cosval = cosval * math.cos((math.pi/2)*input_array[j])
                cosvalList.append(cosval)
                sinval = math.sin((math.pi/2)*input_array[i])
                sinvalList.append(sinval)
                m = m-1
        cosvalList.append(1)
        sinvalList.reverse()
        sinvalList.insert(0, 1)

        f = [a*b for a,b in zip(cosvalList, sinvalList)]
        f = [i*(1+g) for i in f]

        return np.asarray(f)


def MMF14_a(input_array):
        m = 3
        num_of_peak =2
        n = len(input_array)

        k = n-(m-1)

        g = 2 - (math.sin(num_of_peak*math.pi*(input_array[(m-1+k)-1] -0.5*math.sin(math.pi*input_array[(m-2+k)-1]) + (1/(2*num_of_peak)))))**2

        cosvalList = []
        sinvalList = []
        for i in range(0, m-1):
                cosval =1.0
                for j in range(0, m-1):
                        cosval = cosval * math.cos((math.pi/2)*input_array[j])
                cosvalList.append(cosval)
                sinval = math.sin((math.pi/2)*input_array[i])
                sinvalList.append(sinval)
                m = m-1
        cosvalList.append(1)
        sinvalList.reverse()
        sinvalList.insert(0, 1)

        f = [a*b for a,b in zip(cosvalList, sinvalList)]
        f = [i*(1+g) for i in f]

        return np.asarray(f)


def MMF15(input_array):
        m = 3
        num_of_peak =2
        n = len(input_array)

        k = n-(m-1)

        g = 2 - (math.exp(-2*math.log10(2)*((input_array[(m-1+k)-1] -0.1)/0.8)**2))*(math.sin(num_of_peak*math.pi*input_array[(m-1+k)-1]))**2

        cosvalList = []
        sinvalList = []
        for i in range(0, m-1):
                cosval =1.0
                for j in range(0, m-1):
                        cosval = cosval * math.cos((math.pi/2)*input_array[j])
                cosvalList.append(cosval)
                sinval = math.sin((math.pi/2)*input_array[i])
                sinvalList.append(sinval)
                m = m-1
        cosvalList.append(1)
        sinvalList.reverse()
        sinvalList.insert(0, 1)

        f = [a*b for a,b in zip(cosvalList, sinvalList)]
        f = [i*(1+g) for i in f]

        return np.asarray(f)


def MMF15_a(input_array):
        m = 3
        num_of_peak =2
        n = len(input_array)

        k = n-(m-1)

        t = input_array[(m-1+k)-1] -0.5*math.sin(math.pi*input_array[(m-2+k)-1]) +(1/(2*num_of_peak))
        g = 2 - (math.exp(-2*math.log10(2)*((t-0.1)/0.8)**2))*(math.sin(num_of_peak*math.pi*t))**2

        cosvalList = []
        sinvalList = []
        for i in range(0, m-1):
                cosval =1.0
                for j in range(0, m-1):
                        cosval = cosval * math.cos((math.pi/2)*input_array[j])
                cosvalList.append(cosval)
                sinval = math.sin((math.pi/2)*input_array[i])
                sinvalList.append(sinval)
                m = m-1
        cosvalList.append(1)
        sinvalList.reverse()
        sinvalList.insert(0, 1)

        f = [a*b for a,b in zip(cosvalList, sinvalList)]
        f = [i*(1+g) for i in f]

        return np.asarray(f)

def SYM_PART_simple(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        a = 1
        b = 10
        c = 8

        temp_t1 = np.sign(x1)*np.ceil(((abs(x1)) -(a +c/2))/(2*a +c))
        temp_t2 = np.sign(x2)*np.ceil(((abs(x2)) -(b/2))/b)

        t1 = np.sign(temp_t1)*min(abs(temp_t1), 1)
        t2 = np.sign(temp_t2)*min(abs(temp_t2), 1)

        p1 = x1 -t1*(c+2*a)
        p2 = x2 - t2*b

        f1 = (p1 + a)**2 + p2**2
        f2 = (p1 - a)**2 + p2**2

        return np.asarray([f1, f2])



def SYM_PART_rotated(input_array):
        x1 = input_array[0]
        x2 = input_array[1]
        
        w = math.pi/4
        a = 1
        b = 10
        c = 8

        r1 = (math.cos(w))*x1 - (math.sin(w))*x2
        r2 = (math.sin(w))*x1 + (math.cos(w))*x2

        x1 = r1
        x2 = r2

        temp_t1 = np.sign(x1)*np.ceil(((abs(x1) -(a+(c/2))))/((2*a)+c))
        temp_t2 = np.sign(x2)*np.ceil((abs(x2) -(b/2))/b)

        t1 = np.sign(temp_t1)*min(abs(temp_t1), 1)
        t2 = np.sign(temp_t2)*min(abs(temp_t2), 1)

        p1 = x1 - t1*(c +(2*a))
        p2 = x2 - t2*b

        f1 = (p1 +a)**2 + p2**2
        f2 = (p1 -a)**2 + p2**2

        return np.asarray([f1, f2])



def Omni_test(input_array):
        n = len(input_array)

        f1 = 0.0
        f2 = 0.0

        for i in range(0, n):
                f1 = f1 + math.sin(math.pi*input_array[i])
                f2 = f2 + math.cos(math.pi*input_array[i])
        return np.asarray([f1, f2])


def MMF1_z(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        k = 3

        f1 = abs(x1 - 2)
        if (2<= x1 <=3):
                f2 = 1 -math.sqrt(abs(x1 -2)) +2*(x2 -math.sin(2*math.pi*abs(x1-2)+math.pi))**2
        else:
                f2 = 1 -math.sqrt(abs(x1 -2)) +2*(x2 -math.sin(2*k*math.pi*abs(x1-2)+math.pi))**2   

        return np.asarray([f1, f2])


def MMF1_e(input_array):
        x1 = input_array[0]
        x2 = input_array[1]

        a = math.e

        f1 = abs(x1 -2)
        if (2 <= x1 <= 3):
                f2 = 1- math.sqrt(abs(x1 -2)) + 2*(x2 - (a**x1)*math.sin(6*math.pi*abs(x1 -2) +math.pi))**2
        else:
                f2 = 1- math.sqrt(abs(x1 -2)) + 2*(x2 -math.sin(6*math.pi*abs(x1 -2) +math.pi))**2

        return np.asarray([f1, f2])







def evaluate(func, decision,n_obj):
        if func == "MMF1":
                return MMF1(decision)
        elif func == "MMF2":
                return MMF2(decision)
        elif func == "MMF3":
                return MMF3(decision)
        elif func == "MMF4":
                return MMF4(decision)
        elif func == "MMF5":
                return MMF5(decision)
        elif func == "MMF6":
                return MMF6(decision)
        elif func == "MMF7":
                return MMF7(decision)
        elif func == "MMF8":
                return MMF8(decision)
        elif func == "MMF9":
                return MMF9(decision)
        elif func == "MMF10":
                return MMF10(decision)
        elif func == "MMF11":
                return MMF11(decision)
        elif func == "MMF12":
                return MMF12(decision)
        elif func == "MMF13":
                return MMF13(decision)
        elif func == "MMF14":
                return MMF14(decision)
        elif func == "MMF14_a":
                return MMF14_a(decision)
        elif func == "MMF15":
                return MMF15(decision)
        elif func == "MMF15_a":
                return MMF15_a(decision)
        elif func == "SYM_PART_simple":
                return SYM_PART_simple(decision)
        elif func == "SYM_PART_rotated":
                return SYM_PART_rotated(decision)
        elif func == "Omni_test":
                return Omni_test(decision)
        elif func == "MMF1_e":
                return MMF1_e(decision)
        else:
                return MMF1_z(decision)
        