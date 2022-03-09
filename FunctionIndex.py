import numpy as np
import math

#def accDefaultFn(x, a, b, c):
#function has 3 parameters
def accDefaultFn(x,a,b,c):
    return a - b**(c-x)

def accPow2(x,a,b):
    return a*x**b

def accPow3(x,a,b,c):
    return a*x**b+c

def accLog2(x,a,b):
    return a*math.log(x)+b

def accExp2(x,a,b):
    return a * math.e ** (b*x)

def accExp3(x,a,b,c):
    return a * math.e ** (b*x) + c

def accExp4(x,a,b,c,d):
    return c - math.e**(-a*(x**d) + b)

def accExpp3(x,a,b,c):
    return c - math.e**((x-b)**a)

def accExpd3(x,a,b,c):
    return c-(c-a)*math.e**(-b*x)

def accVap3(x,a,b,c):
    return math.e ** (a + b/x + c*math.log(x))

def accMmf4(x,a,b,c,d):
    return (a*b + c*(x**d)) / (b+x**d)

def accWbl4(x,a,b,c,d):
    return a - b*math.e**(-c*x**d)

def accIlog2(x,a,b):
    return b - (a/math.log(x))


#exponential function to model loss
def lossDefaultFn(x, a, b, c):
    return a+b**(c-x)
