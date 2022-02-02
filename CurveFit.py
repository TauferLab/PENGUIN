#import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#exponential function to model accuracy
def expAccFn(x, a, b, c):
    return a-b**(c-x)

#fitting the accuracy exponential function to the list of observed (x,y) accuracy values
def fitAcc(x,y):
    #initial values for a,b,c:
    p0=[10, 1.001, 100]
    
    #lower bounds for (a, b, c) are (0.5, 1.0, 0.0) respectively
    #upper bounds for (a, b, c) are (200.0, infinity, infinity) respectively
    curvefitBounds=[(0.5,1.0,0.0),(200.0, np.inf, np.inf)]

    try:
        #using curve_fit to fit the x, y values to the function given by expAccFn, specifying initial values and bounds for a,b,c.
        param=curve_fit(expAccFn, x, y, p0=p0, bounds=curvefitBounds)
        
        expFitParameters=param[0]
        
        a=expFitParameters[0]
        b=expFitParameters[1]
        c=expFitParameters[2]
        #print("fitAccExp: a = "+str(a)+" b = "+str(b)+" c = "+str(c))

        print("Curve Fit successful")

        #returning the max accuracy prediction and the vector of y values of the fitted function
        return(a, b, c)

    except:
        #Note: For a given CNN it is normal for CurveFit to be unable to find values for a, b, c at some iterations. Don't be alarmed if you see this message occasionally.
        print("CurveFit unable to select a, b, c!")
        return(float('inf'),float('inf'),float('inf'))
