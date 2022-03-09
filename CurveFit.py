#import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit
import FunctionIndex as fn

#only need to import this if wish to use sys.exit() during testing
import sys

#fitting the chosen function to the list of observed (x,y) fitness values
#vector p0 contains the initial parameter values for the function of choice
def fitLearningCurve(function, x, y, p0, lowerBounds, upperBounds):

    numParameters = len(p0)
    curvefitBounds = [lowerBounds,upperBounds]

    try:
        #using curve_fit to fit the x, y values to the given parametric function, specifying initial values and bounds for parameters.
        #curve fit returns the optimized values for fn parameters, as well as the covariance calculations (currently, we don't use covariance)
        param,covariance=curve_fit(function, x, y, p0=p0, bounds=curvefitBounds)
        fittedParameterValues=param

        #print("Curve Fit successful")

        #returning the parameter values of the fitted function
        return fittedParameterValues

    except:
        #Note: For a given CNN it is normal for CurveFit to be unable to find values for the parameters at some iterations. Don't be alarmed if you see this message occasionally.
        print("entered except - CurveFit unable to select values for parameters!")
        
        expFitParameters=[]
        for item in range(numParameters):
            expFitParameters.append(float('inf'))
        print(expFitParameters)
        #sys.exit()
        return expFitParameters


    
    
