import CurveFit as CF
import numpy as np
import pandas as pd

#This script defines different threshold functions. To change which threshold function is used, update which of these functions is given as an argument to stabilizePrediction in PEng4NN_Main.

#this function checks if the values in the array are all within threshold value of their median
def withinNumberThreshold(threshold, array):
    done=True
    median = np.median(array)
    for j in array:
        if (j>median + threshold) or (j<median - threshold):
            done=False
    return done

#this function checks if the values in the array are all within threshold percent of their median
def withinPercentThreshold(threshold, array):
    done=True
    median = np.median(array)

    for j in array:
        if (j>median + (median*threshold/100.0)) or (j<median - (median*threshold/100.0)):
            done=False
    return done


# NOTE: the next function, 'programmaticAccThreshold', is a placeholder example of a programmatic threshold function.
# Determine the range/function you want for threshold and update the indicated lines.
def programmaticAccThreshold(array):
    done=True
    median = np.median(array)

    '--*BEGIN: EDIT HERE TO DEFINE A PROGRAMMATIC ACCURACY THRESHOLD--*'
    # Using the example values below, threshold should range from .1% to 5%, .1% when 90% accuracy & 5% when 2% accuracy.
    threshold=(-1.9/85)*float(median)+2.0+(5.0*(1.9/85))

    if threshold < 0.1:
        threshold = 0.1
    elif threshold > 5.0:
        threshold = 5.0
    '--*END: EDIT HERE TO DEFINE A PROGRAMMATIC ACCURACY THRESHOLD--*'

    for j in array:
        if (j>median + (median*threshold/100.0)) or (j<median - (median*threshold/100.0)):
            done=False
    return done