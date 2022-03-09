import CurveFit as CF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analyzerThresholdFunctions as thresholdFns
import FunctionIndex

#--! analyzePrediction is the function responsible for calling CurveFit to get predictions and then calling the thresholdFunction to determinine if the predictions have converged or not.
# It takes the following arguments (many of these arguments are parameters of PEng4NN, as read from the `PEng4NN-input_file` in `PEng4NN_Main`, or function selections from `CurveFit` or `stabilizerThresholdFunction` scripts): 

# fitnessType = name of fitness measurement being used,
# fnType = function we are fitting to, either `expLossFn` of `expAccFn` from our CurveFit script,
# thresholdFunction = choice of threshold function from stabilizerThresholdFunctions,
# epoch, valLoss, accuracy = arrays containing these datapoints for the NN,
# index = index of where to start fitting iterations (since we are fitting exponential functions, we do not begin fitting until we have at least three datapoints; the third point is at index 2, thus index=2),
# threshold = threshold value argument for the thresholdFunction,
# numToConverge = number of points that must all stay within threshold of each other in order to consider the prediction converged,
# cutErrorPoints toggles the functionality that removes some initial data points if curve fit cannot fit to the data.
# yshift and xshift are optional arguments for how much to shift the x or y values before fitting. if shifted, they will be unshifted before returning the fitted function.
# neverLearnAccMarker = a value such that if the predicted max accuracy is at or below this value we say the NN never learns, 
# NOTE: When we call stabilizePrediction in PEng4NN_Main, we provide an argument for xshift such that, after the shift, the x values start at 1.

def analyzePrediction(fitnessType, fnType, numParameters, initialValues, lowerBounds, upperBounds, fitnessUpBd, fitnessLowBd, fitness, thresholdFunction, epoch, ePred, index=2, threshold=0.0, numToConverge=5, cutErrorPoints=False, yshift=1.0, xshift=1.0, neverLearnAccMarker=10.0):
    #i keeps track of the iteration/index we are on, starting with the value of index
    i=index

    #startindex is the index of the first datapoint to consider. this will always be 0 unless cutErrorPoints is enabled
    startindex=0

    #aValues stores the last numToConverge predicted a values to give to the threshold function, which determines whether the prediction has converged
    aValues=np.array([])

    #multiplying x and y by the chosen shifts
    curveFitX=epoch*xshift
    curveFitY=fitness*yshift
    
    #The variable `converged` tells us whether or not the prediction has converged. It is possible for the prediction process to finish without converging. 
    #Variable `done` will be set to true if the prediction meets the convergence criteria, or if we have used all the data points and still have been unable to reach a converged prediction.
    #Ideally, we want converged and done both to be true at the end of the process, indicating that we have finished because the prediction converged.
    converged=False
    done=False
    
    #initializing the prediction function values as an empty array
    predictedFn=np.array([])

    while done == False:
        #print(startindex, i)

        #use CurveFit to fit the data to an exponential function, giving it the data points from startindex up through the current iteration i
        fnParameters=CF.fitLearningCurve(fnType, curveFitX[startindex:i], curveFitY[startindex:i], initialValues, lowerBounds, upperBounds)
        print(str(fnParameters))
        predictedFn=fnType(curveFitX,*fnParameters)
        #predictedFn=fnType(curveFitX,fnParameters)
        #predictedFn=fnType(curveFitX,a,b,c)

        #calculating the fitness predicted for epoch ePred
        predictedFitness=fnType(ePred*xshift,*fnParameters)
        predictedFitness=predictedFitness*1.0/yshift

        #unshifting the y-values of the prediction
        predictedFn=predictedFn*1.0/yshift
        #a=a*1.0/yshift


        #if we have used all the datapoints trying to get the prediction, we are done, but the prediction has not converged. We set predictedFitness = actual fitness, and we set converged to False.
        if i+1 > np.shape(curveFitX)[0]:
            done=True
            converged=False
            predictedFitness = fitness[-1]
            print("tried max number of points.")
            break

        # Otherwise, we have not used all datapoints, so we check if the prediciton has converged.
        # If CurveFit is unable to select values for fn parameters, then they are all set to inf. 
        # If the first parameter or the predictedFitness is infinite, that means we did not get a prediction from CurveFit at this iteration, thus we do not save a new predictedFitness value.
        # Also, if argument `cutErrorPoints` is True, we increase the startindex to exclude these points.
        if (not np.isfinite(fnParameters[0])) or (not np.isfinite(predictedFitness)):
            #errorIndices=np.append(errorIndices,i)
            #index=errorIndices.shape[0]

            #uncomment below if we wish to reset (i.e. remove any entries in) aValues since we did not get a new prediction this iteration
            #aValues=np.array([])

            #if cutErrorPoints is enabled, startindex = i ensures we cut any previous datapoints so that CurveFit only sees points starting with the datapoint calculated in this iteration
            #this can be helpful if CurveFit consistently cannot find values for a, b, c because of initial noisy or jumpy data.
            if cutErrorPoints==True:
                startindex=i
                i+=1

        #If prediction is finite, we append fitness prediction to the list of the last few predicted fitness values.
        #elif (fnParameters[0]<= lowerBounds[0] + epsilon) or (fnParameters[1]<= lowerBounds[0] + epsilon) or (fnParameters[2]<= lowerBounds[0] + epsilon):
        #    aValues=np.array([])
        #elif predictedFitness > 100.0:
        #    aValues=np.array([])
        elif (predictedFitness > fitnessUpBd) or (predictedFitness < fitnessLowBd):
            aValues=np.array([])
            done=False
        else:
            aValues=np.append(aValues,predictedFitness)

            #uncomment this bit if we want to throw out our history of accuracy predictions if a is > 100.
            #if (a > 100.0):
                #aValues=np.array([])

            #Now, we check if our prediction from CurveFit has converged:
            #--! Has it converged?
            # 1. Are there enough recent predictions to test if it has converged?
            # 2. Are all numToConverge of the recent predictions within threshold of their median?
            # 3. Is predicted acc <= 100 and predicted acc > 0? If not, it is not a realistic prediction.
            #
            # If yes to 1, 2, 3: the prediction has met convergence criteria

            #1:
            if aValues.shape[0] >= numToConverge:
                done=True

                #2: with the chosen threshold function, percentage or number
                #if threshold is 0, we were not given a threshold argument to use--this means the chosen thresholdFunction must be one of the programmatic ones that does not have threshold as an argument
                if threshold==0:
                    done=thresholdFunction(aValues)

                #otherwise, the thresholdFunction takes both threshold and aValues as arguments
                else:
                    done=thresholdFunction(threshold, aValues)
                
                if done:
                    print(aValues)
                aValues=np.delete(aValues,0)        

            #3:
            #if ((predictedFitness > fitnessUpBd) or (predictedFitness < fitnessLowBd)):
            #if (predictedFitness > fitnessUpBd):
            #    done=False
            #elif (predictedFitness < fitnessLowBd):
            #    done=False

            if done==True:
                #if done is true at this point, it is because the prediction met convergence criteria 1 - 3.
                converged=True
                break

            #If we have gotten this far, it means that converged = False, done = False, and we have not yet used all datapoints. We increment i and begin the next iteration.
        i+=1
    
    #if converged:
    #    status="Prediction converged."
    #else:
    #    status="Prediction used all datapoints and did not converge."

    endEpoch=epoch[i-1]

    return converged, predictedFitness, fnParameters, endEpoch, i-1, predictedFn
    #converged = bool value indicating whether curvefit sucessfully converged
    #a,b,c = the values for a,b,c in the predicted exponential function
    #endEpoch = the last epoch (x value) of the final datapoint used by curvefitpredict when it converged
    #predictedFn = the y-values given by the predicted function at each epoch in the x-values.
