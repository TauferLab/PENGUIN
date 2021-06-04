import CurveFit as CF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analyzerThresholdFunctions as thresholdFns

#--! analyzePrediction is the function responsible for calling CurveFit to get predictions and then calling the thresholdFunction to determinine if the predictions have converged or not.
# It takes the following arguments (many of these arguments are parameters of PENGUIN, as read from the `PENGUIN-input_file` in `PENGUIN_Main`, or function selections from `CurveFit` or `stabilizerThresholdFunction` scripts): 

# fitMethod = choice of `fitAcc` or `fitLoss` from our CurveFit script,
# fnType = function we are fitting to, i.e. `expAccFn` from our CurveFit script,
# thresholdFunction = choice of threshold function from stabilizerThresholdFunctions, i.e. `withinNumberThreshold`
# epoch, valLoss, accuracy = arrays containing these datapoints for the NN,
# index = index of where to start fitting iterations (since we are fitting exponential functions, we do not begin fitting until we have at least three datapoints; the third point is at index 2, thus index=2),
# threshold = threshold value argument for the thresholdFunction,
# numToConverge = number of points that must all stay within threshold of each other in order to consider the prediction converged, given by N in `PENGUIN-input_file.txt`
# cutErrorPoints toggles the functionality that removes some initial data points if curve fit cannot fit to the data.
# yshift and xshift are optional arguments for how much to shift the x or y values before fitting. if shifted, they will be unshifted before returning the fitted function.
# lossCheck = a boolean describing whether or not to check if minimum loss has decreased before allowing the prediction to converge,
# neverLearnAccMarker = a value such that if the predicted max accuracy is at or below this value we say the NN never learns, 
# lossCheckEpochThreshold = if the loss check is enabled, this argument is the number of epochs that minimum loss must not decrease before allowing the prediction to converge.
# NOTE: When we call stabilizePrediction in PENGUIN_Main, we provide an argument for xshift such that, after the shift, the x values start at 1.

def analyzePrediction(fitMethod, fnType, thresholdFunction, epoch, valLoss, accuracy=[], index=2, threshold=0.0, numToConverge=3, cutErrorPoints=False, yshift=1.0, xshift=1.0, lossCheck=False, neverLearnAccMarker=10.0, lossCheckEpochThreshold=5):
    #i keeps track of the iteration/index we are on, starting with the value of index
    i=index

    #startindex is the index of the first datapoint to consider. this will always be 0 unless cutErrorPoints is enabled
    startindex=0

    #aValues stores the last numToConverge predicted a values to give to the threshold function, which determines whether the prediction has converged
    aValues=np.array([])

    #multiplying x and y by the chosen shifts
    curveFitX=epoch*xshift
    curveFitY=accuracy*yshift
    
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
        a,b,c=fitMethod(curveFitX[startindex:i], curveFitY[startindex:i])
        predictedFn=fnType(curveFitX,a,b,c)

        #unshifting the y-values of the prediction
        predictedFn=predictedFn*1.0/yshift
        a=a*1.0/yshift

        # If CurveFit is unable to select values for a, b, c, then a, b, c are set to inf. 
        # Hence, if a is infinite, that means we did not get a prediction from CurveFit at this iteration, thus we do not save a.
        if (not np.isfinite(a)):
            
            #if we have used all the datapoints trying to get the prediction and CurveFit has still errored, we are done, but the prediction has not converged. We set a = max observed accuracy, since we don't have a converged prediction
            elif i+1 > np.shape(curveFitX)[0]:
                done=True
                converged=False
                a=accuracy[np.argmax(accuracy)]
                print("tried max number of points.")
                break

        #If a is finite, we append it to the list of the last few predicted "a"s.
        else:
            aValues=np.append(aValues,a)

            #Now, we check if our prediction has converged:
            #--! Has it converged?
            # 1. Are there enough recent predictions to test if it has converged?
            # 2. Are all numToConverge of the recent predictions within threshold of their median?
            # 3. Is predicted a <= 100 and a > 0? If not, it is not a realistic prediction.
            #
            # If loss check is enabled: 
            #      4. If the NN is predicted never to learn, has the minimum loss not decreased for at least lossCheckEpochThreshold number of epochs?
            #
            # If yes to 1, 2, 3, 4: the prediction has met convergence criteria

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
            if (a > 100.0):
                done=False
            elif (a <= 0):
                done=False

            if done==True:
                #if done is true at this point, it is because the prediction met convergence criteria 1 - 3.
                #4: if we are meant to implement a loss check, we do that now. Otherwise, we set converged to True and break from the loop.
                if (lossCheck):
                    #we only need to use the loss check on NNs that are expected not to learn
                    if a <= neverLearnAccMarker:
                        #print("Using loss check on never learn")

                        #minLossIndex is the index of the minimum loss seen so far at iteration i
                        minLossIndex = np.argmin(valLoss[0:i])

                        #getting the epoch when the min loss occured
                        minLossEpoch = epoch[minLossIndex]

                        #getting epoch of current iteration
                        currentEpoch = epoch[i]

                        #if the minLossEpoch is further away from the currentEpoch than the loss check threshold, the NN passes the loss check convergence criterion
                        if (currentEpoch - minLossEpoch) >= lossCheckEpochThreshold:
                            converged = True
                            break
                        else:
                            done = False
                            #print("not done")
                else:    
                    converged=True
                    break

            #If we have gotten this far, it means that converged = False, done = False.
            #However, if we have used all the datapoints trying to get the prediction, we are done, but the prediction has not converged. We set a = max observed accuracy or min observed loss, since we don't have a converged prediction
            if i+1 > np.shape(curveFitX)[0]:
                done=True
                converged=False
                if accuracy == []:
                    a=valLoss[np.argmin(valLoss)]
                else:
                    a=accuracy[np.argmax(accuracy)]
                print("tried max number of points.")
                break 
         
        i+=1
    
    #if converged:
    #    status="Prediction converged."
    #else:
    #    status="Prediction used all datapoints and did not converge."

    endEpoch=epoch[i-1]

    return converged, a, b, c, endEpoch, i-1, predictedFn
    #converged = bool value indicating whether curvefit sucessfully converged
    #a,b,c = the values for a,b,c in the predicted exponential function
    #endEpoch = the last epoch (x value) of the final datapoint used by curvefitpredict when it converged
    #predictedFn = the y-values given by the predicted function at each epoch in the x-values.
