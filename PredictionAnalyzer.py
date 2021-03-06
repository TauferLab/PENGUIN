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

def analyzePrediction(fitMethod, fnType, thresholdFunction, epoch, finalEpoch, valLoss, accuracy=[], index=2, threshold=0.0, numToConverge=3, cutErrorPoints=False, yshift=1.0, xshift=1.0, lossCheck=False, neverLearnAccMarker=10.0, lossCheckEpochThreshold=5):
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

        # use CurveFit to fit the data to an exponential function, 
        # giving it the data points from startindex up through the current iteration i
        a,b,c=fitMethod(curveFitX[startindex:i], curveFitY[startindex:i])
        predictedFn=fnType(curveFitX,a,b,c)

        predictedAcc=fnType(finalEpoch*xshift,a,b,c)
        predictedAcc=predictedAcc*1.0/yshift

        # unshifting the y-values of the prediction
        predictedFn=predictedFn*1.0/yshift
        #a=a*1.0/yshift

        #if we have used all the datapoints trying to get the prediction, we are done, but the prediction has not converged. We set predictedAcc = actual accuracy, and we set converged to False.
        if i+1 > np.shape(curveFitX)[0]:
            done=True
            converged=False
            if accuracy == []:
                predictedAcc=valLoss[np.argmin(valLoss)]
            else:
                predictedAcc = accuracy[-1]
                #predictedAcc=accuracy[np.argmax(accuracy)]
            print("tried max number of points.")
            break


        # Otherwise, we have not used all datapoints, so we check if the prediction has converged.
        # If CurveFit is unable to select values for a, b, c, then a, b, c are set to inf. 
        # Hence, if a or predictedAcc is infinite, that means we did not get a prediction from CurveFit at this iteration, thus we do not save a new prediction.
        if (not np.isfinite(a)) or (not np.isfinite(predictedAcc)):
            done=False
            converged=False

            #if cutErrorPoints is enabled, startindex = i ensures we cut any previous datapoints so that CurveFit only sees points starting with the datapoint calculated in this iteration
            #this can be helpful if CurveFit consistently cannot find values for a, b, c because of initial noisy or jumpy data.
            if cutErrorPoints==True:
                startindex=i
                i+=1

        # if accuracy prediction is finite but a is on the boundary of allowed values,
        # do not allow a prediction to be saved, and clear the prediction list.
        elif (a>=200.0-0.01):
            aValues=np.array([])

        # if a is not on the boundary and the accuracy prediction is finite, 
        # we append the accuracy prediction to the list of the last few predicted accs.
        else:
            aValues=np.append(aValues,predictedAcc)

            #Now, we check if our prediction has converged:
            #--! Has it converged?
            # 1. Are there enough recent predictions to test if it has converged?
            # 2. Are all numToConverge of the recent predictions within threshold of their median?
            # 3. Is predicted accuracy <= 100 and > 0? If not, it is not a realistic prediction.
            #
            # If loss check is enabled: 
            #      4. If the NN is predicted never to learn, has the minimum loss not decreased for at least lossCheckEpochThreshold number of epochs?
            #
            # If yes to 1, 2, 3, 4: the prediction has met convergence criteria

            # Checking 1:
            if aValues.shape[0] >= numToConverge:
                #has passed condition 1, set done to True.
                done=True

                # Checking 2, for the chosen threshold function--percentage threshold or number threshold
                # thresholdFunction tests if the values are within the threshold and returns the result as a boolean.
                # done will be set to True if test 2 is passed, False if test 2 is failed

                # if threshold is 0, we were not given a threshold argument to use--this means the chosen thresholdFunction must be one of the programmatic ones that does not have threshold as an argument
                if threshold==0:
                    done=thresholdFunction(aValues)

                #otherwise, the thresholdFunction takes both threshold and aValues as arguments
                else:
                    done=thresholdFunction(threshold, aValues)
                
                if done:
                    print(aValues)
                aValues=np.delete(aValues,0)        

            #Checking 3. If fails test 3, done is set to False:
            if (predictedAcc > 100.0):
                done=False
            elif (predictedAcc <= 0):
                done=False

            if done==True:
                #if done is true at this point, it is because the prediction met convergence criteria 1 - 3.
                #4: if we are meant to implement a loss check, we do that now. Otherwise, we set converged to True and break from the loop.
                if (lossCheck):
                    #we only need to use the loss check on NNs that are expected not to learn
                    if predictedAcc <= neverLearnAccMarker:
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

            #If we have gotten this far, it means that converged = False, done = False, and we have not yet used all datapoints.
            # We increment i and begin the next iteration.
        i+=1
    
    #if converged:
    #    status="Prediction converged."
    #else:
    #    status="Prediction used all datapoints and did not converge."

    endEpoch=epoch[i-1]

    return converged, predictedAcc, a, b, c, endEpoch, i-1, predictedFn
    #converged = bool value indicating whether curvefit sucessfully converged
    #a,b,c = the values for a,b,c in the predicted exponential function
    #endEpoch = the last epoch (x value) of the final datapoint used by curvefitpredict when it converged
    #predictedFn = the y-values given by the predicted function at each epoch in the x-values.
