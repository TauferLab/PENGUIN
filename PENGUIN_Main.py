#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataProcessing
import os
import PredictionAnalyzer as analyzer
import CurveFit as cf
import analyzerThresholdFunctions as thresholdFns
import matplotlib.pyplot as plt

print("Starting PENGUIN!")

#update to wherever the PENGUIN configuration input file is saved
CONFIG_FILE_PATH="PENGUIN-input_file.txt"

CONFIGURATIONS=pd.read_csv(CONFIG_FILE_PATH, sep=' ', header=0, index_col=False)
print(CONFIGURATIONS)

#dataset is the benchmark dataset used for training the CNNs PENGUIN is running on
DATASET=CONFIGURATIONS["DATASET"][0]

#num to converge is the number of recent curve-fit predictions that will be required to meet the convergence criteria in order for the prediction to converge
#lossNumToConverge=CONFIGURATIONS["N"][0]
#accNumToConverge=CONFIGURATIONS["N"][0]
numToConverge=CONFIGURATIONS["N"][0]

epochFrequency=CONFIGURATIONS["E"][0]

#e_max is the maximum number of epochs PENGUIN will use.
e_max=CONFIGURATIONS["e_max"][0]

#threshold=maximum variability allowed for convergence
threshold=CONFIGURATIONS["threshold"][0]

# --- LOSS CHECK ---
# a loss check can be used to disallow stabilization if the loss is still decreasing.
# it may be helpful for especially noisy datasets.
# we have not used a loss check in our results, but we have built it in as a possibility in the future.
#lossCheck = boolean to toggle on or off a loss check
lossCheck = CONFIGURATIONS["loss_check"][0]
#epoch threshold for the loss check
lossCheckEpochThreshold = CONFIGURATIONS["L"][0]
# --- END LOSS CHECK SEGMENT ---

#initializing list to store the y-values of the predictive function returned by PENGUIN's curve fit
PREDICTEDFN=[]

#directory where the CNN data txt files are stored
DATAFILES_DIRECTORY = CONFIGURATIONS["DATA_DIRECTORY"][0]

#getting a list of all CNN files
datafiles = dataProcessing.getDataFiles(DATAFILES_DIRECTORY)

#directory where PENGUIN data about all the CNNs will be saved
summaryDataDir = CONFIGURATIONS["DATA_SUMMARY_SAVEPATH"][0]

#initializing the array to store PENGUIN's accuracy predictions
maxValAcc_Predictions=np.array([])

#looping through all the CNNs--each item is the filepath for a given CNN's datafile
for item in datafiles:
#for item in "/Users/ariel/Downloads/CIFAR-100_models/2021_02_15_14_01_32_105932.txt",:
    print(item)
    #the name of the file is the model name--to get the model ID, we split off just the filename from the full path to the file
    modelName=item.split("/")[-1]
    print(modelName)

    #epoch, trainLoss, valLoss, accuracy = dataProcessing.grabNNData(item, header='infer', sep=' ')
    epoch, trainLoss, valLoss, accuracy, batch_size, learning_rate, convKers = dataProcessing.grabNNData(item, header='infer', sep=' ')

    numConvolutions = len(str(convKers[0]).split("-"))

    #thus far in PEng4NN we save trainLoss valLoss and accuracy with the same frequency. 
    #if in the future, we save training loss more frequently that validation data, we would need different epoch vectors for training and validation.
    #however, for now, trainEpoch and validationEpoch are both the epoch vector returned by dataProcessing.grabNNData.
    trainEpoch = epoch

    #if some loss or accuracy values are not finite, then we skip this file & continue to next iteration (i.e. to the next item--that is, CNN--in the datafiles)
    if (not np.isfinite(valLoss).all()) or (not np.isfinite(accuracy).all()):
        print("ERROR with file "+str(item))
        continue

    '''--! Calculating the epoch when the NAS would terminate training, based on its training loss early termination criterion.'''
            
    #For the sake of comparison, we calculate the time MENNDL would terminate training each NN according to it's built-in termination criterion
    #numEpochsThresh is MENNDL's early termination criterion threshold
    #according to MENNDL's early termination criterion, if min loss does not decrease for 10 epochs, MENNDL will terminate training the CNN.
    numEpochsThresh=10
    MENNDLSTOPTRAINING_INDEX,menndlMinTrainLoss = (dataProcessing.stopsDecreasing(trainLoss, epoch, numEpochsThresh))
    timeMenndlTrainingEnds=trainEpoch[MENNDLSTOPTRAINING_INDEX]
    
    #MENNDL trains for a maximum of 20 epochs, so if the loss threshold would not stop it until after 20 epochs, MENNDL would stop at 20 epochs
    if (timeMenndlTrainingEnds > 20.0):
        timeMenndlTrainingEnds = 20.0

    #--!CurveFit Accuracy:    
    print("PENGUIN fitting to predict ACCURACY")

    #uncomment below to grab the max observed accuracy index and value:
    #get index of the maximum accuracy
    #MAXACC_INDEX = np.argmax(accuracy)
    #get the value of the maximum accuracy
    #maxAcc = accuracy[MAXACC_INDEX]

    #in comparing to the NAS, we require PENGUIN to terminate at e_max 20 epochs if it does not terminate earlier than that.
    totalNumEpochs = e_max
    maxDatapoints = int(1/epochFrequency * totalNumEpochs)

    #truncating epoch, trainLoss, valLoss, accuracy to e_max epochs worth of datapoints
    epoch = epoch[0:maxDatapoints]
    trainLoss=trainLoss[0:maxDatapoints]
    valLoss=valLoss[0:maxDatapoints]
    accuracy=accuracy[0:maxDatapoints]

    # this is the actual accuracy at the epoch we are trying to predict accuracy; 
    # this is the ground truth to compare our predicitons to
    actualAcc = accuracy[-1]
    actualAccEpoch = epoch[-1]
    print("actual acc at epoch "+str(actualAccEpoch)+" = "+str(actualAcc))

    #get index of the observed maximum accuracy across just the maxDatapoints used by PENGUIN
    truncated_MAXACC_INDEX = np.argmax(accuracy)
    #get the value of the observed maximum accuracy across just the maxDatapoints used by PENGUIN
    truncated_maxAcc = accuracy[truncated_MAXACC_INDEX]

    yshift=1.0
    xshift=2.0
    #calling stabilizer & curve fit to predict max acc
    stabilizedAcc, predictedAcc, aAcc, bAcc, cAcc, endepochAcc, curvefitAccTrainingIndex, predictedfnAcc = analyzer.analyzePrediction(cf.fitAcc, cf.expAccFn, thresholdFns.withinNumberThreshold, epoch, totalNumEpochs, valLoss, accuracy=accuracy, index=2, threshold=threshold, numToConverge=numToConverge, cutErrorPoints=False, yshift=yshift, xshift=xshift, lossCheck=lossCheck, neverLearnAccMarker=19.61, lossCheckEpochThreshold=lossCheckEpochThreshold)
    print("predicted accuracy="+str(aAcc))
    print("Fitted function: a = "+str(aAcc)+", b = "+str(bAcc)+", c = "+str(cAcc)+"\nFitted fn used xshift "+str(xshift)+" and yshift "+str(yshift))

    #if want to see the x,y values of the fn predicted by curve fit predict, save PREDICTEDACCFN to a csv
    PREDICTEDACCFN={'epochs':epoch,'actualAccuracy': accuracy, 'predictedAccFunction': predictedfnAcc}
    PREDICTEDACCFN=pd.DataFrame(PREDICTEDACCFN)
    #PREDICTEDACCFN.to_csv(directory + "curvePredictedValAccFn.csv", index=False, header=True)

    #calculating the error in the predicted max acc:
    #accError=np.absolute(maxAcc-aAcc)
    accError=np.absolute(actualAcc-predictedAcc)
    #maxValAcc_Error=str(modelName)+" "+str(numConvolutions)+" "+str(learning_rate[0])+" "+str(batch_size[0])+" "+str(endepochAcc)+" "+str(timeMenndlTrainingEnds)+" "+str(truncated_maxAcc)+" "+str(aAcc)+" "+str(accError)+" "+str(predictedAcc)+" "+str(actualAcc)
    #maxValAcc_Errors=np.append(maxValAcc_Errors,maxValAcc_Error)
    
    maxValAcc_Prediction=str(modelName)+" "+str(numConvolutions)+" "+str(learning_rate[0])+" "+str(batch_size[0])+" "+str(endepochAcc)+" "+str(timeMenndlTrainingEnds)+" "+str(truncated_maxAcc)+" "+str(aAcc)+" "+str(accError)+" "+str(predictedAcc)+" "+str(actualAcc)
    maxValAcc_Predictions=np.append(maxValAcc_Predictions,maxValAcc_Prediction)

    ACCPREDICTIONS=pd.DataFrame(maxValAcc_Predictions)
    ACCPREDICTIONS.columns=["modelID numConvLayers learningRate batchSize timeStabilized timeMenndlTrainingEnds maxAcc aValue error predictedAcc actualAcc",]

    #saving PENGUIN's predictions, metadata about the NNs, the actual acc values, and the prediction error in a csv file.
    ACCPREDICTIONS.to_csv(path_or_buf=str(summaryDataDir)+DATASET+"-PENGUIN_results-N_"+str(numToConverge)+"-t_"+str(threshold)+".csv", index=False, header=True)

