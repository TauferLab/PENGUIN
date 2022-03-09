#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataProcessing
import os
import PredictionAnalyzer as analyzer
import CurveFit as cf
import analyzerThresholdFunctions as thresholdFns
import graphPEng4NNPredictions as graph
import matplotlib.pyplot as plt
import argparse
import FunctionIndex as fn
import sys
import csv


parser = argparse.ArgumentParser(description="Simulate PENGUIN asyncronously on a set of trained NNs. For each NN, record the epoch when the prediction stabilizes, as well as the value of the final stable prediction.")

# Adding command line arguments
# directories for loading and saving the data
parser.add_argument('--loadDirectory', required=True, help='The path to the directory storing the txt files of NN learning curve data.')
parser.add_argument('--saveDirectory', required=True, help='The path to the directory where PENGUIN\'s results will be stored.')

# Arguments for PENGUIN's parameters
# In our paper, we used t = 0.5, N = 3, functionType = "accDefaultFn", fitType = "fitAcc", numParameters = 3, lowerBounds = 0.5,1.0,0.0 , upperBounds = 200,inf,inf ,
# fitnessUpperBound = 100, fitnessLowerBound = 0, E = 0.5, cMin = 3, --ePred = 20 for MENNDL, 10 for EvoCNN, 25 for NSGA-Net,

# t is the threshold used by the prediction analyzer. when the variance of the last N predictions is all within threshold t, then the prediction is converged
# N is the number of recent predictions to consider in the prediction analyzer.
# f is the name of the function to use from the list in FunctionIndex.
# fitType is the type of fitness measurement used. For now, options are "accuracy" or "loss".
# numParameters is the number of parameters in the selected function.
# lowerBounds is a comma separated list of bounds for each of the function parameters. Note that the function MUST be defined at the lowerBounds given.
# upperBounds is a comma separated list of bounds for each of the function parameters. Note that the function MUST be defined at the upperBounds given.
# fitnessUpperBound is the largest fitness value that makes sense.
# fitnessLowerBound is the smallest fitness value that makes sense.
parser.add_argument('-t', '--threshold', required=False, help='Threshold value for prediction engine', type=float, default=0.5)
parser.add_argument('-N', '--numToConverge', required=False, help='Number of recent predictions that must stay within the threshold to reach convergence.', type=int, default=3)
parser.add_argument('-f', '--functionType', required=False, help='Name of the function from the list in FunctionIndex (e.g. "accWbl4" or "accPow2"); if no function is specified, a default function will be used.',default="accDefaultFn")
parser.add_argument('--fitType', required=False, help='Type of fitness measurement. If applying PENGUIN asynchronously to NN learning curve dataframes, this should be the name of the column that stores fitness. e.g. "accuracy" or "loss"', default='valAcc')
parser.add_argument('--numParameters', required=False, type=int, help='Number of parameters of the selected function', default=3)
parser.add_argument('--lowerBounds', required=False, help='comma separated list of lower bounds for each of the parameters. The function MUST be defined at the bounds.', default="0.5,1.0,0.0")
parser.add_argument('--upperBounds', required=False, help='comma separated list of upper bounds for each of the parameters. The function MUST be defined at the bounds.', default="200,inf,inf")
parser.add_argument('--fitnessUpperBound', required=False, type=float, help='Largest possible reasonable value for fitness prediction. (e.g., if fitness is measured by accuracy percentage, it should not be greater than 100.)', default=100)
parser.add_argument('--fitnessLowerBound', required=False, type=float, help='Smallest possible reasonable value for fitness prediction. (e.g., if fitness is measured by accuracy percentage, it should not be less than 0.)', default=0)
parser.add_argument('-E', '--epochFrequency', required=False, type=float, help='Number of epochs to elapse per iteration.', default=0.5)
parser.add_argument('--cMin', required=False, type=int, help='Minimum number of datapoints necessary for fitting the function. (i.e. PENGUIN will not attempt to fit a predictive function until we have at least cMin datapoints.)', default=3)
#if augmenting MENNDL ePred=20; for EvoCNN ePred=10; for NSGA-Net ePred=25
parser.add_argument('--ePred', required=False, type=float, help='The epoch for which to predict NN fitness. Usually this will match the number of epochs a NAS trains for by default. (e.g. ePred=20 if augmenting MENNDL or ePred=25 if augmenting NSGA-Net)', default=20)
parser.add_argument('--initialValues', required=False, help='comma separated list of initial values for all parameters', default="10,1.001,100")
parser.add_argument('--nnMetadataColumns', required=False, help='space separated list of names of columns in NN learning curve file that contain data to track. For example "learning_rate batch_size".', default="learning_rate batch_size")
parser.add_argument('--datasetName', required=False, help='Name of the dataset NNs were trained on. e.g.: "CIFAR100" or "MNIST"', default="CIFAR100")

args = parser.parse_args()
DATALOAD_DIRECTORY = args.loadDirectory
DATASAVE_DIRECTORY = args.saveDirectory

threshold, numToConverge = args.threshold, args.numToConverge
print("N ="+str(numToConverge)+" threshold ="+str(threshold))

# getting the class/function associated with the string specified by the user
parametricFnStr = "FunctionIndex."+str(args.functionType)
parametricFn = dataProcessing.get_class(parametricFnStr)
print(parametricFn)

# getting the function associated with the fitness type specified by the user
fitnessType, numParameters = args.fitType, args.numParameters

# using a helper function to construct a list of the parameter bounds from the comma separated strings
lowerBounds, upperBounds = dataProcessing.lowerBounds(numParameters,args.lowerBounds), dataProcessing.lowerBounds(numParameters,args.upperBounds)

fitnessUpBd, fitnessLowBd, epochFrequency, C_min, ePred, learningCurveColumns, DATASET = args.fitnessUpperBound, args.fitnessLowerBound, args.epochFrequency, args.cMin, args.ePred, args.nnMetadataColumns, args.datasetName

initialValues=[]
initialVals = (args.initialValues).split(",")
for item in initialVals:
    initialValues.append(float(item))

#initializing list to store the y-values of the predictive function returned by curve-fit-predict
PREDICTEDFN=[]

#getting a list of all CNN files
datafiles = dataProcessing.getDataFiles(DATALOAD_DIRECTORY)

#calculating the index where we should begin the first prediction.
#cmin is the no. of points required to make a prediction. the cmin-th point occurs at index cmin-1
startIndex = C_min - 1

#directory to save graphs of PEng4NNs predictive functions
graphDir=DATASAVE_DIRECTORY+"graphs"

fitnessPredictionArray=np.array([])
badFile=DATASAVE_DIRECTORY+"1percent"
badModels=np.array([])

#looping through all the CNNs--each item is the filepath for a given CNN's datafile
for item in datafiles:
#for item in ["/CIFAR100/models/2021_02_15_13_53_18_593695.txt"]:
#for item in [datafiles[1]]:            
    #the name of the file is the model name--to get the model ID, we split off just the filename from the full path to the file
    modelName=item.split("/")[-1]
    #print(modelName)

    epoch, fitness, columnData = dataProcessing.grabNNData(item, fitnessTitle=fitnessType, columns=learningCurveColumns, header='infer', sep=' ')

    #if some fitness value is not finite, then there is an error in the datafile.
    #we should skip it & continue to next iteration (i.e. to the next item--that is, CNN--in the datafiles)
    if (not np.isfinite(fitness).all()):
        print("ERROR with file "+str(item))
        continue

    #slicing the data based on how frequently valloss/acc are calculated so that we sample data only every half epoch
    #epoch, valLoss, accuracy=dataProcessing.sliceData(0.5, epoch, y=valLoss, z=accuracy)

    #trainEpoch, trainLoss=dataProcessing.sliceData(1.0, trainEpoch, y=trainLoss)

    '''--! Calculating the epoch when Menndl would terminate training, based on its training loss early termination criterion.'''    
    '''       
    #numEpochsThresh is MENNDL's early termination criterion threshold
    #according to MENNDL's early termination criterion, if min loss does not decrease for 10 epochs, MENNDL will terminate training the CNN.
    #numEpochsThresh=10
    #MENNDLSTOPTRAINING_INDEX,menndlMinTrainLoss = (dataProcessing.stopsDecreasing(trainLoss, trainEpoch, numEpochsThresh))
    #timeMenndlTrainingEnds=trainEpoch[MENNDLSTOPTRAINING_INDEX]
                
    #if (timeMenndlTrainingEnds > 20.0):
    #    timeMenndlTrainingEnds = 20.0
    '''

    #--!CurveFit fitness:
                
    #the maximum number of datapoints PENGUIN has to work with equals (final epoch) x (1/epoch frequency)
    maxDatapoints = int(1/epochFrequency * ePred)

    #For our static testing, PENGUIN should only be allowed to see up to maxDatapoints.
    #However, the NN training data may include more epochs than this.
    #So we truncate epoch, trainLoss, valLoss, accuracy for PENGUIN at maxDatapoints
    epoch = epoch[0:maxDatapoints]
    fitness = fitness[0:maxDatapoints]

    #trainLoss=trainLoss[0:maxDatapoints]
    #valLoss=valLoss[0:maxDatapoints]
    #accuracy=accuracy[0:maxDatapoints]

    #saving actual fitness at the epoch for which we are trying to predict fitness--ePred; this is the ground truth to compare our predictions to
    actualFitness = fitness[-1]
    actualFitnessEpoch = epoch[-1]
    print("actual fitness at epoch "+str(actualFitnessEpoch)+" = "+str(actualFitness))

    #xshift is the multiplier used to normalize the x-values (i.e., the epoch vector)
    #we shift the x-values so that the initial x-value will be 1.0.
    yshift=1.0
    xshift=2.0

    #calling stabilizer & curve fit to predict max acc
    hasStabilized, predictedFitness, parameters, endepochFit, curvefitTrainingIndex, predictedFitnessFunction = analyzer.analyzePrediction(fitnessType, parametricFn, numParameters, initialValues, lowerBounds, upperBounds, fitnessUpBd, fitnessLowBd, fitness, thresholdFns.withinNumberThreshold, epoch, ePred, index=startIndex, threshold=threshold, numToConverge=numToConverge, cutErrorPoints=False, yshift=yshift, xshift=xshift)
    #print("predicted accuracy="+str(aAcc))
    #print("Fitted function: a = "+str(aAcc)+", b = "+str(bAcc)+", c = "+str(cAcc)+"\nFitted fn used xshift "+str(xshift)+" and yshift "+str(yshift))
    paramString=str(parameters[0])
    for item in parameters[1:]:
        paramString = paramString +','+ str(item)
    #parameters=str(parameters[0])+','+str(parameters[1])+','+str(parameters[2])
    #parameters=parameters.replace('\t',',')
    #parameters=parameters.replace(' ',',')
    print("Fitted function parameters: = "+str(paramString)+"\nFitted fn used xshift "+str(xshift)+" and yshift "+str(yshift))

    #to save x & y values of the fn predicted by curve fit predict, save PREDICTEDACCFN to a csv
    PREDICTEDFITNESSFN={'epochs':epoch,'actualFitness': fitness, 'predictedFitnessFunction': predictedFitnessFunction}
    PREDICTEDFITNESSFN=pd.DataFrame(PREDICTEDFITNESSFN)
    #PREDICTEDFITNESSFN.to_csv(directory + "curvePredictedFitnessFn.csv", index=False, header=True)

    #calculating the error in the predicted max acc:
    #numerical error:
    #accError=np.absolute(maxAcc-aAcc)
                
    fitnessError=np.absolute(actualFitness-predictedFitness)
    paramString=paramString.replace(',','-')
    fitnessPredictionData=str(modelName)+" "+str(endepochFit)+" "+str(columnData)+" "+str(paramString)+" "+str(fitnessError)+" "+str(predictedFitness)+" "+str(actualFitness)
    print(fitnessPredictionData)
    fitnessPredictionArray=np.append(fitnessPredictionArray,fitnessPredictionData)

#saving the PENGUIN accuracy data
FITNESSARRAY=pd.DataFrame(fitnessPredictionArray)
FITNESSARRAY.columns=["modelID timeStabilized "+ str(learningCurveColumns) +" parameterValues error predictedFitness actualFitness",]
FITNESSARRAY.to_csv(path_or_buf=str(DATASAVE_DIRECTORY)+DATASET+"-PENGUIN-MODULAR-TEST-PEng4NN_results-noLossCheck-"+str(ePred)+"epochs-N_"+str(numToConverge)+"-t_"+str(threshold)+".csv", sep="\n", index=False, header=True)

fitnessPredictionArray=np.array([])