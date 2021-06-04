import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import CurveFit
import shutil

#find all DIRECTORIES containing non-hidden files ending in FILENAME
def getDataDirectories(DIRECTORY, FILENAME="valLoss.txt"):
    directories=[]
    for directory in os.scandir(DIRECTORY):
        for item in os.scandir(directory):
            if item.name.endswith(FILENAME) and not item.name.startswith("."):
                directories.append(directory.path)
    return directories

#get all non-hidden data files in DIRECTORY with extension EXT
def getDataFiles(DIRECTORY, EXT='txt'):
    datafiles=[]
    for item in os.scandir(DIRECTORY):
        if item.name.endswith("."+EXT) and not item.name.startswith("."):
            datafiles.append(item.path)
    return datafiles

#checking if loss ever doesn't decrease for numEpochs epochs in a row.
def stopsDecreasing(loss, epoch, numEpochs):
    minLoss=np.inf
    epochMin=0
    for i in range(0,loss.size):
        if loss[i] < minLoss:
            minLoss=loss[i]
            epochMin=epoch[i]
        elif (epoch[i]-epochMin) >= numEpochs:
            return i, minLoss
        
    return i, minLoss

#dirpath is where the accuracy and loss files are stored. want to move the files into the same format expected by grabNNData.
def createFolders(SEARCHDIR, SAVEDIR):
    for item in os.scandir(SEARCHDIR):
        name=str(item.name)
        files=name.split('-')
        SAVEFULLDIR=SAVEDIR+str(files[0])
        if not os.path.exists(SAVEFULLDIR):
            try:
                os.makedirs(SAVEFULLDIR)
            except FileExistsError:
                #directory already exists--must have been created between the if statement & our attempt at making directory
                pass
        shutil.move(item.path, SAVEFULLDIR+"/"+str(files[1]))
    

#a function to read in information (e.g. accuracy, loss) stored at FILENAME
def grabNNData(FILENAME, header='infer', sep=' '):
    data = pd.read_csv(FILENAME, sep, header=header)

    if ('epochs' in data.columns) and ('trainLoss' in data.columns) and ('valLoss' in data.columns) and ('valAcc' in data.columns) and ('batch_size' in data.columns) and ('learning_rate' in data.columns):

        sortedData=data.sort_values(by="epochs", axis=0, ascending=True)

        epoch=np.array(sortedData['epochs'])
        trainLoss=np.array(sortedData['trainLoss'])
        valLoss=np.array(sortedData['valLoss'])
        valAcc=np.array(sortedData['valAcc'])

        batch_size=np.array(sortedData['batch_size'])
        learning_rate=np.array(sortedData['learning_rate'])

        convKers=np.array(sortedData['convKernels'])
        
        return(epoch, trainLoss, valLoss, valAcc, batch_size, learning_rate, convKers)

    elif ('epochs' in data.columns) and ('trainLoss' in data.columns) and ('valLoss' in data.columns) and ('valAcc' in data.columns):

        sortedData=data.sort_values(by="epochs", axis=0, ascending=True)
            
        epoch=np.array(sortedData['epochs'])
        trainLoss=np.array(sortedData['trainLoss'])
        valLoss=np.array(sortedData['valLoss'])
        valAcc=np.array(sortedData['valAcc'])

    else:
        print("Missing a column in NN datafile")
        raise Exception('NN datafile is missing one of the expected columns: epochs trainLoss valLoss valAcc [optional extra columns: batch_size, learning_rate]')
        

#slice data could be used to test values of E other than E=0.5, which we use by default
def sliceData(xsize, x, y, z=None, w=None):
    #we can slice the data to sample less often, but not more often. We verify that we're not being asked for a granularity that is smaller than the frequency of datapoints in the vectors.
    if x[0] > xsize:
        return x,y,z,w
    else:
        result=(1.0/x[0])*xsize
        #result is how often we should take datapoints if we wish to consider values every xsize

        x=x[int(result-1)::int(result)] 
        y=y[int(result-1)::int(result)]

        if z is not None:
            z=z[int(result-1)::int(result)]
            if w is None:
                return x,y,z
        else:
            return x,y

        #if we get to this point in function, it means z and w are both not None.
        w=w[int(result-1)::int(result)]
        return x,y,z,w

