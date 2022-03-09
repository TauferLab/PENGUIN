### Running PENGUIN
See the instructions in the "TestRun.md" file to run a small test of PENGUIN. 

PENGUIN is currently designed to run statically for testing, using data from trained CNNs. The CNNs we used in our tests are available in our Harvard Dataverse repository: https://doi.org/10.7910/DVN/ZXTCGF. In order to replicate these tests, the user can download our CNN dataset, storing all CNNs for a given benchmark dataset in a common folder. The user should add the folder where the CNNs are stored to `PENGUIN-input_file.txt` under `Data_Directory` and update `Data_Summary_Savepath` to the directory where they wish the results of PENGUIN's run to be saved. Then, running `PENGUIN_Main` will execute PENGUIN on the CNNs stored in the `Data_Directory` folder and save the results in `Data_Summary_Savepath`.

### PENGUIN_Main
`PENGUIN_Main` begins by reading an input file (`PENGUIN-input_file.txt`) that configures PENGUIN's parameters. For a given CNN, its complete training and validation info, together with all parameters defining the CNN, are loaded from a txt file named with that CNN's ID. PENGUIN_Main iterates through the text files for all CNNs on a given benchmark dataset, and simulates PENGUIN's iterative process for each CNN. 

### PENGUIN-input_file

`PENGUIN_Main.py` reads an input file (`PENGUIN-input_file.txt`) that lists the configurations to use for all of PENGUIN's parameters. If you wish to change parameter values, edit this file. See below a table describing each of the parameters.

| Parameter | Description |
| --------- | ----------- |
| DATASET | the benchmark dataset used to train the CNNs on which PENGUIN will be run. (i.e. `CIFAR100`, `FashionMNIST`, or `SVHN`)|
| loss_or_accuracy | String indicating if PENGUIN should predict loss or accuracy. In the future we will expand functionality to predict loss. Currently we only predict accuracy. (use `loss` or `accuracy`) |
| N | Number of most recent accuracy predictions to consider in the analysis of convergence. We use `N = 3`. |
| E | Number of epochs per iteration. We use `E = 0.5`. |
| e_max | Maximum number of epochs PENGUIN trains NNs. |
| threshold | max variability allowed for convergence. We use `t = 0.5`. |
| C_min | Minimum number of datapoints to fit the function to. (i.e. PENGUIN will not attempt to fit a predictive function until we have at least C_min datapoints.) We use `C_min = 3`. |
| loss_check | Indicates whether the loss check is enabled. We use `loss_check = False` for all our tests, but are developing the loss check mechanism for potential use with noisy and unbalanced data. |
| L | Number of epochs of data to consider in the loss check. We use `L = 5`. |
| DATA_DIRECTORY | path to directory where the txt files containing the CNNs' data are stored |
| DATA_SUMMARY_SAVEPATH | path to directory where PENGUIN's results on all the CNNs should be saved |

### PredictionAnalyzer

This script contains the function `analyzePrediction`, which is responsible for iteratively calling the `fitAcc` function from `CurveFit` to get the predictive accuracy function at each iteration (i.e. with one more datapoint), and determining if the prediction at the current iteration has converged (with the help of the threshold function of choice from `analyzerThresholdFunctions`). For a given CNN, it terminates the iterative prediction process as soon as a prediction converges.

### CurveFit

This script defines the functions that we fit to the CNN accuracy data. It defines our function that uses Sci-Py's curve_fit to solve for the fitted function parameters and includes specifications of our initial values and bounds for the function parameters.

### analyzerThresholdFunctions

This script defines different threshold functions that can be used by the PredictionAnalyzer. We use `withinNumberThreshold`, but in the future, we will study the effects of different threshold functionality, like a percentage threshold or programmatic threshold.

### dataProcessing

This script is responsible for simple processing tasks associated with loading data in PENGUIN_Main. For example, its `getDataFiles` function takes a directory path as an argument and returns all files ending in the given extension. It is used by PENGUIN_Main to get a list of all the CNN txt files.

### PENGUIN Dependencies
Python v. 3.6.10

Pandas v. 0.24.2

Numpy v. 1.17.4

Sci-Py v. 1.3.1

### Copyright and License
Copyright (c) 2021, Global Computing Lab
PENGUIN is distributed under terms of the Apache License, Version 2.0 with LLVM Exceptions.
See LICENSE for more details.
