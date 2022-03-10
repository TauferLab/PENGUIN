## Running PENGUIN
To run a small test of PENGUIN, see the instructions in the "TestRun.md" file. 

PENGUIN is currently designed to run statically for testing, using data from trained CNNs. The CNNs we used in our tests are available in our Harvard Dataverse repository: https://doi.org/10.7910/DVN/ZXTCGF. In order to replicate these tests, the user can download our CNN dataset, storing all CNNs for a given benchmark dataset in a common folder. The user should also create a folder where results of PENGUIN's runs can be saved. Then, running `PENGUIN_Main --loadDirectory <path to folder with NN dataset> --saveDirectory <path to folder where PENGUIN results will be saved>` will execute PENGUIN on the CNNs stored in the `loadDirectory` folder and save the results in `saveDirectory`.

### PENGUIN_Main
`PENGUIN_Main` begins by reading and configuring PENGUIN's parameters from the command line arguments. For a given CNN, its complete training and validation info, together with all parameters defining the CNN, are loaded from a txt file named with that CNN's ID. PENGUIN_Main iterates through the text files for all CNNs on a given benchmark dataset, and simulates PENGUIN's iterative process for each CNN. 

*PENGUIN parameters*

PENGUIN's run parameters may be configured at command line when running `PENGUIN_Main.py`. If you wish to specify a value for any given parameter, add `--<parameter flag> <desired parameter value>` to the run command. (i.e. `PENGUIN_Main.py --loadDirectory <path to NN data> --saveDirectory <path to save folder> --<parameter flag> <desired parameter value>`.) See below a table describing all of the parameters.

| Parameter | Command Line Flag | Description |
| --------- | ----------------- | ----------- |
| DATA_LOADPATH | --loadDirectory | The path to the directory storing the txt files of NN learning curve data. This parameter must be specified; it has no default value. |
| DATA_SAVEPATH | --saveDirectory | The path to the directory where PENGUIN's results will be stored. This parameter must be specified; it has no default value. |
| DATASET | --datasetName | the name of the benchmark dataset that was used to train the CNNs on which PENGUIN will now be run. (i.e. `CIFAR100`, `FashionMNIST`, or `SVHN`) The default value is `CIFAR-100`|
| N | -N or --numToConverge | Number of most recent accuracy predictions to consider in the analysis of convergence. The N most recent predictions must stay within threshold in order to reach convergence. The default value is `3`. |
| threshold | -t or --threshold | The threshold value for the prediction engine. It is the max variability allowed within recent predictions for convergence. The default value is `0.5`. |
| E | -E or --epochFrequency | Number of elapsed epochs per iteration. Generally, if applying PENGUIN asynchronously, this should match the frequency of recorded fitness in the NN dataframes. The default value is `0.5`. |
| e_pred | --ePred | The epoch for which to predict NN fitness. Usually this will match the number of epochs a NAS trains for by default. (e.g. ePred=20 if augmenting MENNDL or ePred=25 if augmenting NSGA-Net). The default value is `20`. |
| C_min | --cMin | Minimum number of datapoints necessary for fitting the function. (i.e. PENGUIN will not attempt to fit a predictive function until we have at least C_min datapoints.) Generally, this should match the number of parameters of the parametric function being used. Default value is `3`. |
| fitness_type | --fitType | String indicating the name of the fitness measurement used. If applying PENGUIN asynchronously to NN learning curve dataframes, this should match the name of the column that stores fitness. (e.g. "accuracy" or "loss") The default value is `valAcc`, the title of the column where validation accuracy is saved in our CNN dataframes. |
| fitness_up_bd | --fitnessUpperBound | Largest possible reasonable value for fitness prediction. (e.g., if fitness is measured by accuracy percentage, it should not be greater than 100.) Default value is `100`.|
| fitness_low_bd | --fitnessLowerBound | Smallest possible reasonable value for fitness prediction. (e.g., if fitness is measured by accuracy percentage, it should not be less than 0.) Default value is `0`.|
| parametric function | -f or --functionType | Name of the function to use from the list in FunctionIndex (e.g. "accWbl4" or "accPow2"); if no function is specified, the default function is `accDefaultFn`. |
| num of parameters | --numParameters | The number of parameters of the selected parametric function. Default value is `3`. |
| lower_bounds | --lowerBounds | A comma separated list of lower bounds for each of the parametric function parameters. Bounds are floats; use "ninf" for negative infinity. The function MUST be defined at the bounds Default value is `0.5,1.0,0.0`. |
| upper_bounds | --upperBounds | A comma separated list of upper bounds for each of the parametric function parameters. Bounds are floats; use "inf" for positive infinity. The function MUST be defined at the bounds. Default value is `200,inf,inf`. |
| initial_values | --initialValues | A comma separated list of initial values for all the parametric function parameters. Default value is `10,1.001,100`. |
| NN_metadata_columns | --nnMetadataColumns | A space separated list of the names of columns in NN learning curve file that contain data to track. For example "learning_rate batch_size". Default value is `learning_rate batch_size`. |



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
