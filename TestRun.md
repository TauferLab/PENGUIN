### PENGUIN Test Run

This repo contains a small sample of three NN data files in `data-CIFAR100/models`, and PENGUIN can be tested at a small scale just on these three files. The first paragraph of the README describes how to perform a test run of PENGUIN on our entire dataset. PENGUIN may also be tested with different data, functions, or parameter values by specifying these settings in the command line arguments as described in the README.

Begin by reading through the README; it includes information about each of the code files.

To run a small-scale PENGUIN test, you will need to: 
  1. Clone this repo. Call `git clone https://github.com/TauferLab/PENGUIN`  from inside whatever folder you would like the code repository to be stored in.
  2. Create a new folder inside `data-CIFAR100/` to store results of your run, called for example `PENGUINResults`. 
  3. Run PENGUIN_Main.py as follows: `python3 PEng4NN_Main.py --loadDirectory data-CIFAR100/models/ --saveDirectory data-CIFAR100/PENGUINResults/`  *(Note: `loadDirectory` should be wherever the NN data is stored, and `saveDirectory` should the path to whatever folder you created to store results of the run. A small sample set containing three of our NN data files is stored in `data-CIFAR100/models/`, so using this path as the loadDirectory when running PENGUIN_Main.py will test PENGUIN on this small sample set.)*
  4. Check that the content of the file created in `data/CIFAR100/PENGUINResults/` matches the expected output at the bottom of this document.

The code is thoroughly commented; reading through it should clarify the overall flow of the PENGUIN_Main and PredictionAnalyzer files.

Once the test is running and your output matches, try varying `N` and `threshold` values for different PENGUIN runs, using the "-N" and "--threshold" flags to specify values for these parameters at command line. (N must be an integer value; the default value is 3. Try different values for N between e.g. 2 and 6. Threshold is a float; the default value is 0.5. Try different values for threshold between e.g. 0.1 and 1.0)

EXPECTED OUTPUT:

    modelID numConvLayers learningRate batchSize timeStabilized timeMenndlTrainingEnds maxAcc aValue error predictedAcc actualAcc
    2021_02_15_13_56_59_917381.txt 3 0.010439798407753376 186.0 3.0 20.0 1.29843781700142 2.0336024481156483 0.06274636629898334 1.1887354107299024 1.125989044430919
    2021_02_15_13_55_26_539263.txt 3 0.11464760689500715 25.0 4.0 20.0 28.77 25.712119946140604 1.2779412257391805 25.712058774260818 26.99
    2021_02_15_14_01_32_105932.txt 3 0.7803497065523131 152.0 8.5 20.0 17.105263157894736 24.314136828801946 5.97758974967757 22.819695012835464 16.842105263157894
