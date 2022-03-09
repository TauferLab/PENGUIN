### PENGUIN Test Run
Begin by reading the README; it includes information about each of the code files.

To run a small-scale PENGUIN test, you will need to: 
  1. Clone this repo. Call `git clone https://github.com/TauferLab/PENGUIN`  from inside whatever folder you would like the code repository to be stored in.
  2. Create a new folder inside `data-CIFAR100/` called `PENGUINResults`. This is where the results of your run will be stored.
  3. Run PENGUIN_Main.py
  4. Check that the contents of the file created in `data/CIFAR100/PENGUINResults/` matches the expected output at the bottom of this document.

The first paragraph of the README describes how to test a run of PENGUIN on our entire dataset. This repo contains a small sample of three of our NN data files in `data-CIFAR100/models`, and PENGUIN can be tested at a small scale just on these three files. The config file `PENGUIN-input_file.txt` is setup to automatically grab the data from the folder `data-CIFAR100/models`. This is why a test can be run just by running `PENGUIN_Main.py`. To run other tests with PENGUIN, it is necessary to update the settings in the input file as described in the README.

The code is thoroughly commented; reading through it should clarify the overall flow of the PENGUIN_Main and PredictionAnalyzer files.

Once the test is running and your output matches, you can try playing around with different values for N and threshold in the file `PENGUIN-input_file.txt`

EXPECTED OUTPUT:

    modelID numConvLayers learningRate batchSize timeStabilized timeMenndlTrainingEnds maxAcc aValue error predictedAcc actualAcc
    2021_02_15_13_56_59_917381.txt 3 0.010439798407753376 186.0 3.0 20.0 1.29843781700142 2.0336024481156483 0.06274636629898334 1.1887354107299024 1.125989044430919
    2021_02_15_13_55_26_539263.txt 3 0.11464760689500715 25.0 4.0 20.0 28.77 25.712119946140604 1.2779412257391805 25.712058774260818 26.99
    2021_02_15_14_01_32_105932.txt 3 0.7803497065523131 152.0 8.5 20.0 17.105263157894736 24.314136828801946 5.97758974967757 22.819695012835464 16.842105263157894
