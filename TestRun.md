### PENGUIN Test Run
Begin by reading the README; it includes information about each of the code files.

To run a small-scale PENGUIN test, you will need to: 
  1. Clone this repo. Call `git clone https://github.com/TauferLab/PENGUIN`  from inside whatever folder you would like the code repository to be stored in.
  2. Create a new folder inside `data-CIFAR100/` called `PENGUINResults`. This is where the results of your run will be stored.
  3. Run PENGUIN_Main.py
  4. Check that the contents of the file created in `data/CIFAR100/PENGUINResults/` matches the desired output at the bottom of this document.

The first paragraph of the README describes how to test a run of PENGUIN on our entire dataset. This repo contains a small sample of three of our NN data files in `data-CIFAR100/models`, and PENGUIN can be tested at a small scale just on these three files. The config file `PENGUIN-input_file.txt` is setup to automatically grab the data from the folder `data-CIFAR100/models`. This is why a test can be run just by running `PENGUIN_Main.py`. To run other tests with PENGUIN, it is necessary to update the settings in the input file as described in the README.

The code is thoroughly commented; reading through it should clarify the overall flow of the PENGUIN_Main and PredictionAnalyzer files.

Once the test is running and your output matches, you can try playing around with different values for N and threshold in the file `PENGUIN-input_file.txt`

