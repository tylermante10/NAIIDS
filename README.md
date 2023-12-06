# Network Artificial Intellegent Intrusion Detection System
## Directory contents:
* SGDtest.py to run machine learning on sample data (within repo)
* SGDClassifier.py to run machine learning on full dataset (to be downloaded)
* Test-files: Working directory (unimportant files)
* Model-improvements: Directory for attempting to improve model accuracy
## Preliminary Setup
1. Visit the link below to access the data we used.
* https://drive.google.com/drive/folders/1tukyzG5DkGZkwaXoiwhvTo0b01WOofoN?usp=drive_link
* Download the files test.db and train.db 
2. In **SGDClassifier.py** replace line **20** with this string: "connection = sql.connect(/path/to/downloaded/**train.db**)" <br>
   a. e.g. connection = sql.connect('/mnt/c/...')
3. In **SGDClassifier.py** replace line **71** with this string: "connection = sql.connect(/path/to/downloaded/**test.db**)"
4. Have a linux environment with python installed

## How to test on your local computer:

**PART 1**
1. Command: "git clone https://github.com/tylermante10/NAIIDS.git"
2. With a python interpreter, type "python(3) SGDTest.py" <br>
    a. Note: this is a basic version, it only predicts based on a small sample. SGDTest is meant as a one-click "trial" run of our model (all that will be displayed is a Prediction Accuracy). See Part 2 for more involvement.
4. Results will print!

**PART 2**
1. Command: "pip install matplotlib" <br>
2. With a python interpreter, type "python(3) SGDclassifier.py"
    a. Note: This will only work by using the path to the files as described above. <br>
3. Results will print. <br>
    a. Results include: Confusion Matrix, Classification Report, ROC curve (this will be saved in a file called "ROC_curve_comparison.png", and Prediction Accuracy.

**Extra Code Artifacts** <br>
See Comments with TODO highlighted for areas to change file paths (as done above in part 2)
1. If you would like to test our partial fitting methods, change directories to model-improvements and run the command "python(3) SGDepochs.py" <br>
a. Note: This resulted in lessened accuracy around 5% worse, hence why the full fit was used
2. If you would like to view results for feature extraction, see the commented sections in SGDredux.py and execute that file similar to the above commands.


