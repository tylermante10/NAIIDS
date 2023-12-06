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
2. In **SGDClassifier.py** replace line **20** with this string: "connection = sql.connect(/path/to/downloaded/**train.db**)"
   ### a. e.g. connection = sql.connect('/mnt/c/...')
3. In **SGDClassifier.py** replace line **71** with this string: "connection = sql.connect(/path/to/downloaded/**test.db**)"
4. Have a linux environment with python installed

## How to test on your local computer:

**PART 1**
1. Command: "git clone {insert-link-here}"
2. With a python interpreter, type "python(3) SGDTest.py" <br>
    a. Note: this is a basic version, it only predicts based on a small sample. SGDTest is meant as a one-click "trial" run of our model (all that will be displayed is a Prediction Accuracy). See Part 2 for more involvement.
4. Results will print!

**PART 2**
1. Command: "pip install matplotlib" <br>
2. With a python interpreter, type "python(3) SGDclassifier.py"
    a. Note: This will only work by using the path to the files as described above. <br>
3. Results will print. <br>
    a. Results include: Confusion Matrix, Classification Report, ROC curve (this will be saved in a file called "ROC_curve_comparison.png", and Prediction Accuracy.


