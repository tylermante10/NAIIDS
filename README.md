# Network Artificial Intellegent Intrusion Detection System
## Directory contents:
* SGDClassifier.py to run machine learn expressions
* Test-files: Working directory (unimportant files)
## Preliminary Setup
1. Visit the link below to access the data we used.
* https://drive.google.com/drive/folders/1tukyzG5DkGZkwaXoiwhvTo0b01WOofoN?usp=drive_link
* Download the above files test.db and train.db 
2. Replace line 17 with this string: "connection = sql.connect(/path/to/downloaded/files)"
   ### a. e.g. connection = sql.connect('/mnt/c/...')
4. Have a linux environment with python installed

## How to test on your local computer:

**PART 1**
1. Command: "git clone {insert-link-here}"
2. With a python interpreter, type "python(3) SGDTest.py" 
    a. Note: This will work by using the path to the files as described above.
    b. Note: this is a basic version, it only predicts based on a small sample. SGDTest is meant as a one-click "trial" run of our model (all that will be displayed is a Prediction Accuracy). See Part 2 for more involvement.
3. Results will print!

**PART 2**
_Assuming you have completed the above sets (i.e. correctly connecting to test.db and train.db)_
1. Command: "git clon {insert-link-here}"
2. Command: "pip install matplotlib"
3. With a python interpreter, type "python(3) SGDclassifier.py"
    a. Note: This will work by using the path to the files as described above.
4. Results will print.
    a. Results include: Confusion Matrix, Classification Report, ROC curve (this will be saved in a file called "ROC_curve_comparison.png", and Prediction Accuracy.


