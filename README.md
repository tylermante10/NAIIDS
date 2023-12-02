# Network Artificial Intellegent Intrusion Detection System
## Preliminary Setup
1. Visit the link below to access the data we used.
https://drive.google.com/drive/folders/1udtPVs8erYtohOUlHz4FZSUTy5HGXo_E?usp=sharing and download the files test.db and train.db (the excel sheets just split up the data differently - pre-processing)
2. Replace line 17 with this string: "connection = sql.connect(/path/to/downloaded/files)"
   ### a. e.g. connection = sql.connect('/mnt/c/...')
4. Have a linux environment with python installed

How to test on your local computer:

1. Command: "git clone {insert-link-here}"
2. With a python interpereter, type "python(3) SGDTest.py" 
    a. Note: This will work with using the path to the files as described above.
3. Results will print!



