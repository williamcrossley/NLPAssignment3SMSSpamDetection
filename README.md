# NLPAssignment3SMSSpamDetection
Using Bag of words, CNN, and BERT models to detect SMS spam messages.

- UCISmallDataSet - Sourced from https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- MendeleyLargeDataSet - Sourced from https://data.mendeley.com/datasets/vmg875v4xs/1

- We will be using the UCI Small DataSet since we need this to run on free cloud resourses for marking purposes.
- We can revert to using the large data set if results are unsatisfactory and the dataset is suspected as the cause

- Steps to run:
1. Create venv with (from project root):
	- "py -m venv .venv"
	- "./.venv/Scripts/python.exe -m pip install -r requirements.txt"
	- "./.venv/Scripts/Activate.ps1" (per terminal session)
	- Alternatively, just scope your commpands to the python install in ./venv/Scripts/python.exe and install the dependencies but just activate the venv, its easier.
2. run python ./main.py
		- Later I'll split this into probably a training script and a run script, with trained states stored in file, but for now this is fine.
