# NLPAssignment3SMSSpamDetection
Using Bag of words and BERT models to detect SMS spam messages.

- UCISmallDataSet - Sourced from https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- MendeleyLargeDataSet - Sourced from https://data.mendeley.com/datasets/vmg875v4xs/1

- We will be training on the UCI Small DataSet since we need this to run on free cloud resourses for marking purposes.
- The code *does* support using the Mendeley set for training using one of the helper methods in Helper.py to transform
  the set into the format expected (LABEL\tTEXT), but it is highly recommended to just use the UCI set,
	that is what both models have been tweaked to run best off.

- Steps to run:
1. Create venv with (from project root):
	- "py -m venv .venv"
	- "./.venv/Scripts/python.exe -m pip install -r requirements.txt"
	- "./.venv/Scripts/Activate.ps1" (per terminal session)
	- Alternatively, just scope your commpands to the python install in ./venv/Scripts/python.exe and install the dependencies but just activate the venv, its easier.
2. Run: "python ./main.py" for a single text test configured in main.py
		OR
3. Run: "python Tests\Benchmark\Benchmark.py" for a benchmark test which trains both models on the UCI set, and tests
		on 500 ham and 500 spam from the Mendeley set, with a lot of nice metrics for each model :)
