# NLPAssignment3SMSSpamDetection
Using Bag of words and BERT models to detect SMS spam messages.

- UCISmallDataSet - Sourced from https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- MendeleyLargeDataSet - Sourced from https://data.mendeley.com/datasets/vmg875v4xs/1

- We train on the UCI Small DataSet since we need this to run on free cloud resourses for marking purposes.
- The code *does* support using the Mendeley set for training using one of the helper methods in Helper.py to transform
  the set into the format expected (LABEL/TEXT tuple), but it is highly recommended to just use the UCI set,
	that is what both models have been tweaked to run best off.

- For Benchmark testing, we use the UCI set for training, and 500 spam/smishing, 500 ham from the Mendeley set
  to ensure our models weren't just learning the style / language of a particular set which is vulernable to
  sourcing/date/location biases etc. Using a different set better tests actual real world model performance

- Since the bert training is reasonably expensive, I have implemented caching of the model with metadata tracking
  to update when the model config updates (training data, parameters etc). If you want to retrain after you have already trained,
  you can delete the artifacts folder (or benchmark / tiny-bert subfolders), or if writing code for it, you can use force_retrain

- Steps to run:
1. Create venv with (from project root):
	- "py -m venv .venv"
	- "./.venv/Scripts/python.exe -m pip install -r requirements.txt"
	- "./.venv/Scripts/Activate.ps1" (per terminal session)
	- Alternatively, just scope your commpands to the python install in ./venv/Scripts/python.exe and install the dependencies but just activate the venv, its easier.
2. Run: "python ./main.py" for a single ham/spam test configured in main.py. Feel free to modify the texts
		OR
3. Run: "python Tests\Benchmark\Benchmark.py" for the benchmark test which trains both models on the UCI set, and tests
		on 500 ham and 500 spam from the Mendeley set, with a lot of nice metrics for each model :)
