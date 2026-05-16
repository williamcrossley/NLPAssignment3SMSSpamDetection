# Heavily inspired by the week 7 lab, but made more robust and reusable.
# Does support fine tuning caching and auto detection of training presence, and will rerun if training is removed.
# Also added metadata tracking on the chached model so you can switch states between single text tests run against the full training set,
# or use a subset of the training data for training, and a subset for testing.
# It was also raelly helpful for testing.

# In terms of design choices and assumptions, we are using the same tiny bert from the lab, and the supplied tokeniser, fine tuner/trainer.
# the majority of this file just deals with model prep, cache management, and evaluation. The fine tuning and tokenisation is pretty much as is from the lab, with some minor adjustments to work with the new dataset management.

# Initial testing OK but not great, phrases like "You have won a free lottery ticket! Click here to claim your prize."
# are 98% ham, probably due to imbalance of ham/spam ratio in the UCI set. Further inv required
# Its still reporting 98% accuracy and an f1 score of 0.98, but I am suspicious our benchmarking test is wrong.

import json
import logging
import os
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments

os.environ["WANDB_DISABLED"] = "true"
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


class BERTSpamClassifier:
	def __init__(
		self,
		dataset_path=None,
		dataset_lines=None,
		model_name="prajjwal1/bert-tiny",
		output_dir=None,
		max_length=128,
		test_size=0.2,
		seed=42,
		num_train_epochs=3,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=32,
	):
		self.repo_root = Path(__file__).resolve().parents[2]
		self.dataset_path = Path(dataset_path) if dataset_path else self.repo_root / "UCISmallDataSet.txt"
		self.dataset_lines = dataset_lines
		self.model_name = model_name
		self.max_length = max_length
		self.test_size = test_size
		self.seed = seed
		self.num_train_epochs = num_train_epochs
		self.per_device_train_batch_size = per_device_train_batch_size
		self.per_device_eval_batch_size = per_device_eval_batch_size

		# Output dir (model caching etc) defaults to bert_tiny_sms, so you can use it for individual text testing.
		# Benchmarking uses its own cached model dir (bert_benchmark) so it can have different training states without interfering with the single test model.
		base_output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "artifacts" / "bert_tiny_sms"
		self.output_dir = base_output_dir
		self.checkpoint_dir = self.output_dir / "trainer"
		self.model_dir = self.output_dir / "model"
		self.metadata_path = self.output_dir / "training_state.json"

		self.label_to_id = {"ham": 0, "spam": 1}
		self.id_to_label = {0: "ham", 1: "spam"}

		self.tokenizer = None
		self.model = None
		self.trainer = None
		self.dataset = None

	def train(self, force_retrain=False, max_train_samples=None, max_test_samples=None):
		if not force_retrain and self._has_valid_cached_model(max_train_samples, max_test_samples):
			self._load_cached_model(max_train_samples, max_test_samples)
			evaluation = self.evaluate(
				force_retrain=False,
				max_train_samples=max_train_samples,
				max_test_samples=max_test_samples,
			)
			return {
				"cached": True,
				"pre_eval_accuracy": None,
				"post_eval_accuracy": evaluation["accuracy"],
				"train_result": None,
			}

		self.dataset = self._prepare_dataset(
			max_train_samples=max_train_samples,
			max_test_samples=max_test_samples,
		)
		self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
		tokenized_dataset = self.dataset.map(self._tokenize_function, batched=True)
		tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

		self.model = BertForSequenceClassification.from_pretrained(
			self.model_name,
			num_labels=2,
			id2label=self.id_to_label,
			label2id=self.label_to_id,
		)

		# Evaluate before training
		trainer_pre = Trainer(
			model=self.model,
			args=TrainingArguments(output_dir=str(self.checkpoint_dir), dataloader_pin_memory=False, report_to=[]),
			eval_dataset=tokenized_dataset["test"],
			compute_metrics=self._compute_metrics,
		)
		pre_results = trainer_pre.evaluate()
		pre_accuracy = pre_results["eval_accuracy"]

		# Actually fine tune the model
		self.trainer = Trainer(
			model=self.model,
			args=self._build_training_arguments(),
			train_dataset=tokenized_dataset["train"],
			eval_dataset=tokenized_dataset["test"],
			compute_metrics=self._compute_metrics,
		)

		train_result = self.trainer.train()
		self.trainer.save_model(str(self.model_dir))
		self.tokenizer.save_pretrained(str(self.model_dir))
		self._write_training_metadata(max_train_samples, max_test_samples)

		post_results = self.trainer.evaluate()
		post_accuracy = post_results["eval_accuracy"]

		return {
			"cached": False,
			"pre_eval_accuracy": pre_accuracy,
			"post_eval_accuracy": post_accuracy,
			"train_result": train_result.metrics,
		}

	def evaluate(self, force_retrain=False, max_train_samples=None, max_test_samples=None):
		self._ensure_model_loaded(
			force_retrain=force_retrain,
			max_train_samples=max_train_samples,
			max_test_samples=max_test_samples,
		)

		self.dataset = self._prepare_dataset(
			max_train_samples=max_train_samples,
			max_test_samples=max_test_samples,
		)

		tokenized_dataset = self.dataset.map(self._tokenize_function, batched=True)
		tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

		evaluation_trainer = Trainer(
			model=self.model,
			args=TrainingArguments(output_dir=str(self.checkpoint_dir), dataloader_pin_memory=False, report_to=[]),
			eval_dataset=tokenized_dataset["test"],
			compute_metrics=self._compute_metrics,
		)
		results = evaluation_trainer.evaluate()
		return {"loss": results["eval_loss"], "accuracy": results["eval_accuracy"]}

	def predict(self, text, train_if_needed=True, force_retrain=False):
		if train_if_needed:
			self._ensure_model_loaded(force_retrain=force_retrain)
		elif self.model is None or self.tokenizer is None:
			self._load_cached_model()

		encoded = self.tokenizer(
			text,
			padding="max_length",
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)

		self.model.eval()
		with torch.no_grad():
			outputs = self.model(**encoded)
			probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

		predicted_id = int(np.argmax(probabilities))
		return {
			"label": self.id_to_label[predicted_id],
			"confidence": float(probabilities[predicted_id]),
			"ham_probability": float(probabilities[self.label_to_id["ham"]]),
			"spam_probability": float(probabilities[self.label_to_id["spam"]]),
		}

	def print_training_summary(self, training_result):
		if (training_result["cached"]):
			print("Loaded cached model. Skipping training.")
			return
		comparison_table = [
			["Before Fine-Tuning", f"{training_result['pre_eval_accuracy']:.4f}"],
			["After Fine-Tuning", f"{training_result['post_eval_accuracy']:.4f}"],
		]
		print(tabulate(comparison_table, headers=["Stage", "Accuracy"], tablefmt="github"))

	def _ensure_model_loaded(self, force_retrain=False, max_train_samples=None, max_test_samples=None):
		if force_retrain or not self._has_valid_cached_model(max_train_samples, max_test_samples):
			self.train(
				force_retrain=True,
				max_train_samples=max_train_samples,
				max_test_samples=max_test_samples,
			)
			return

		if self.model is None or self.tokenizer is None:
			self._load_cached_model(max_train_samples, max_test_samples)

	def _load_cached_model(self, max_train_samples=None, max_test_samples=None):
		if not self._has_valid_cached_model(max_train_samples, max_test_samples):
			raise FileNotFoundError("No cached BERT model was found. Run train() first.")

		self.tokenizer = BertTokenizerFast.from_pretrained(str(self.model_dir))
		self.model = BertForSequenceClassification.from_pretrained(str(self.model_dir))

	def _build_training_arguments(self):
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

		return TrainingArguments( # TODO: Explain args
			output_dir=str(self.checkpoint_dir),
			seed=self.seed,
			num_train_epochs=self.num_train_epochs,
			per_device_train_batch_size=self.per_device_train_batch_size,
			per_device_eval_batch_size=self.per_device_eval_batch_size,
			eval_strategy="epoch",
			save_strategy="epoch",
			load_best_model_at_end=True,
			metric_for_best_model="accuracy",
			greater_is_better=True,
			save_total_limit=1,
			logging_dir=str(self.output_dir / "logs"),
			logging_steps=1000,
			report_to=[],
			dataloader_pin_memory=False,
		)

	def _prepare_dataset(self, max_train_samples=None, max_test_samples=None):
		labels = []
		texts = []

		source_lines = self.dataset_lines if self.dataset_lines is not None else self._read_dataset_lines()
		for label, text in self._parse_labelled_messages(source_lines):
			labels.append(self.label_to_id[label])
			texts.append(text)

		train_texts, test_texts, train_labels, test_labels = train_test_split(
			texts,
			labels,
			test_size=self.test_size,
			random_state=self.seed,
			stratify=labels,
		)

		if max_train_samples is not None:
			train_texts = train_texts[:max_train_samples]
			train_labels = train_labels[:max_train_samples]

		if max_test_samples is not None:
			test_texts = test_texts[:max_test_samples]
			test_labels = test_labels[:max_test_samples]

		return DatasetDict(
			{
				"train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
				"test": Dataset.from_dict({"text": test_texts, "label": test_labels}),
			}
		)

	def _tokenize_function(self, example):
		return self.tokenizer(
			example["text"],
			padding="max_length",
			truncation=True,
			max_length=self.max_length,
		)

	def _compute_metrics(self, eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		accuracy = accuracy_score(labels, predictions)
		return {"accuracy": accuracy}

	def _read_dataset_lines(self):
		with self.dataset_path.open(encoding="utf-8") as dataset_file:
			return dataset_file.read().splitlines()

	def _parse_labelled_messages(self, labelled_messages):
		parsed_messages = []
		for line in labelled_messages:
			label, text = line.split("\t", 1)
			normalised_label = label.strip().lower()
			if normalised_label not in self.label_to_id:
				raise ValueError(f"Unsupported label '{label}'. Expected 'ham' or 'spam'.")
			parsed_messages.append((normalised_label, text.strip()))
		return parsed_messages

	def _has_valid_cached_model(self, max_train_samples=None, max_test_samples=None):
		if not self.model_dir.exists() or not self.metadata_path.exists():
			return False

		try:
			metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
		except json.JSONDecodeError:
			return False

		return metadata == self._training_metadata(max_train_samples, max_test_samples)

	def _write_training_metadata(self, max_train_samples=None, max_test_samples=None):
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.metadata_path.write_text(
			json.dumps(self._training_metadata(max_train_samples, max_test_samples), indent=2),
			encoding="utf-8",
		)

	def _training_metadata(self, max_train_samples=None, max_test_samples=None):
		if self.dataset_lines is not None:
			dataset_content = "\n".join(self.dataset_lines).encode("utf-8")
			dataset_hash = sha256(dataset_content).hexdigest()
			dataset_path_str = "in_memory"
		else:
			dataset_bytes = self.dataset_path.read_bytes()
			dataset_hash = sha256(dataset_bytes).hexdigest()
			dataset_path_str = str(self.dataset_path.resolve())
		return {
			"dataset_path": dataset_path_str,
			"dataset_sha256": dataset_hash,
			"model_name": self.model_name,
			"max_length": self.max_length,
			"test_size": self.test_size,
			"seed": self.seed,
			"num_train_epochs": self.num_train_epochs,
			"per_device_train_batch_size": self.per_device_train_batch_size,
			"per_device_eval_batch_size": self.per_device_eval_batch_size,
			"max_train_samples": max_train_samples,
			"max_test_samples": max_test_samples,
		}
