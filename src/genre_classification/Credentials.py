import json
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
from pathlib import Path
from .config import KAGGLEHUB_USERNAME, KAGGLEHUB_KEY, HUGGINGFACE_TOKEN, ENV_PATH

def kagglehub_credentials():

	if KAGGLEHUB_USERNAME and KAGGLEHUB_KEY:
		return

	username, key = None, None

	# path = Path.home() / ".kaggle" / "kaggle.json"
	# if path.exists():
	# 	with open(path, "r") as f:
	# 		creds = json.load(f)

	# 	if "username" in creds and "key" in creds:
	# 		username = creds["username"]
	# 		key = creds["key"]

	if not username or not key:
		root = tk.Tk()
		root.withdraw()
		while not username or not key:
			username = tk.simpledialog.askstring("KaggleHub Credentials", "Enter your KaggleHub Username:")
			key = tk.simpledialog.askstring("KaggleHub Credentials", "Enter your KaggleHub Key:")

			if not username or not key:
				tk.messagebox.showerror("Error", "Both Username and Key are required. Please try again.")

	try:
		with open(ENV_PATH, "a") as env_file:
			env_file.write(f"KAGGLEHUB_USERNAME=\"{username}\"\n")
			env_file.write(f"KAGGLEHUB_KEY=\"{key}\"\n")
	except Exception as exception:
		raise exception

def huggingface_credentials() -> bool:
	if HUGGINGFACE_TOKEN:
		return True

	token = None
	path = Path.home() / ".cache" / "huggingface" / "token"
	if path.exists():
		with open(path, "r") as f:
			token = f.read().strip()

	if not token:
		root = tk.Tk()
		root.withdraw()
		while not token:
			token = tk.simpledialog.askstring("Hugging Face Credentials", "Enter your Hugging Face Token:")

			if not token:
				tk.messagebox.showerror("Error", "Invalid token. Please try again.")

	try:
		with open(ENV_PATH, "a") as env_file:
			env_file.write(f"HUGGINGFACE_TOKEN=\"{token}\"\n")
	except Exception as exception:
		raise exception
