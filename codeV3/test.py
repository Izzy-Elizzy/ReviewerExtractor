import requests
from urllib.parse import urlencode, quote_plus
import numpy as np
import sys
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import TextAnalysis as TA
import ADSsearcherpkg as ADS

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
TESTFILE = os.getenv("names")
API_KEY = os.getenv("token")
STOPWORDS = os.getenv("stopwords")

dataframe= ADS.run_file_search(filename=TESTFILE, token=API_KEY, stop_dir=STOPWORDS)