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
TESTFILE = os.getenv("fellows")
API_KEY = os.getenv("token")
STOPWORDS = os.getenv("stopwords")

dataframe= ADS.run_file_search(TESTFILE, API_KEY, STOPWORDS)

dataframe.to_csv("output.csv")