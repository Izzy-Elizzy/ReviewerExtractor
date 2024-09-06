import requests
from urllib.parse import urlencode, quote_plus
import numpy as np
import sys
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

path_stop= ''
stop_file='stopwords.txt'
stop_dir=path_stop+stop_file
sys.path.append(path_stop)

import TextAnalysis as TA
import ADSsearcherpkg as AP

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
token = os.getenv("token")

dataframe=AP.run_file_fellows(filename= 'codeV3\example3.csv',
               token=token, stop_dir=stop_file)