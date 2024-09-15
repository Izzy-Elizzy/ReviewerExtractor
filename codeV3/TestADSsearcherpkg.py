import pytest
import os
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import ADSsearcherpkg as ADS

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
API_KEY = os.getenv("token")
STOPWORDS = os.getenv("stopwords")
TESTFILE = os.getenv("testfile")  # Make sure this is set to a valid CSV file path


# Mock Test Data 

df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2'],
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
})

df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5'],
    'C': ['C3', 'C4', 'C5'],
    'D': ['D3', 'D4', 'D5']
})

df3 = pd.DataFrame({
    'A': ['A6', 'A7', 'A8'],
    'B': ['B6', 'B7', 'B8'],
    'C': ['C6', 'C7', 'C8'],
    'D': ['D6', 'D7', 'D8']
})


# def testLegacyAppendAndConcatFunctionality():
    
#     # Test old append method
#     result_of_append = df1.append([df2, df3], ignore_index=True)

#     # Test new concat method
#     result_of_concat = pd.concat([df1, df2, df3], ignore_index=True)

#     assert result_of_append.equals(result_of_concat)


def testEnvironmentalVariables():
    assert API_KEY and STOPWORDS and TESTFILE != None

#Regression tests
@pytest.mark.parametrize("search_type", ['fellows', 'insts', 'names'])
def testRegressionRunFileSearch(search_type):
    """
    Test that the combined 'run_file_search' function produces the same output
    as the individual functions for fellows, institutions, and names.
    """
    deprecated_func_name = f"run_file_{search_type}_deprecated"
    
    # Call deprecated individual functions
    deprecated_dataframe = getattr(ADS, deprecated_func_name)(filename=TESTFILE, token=API_KEY, stop_dir=STOPWORDS)
    
    # Call the combined function 
    combined_dataframe = ADS.run_file_search(filename=TESTFILE, token=API_KEY, stop_dir=STOPWORDS, search_type=search_type)

    deprecated_dataframe.to_csv("dep.csv")
    combined_dataframe.to_csv('com.csv')

    # Assert that both DataFrames are equal
    assert deprecated_dataframe.equals(combined_dataframe)