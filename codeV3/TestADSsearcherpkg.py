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
FELLOWS = os.getenv("fellows") # Make sure this is set to a valid CSV file path
NAMES = os.getenv("names")
INST = os.getenv("institutions")

def testEnvironmentalVariables():
    assert API_KEY and STOPWORDS and FELLOWS and NAMES and INST != None

# # Regression tests
# @pytest.mark.parametrize("search_type", ['insts', 'names','fellows'])
# def testRegressionRunFileSearch(search_type):
#     """
#     Test that the combined 'run_file_search' function produces the same output
#     as the individual functions for fellows, institutions, and names.
#     """
#     deprecated_func_name = f"run_file_{search_type}_deprecated"
    
#     # Call deprecated individual functions
#     deprecated_dataframe = getattr(ADS, deprecated_func_name)(filename=FELLOWS, token=API_KEY, stop_dir=STOPWORDS)
    
#     # Call the combined function with appropriate keyword arguments
#     if search_type == "insts":
#         combined_dataframe = ADS.run_file_search(filename=INST, token=API_KEY, stop_dir=STOPWORDS)
#     elif search_type == "names":
#         combined_dataframe = ADS.run_file_search(filename=NAMES, token=API_KEY, stop_dir=STOPWORDS)
#     else:  # 'fellows'
#         combined_dataframe = ADS.run_file_search(filename=FELLOWS, token=API_KEY, stop_dir=STOPWORDS)

#     # For debugging:
#     deprecated_dataframe.to_csv("dep_{search_type}.csv")
#     combined_dataframe.to_csv('com_{search_type}.csv')

#     # Assert that both DataFrames are equal
#     assert deprecated_dataframe.equals(combined_dataframe)


# def testRegressionAdsSearch():
#     """
#     Test that the `ads_search` function produces the same results as the
#     deprecated `ads_search_deprecated` function for a given set of criteria.
#     """

#     # Choose test criteria (name, institution, year)
#     name = "Browning, Matthew"
#     institution = "University of California, Berkeley"
#     year = 2005

#     # Call both functions with the same criteria
#     result_df = ADS.ads_search(name=name, institution=institution, year=year, 
#                                 token=API_KEY, stop_dir=STOPWORDS)
    
#     deprecated_result_df = ADS.ads_search_deprecated(name=name, institution=institution, year=year,
#                                                    token=API_KEY,stop_dir=STOPWORDS)
    
#     print(type(result_df))
#     print(type(deprecated_result_df))

#     result_df.to_csv("new.csv")
#     deprecated_result_df.to_csv("old.csv")

#     # Assert that both DataFrames are equal
#     assert result_df.equals(deprecated_result_df)


def testFormatYear():
    assert ADS.format_year(2020) == "[2019 TO 2024]"
    assert ADS.format_year(2020.5) == "[2019 TO 2024]"
    assert ADS.format_year("2020") == "[2019 TO 2024]"
    assert ADS.format_year("[2010 TO 2020]") == "[2010 TO 2020]" 