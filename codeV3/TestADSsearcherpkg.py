import pytest
import ADSsearcherpkg as ADS
import pandas as pd

#Mock Test Data 

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


def testLegacyAppendAndConcatFunctionality():
    
    # Test old append method
    result_of_append = df1.append([df2, df3], ignore_index=True)

    # Test new concat method
    result_of_concat = pd.concat([df1, df2, df3], ignore_index=True)

    assert result_of_append.equals(result_of_concat)


# Regression tests

# def testRunFileFellows():
# def testRunFileInsts():
# def testRunFileNames():