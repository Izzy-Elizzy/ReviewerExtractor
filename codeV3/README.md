
**Purpose of this Repository** 
This repository branch is the result of my Summer 2023 Internship at NASA Headquarters. The goals were to creat tools that will enable NASA Headquarters Astrophysics Program Scientists to 
1) Collect data on institutional demographics of proposers and peer reviews participants (or prospective participants)
2) To identify people's expertises based on their publication history through the NASA ADS archive.
Using these tools we tried to reach out to possible reviewers from institutions that normally do not participate, and hopefully change the overall pool of applicants to NASA Astrophysics funds in the future. 

The code represents a version 2.0 of a previous work done by Máire Volz (https://github.com/maireav/NASA-Internship). In the examples, we provide a method of finding experts in specific matters independent from their institutions. Additionally, we show a method that provides specific inromation on researchers at MSIs (Minority Serving Institutions) that have published in Astronomy-focused journals. 

**What the code does:** 
The main aspect is the notebook called "ADS_search.ipynb". The main function can be called in different ways (see examples), but the general idea is to identify researchers with specific expertise, identified by their past publications (refereed or not). 

"Under the hood", this program accesses ADS through the ADS specific API and searches for the specific data according to the user's inputs (some are mandatory and some are optional). The output is a .csv file with all the information collected via the API('First Author', 'Bibcode', 'Title', 'Publication Date', 'Keywords', 'Affiliations', and 'Abstract') and the N-grams created by our code, for visual inspection by the user. The user can then easily determine the expertise of each author ADS returned. 

**TIPS To Maximize The Code Results**
- Extend the year range of research as people may have published in years past
- When using list of universities or researchers names check the output dataframe:
1) Names may be missing because the input file had them mispelled;
2) Universities may be missing because the way they were searched was not how people wrote their affiliations in their article and so the ADS query cannot find a match (e.g. Cal Poly Pomona vs. California Politechnique Institute). Possible solutions are to try and run single searches on Names / Intitutions using different formats
- Perform simple searches ("CMD+F) on the excel version of the output dataframe: this will allow you to search words in the published abstracts as well and not rely only on the N-grams.
  
**What the User needs:**
The user needs an ADS API Access Token (can be found here:  https://ui.adsabs.harvard.edu/help/api/), which searches the input into ADS. Other libraries needed include: nltk and pandas version 1.5. 

Please look at the [Jupyter Notebook v1 of the code](https://github.com/ninoc/ReviewerExtractor/blob/main/codeV1/ExpertiseFinder_Tutorial.ipynb) to learn about all the possible keywords.


**Current files:**
Some files are needed to run the actual search, while others are utilized in post-processing and expertise identification (e.g. N-grams creation): 
- ADSsearcherpkg: Python file that has all of the functions used to find the expertises of the authors and produce an organized data frame with each row being an individual author and columns: 'Input Author','Input Institution', 'First Author', 'Bibcode', 'Title', 'Publication Date', 'Keywords', 'Affiliations', 'Abstract', 'Data Type'
- TextAnalysis.py: Python file that has all the functions in order to determine the top words, bigrams and trigrams in each publication.
- stopwords.txt: Text file that has a list of the stop words for language processing. 
- ADS_search.ipynb: A notebook that contains the different examples of how to use the ADSsearcherpkg functions with different input cases. These input cases include just an author, just an institution, and a csv file of 3 authors with their corresponding institutions.

