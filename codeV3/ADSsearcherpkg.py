import requests
from urllib.parse import urlencode
import numpy as np
import TextAnalysis as TA
import itertools
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

import pandas as pd

def do_search(auth_name, inst, t, q):
    results = requests.get(
        "https://api.adsabs.harvard.edu/v1/search/query?{}".format(q),
        headers={'Authorization': 'Bearer ' + t}
    )
    data = results.json()["response"]["docs"]
    pdates = [d['pubdate'] for d in data]
    affiliations = [d['aff'][0] for d in data]
    bibcodes = [d['bibcode'] for d in data]
    f_auth = [d['first_author'] for d in data]
    keysw = [d.get('keyword', []) for d in data]
    titles = [d.get('title', '') for d in data]
    abstracts = [d.get('abstract', '') for d in data]
    ids = [d.get('identifier', []) for d in data]

    df = pd.DataFrame({
        'Input Author': [auth_name] * len(data),
        'Input Institution': [inst] * len(data),
        'First Author': f_auth,
        'Bibcode': bibcodes,
        'Title': titles,
        'Publication Date': pdates,
        'Keywords': keysw,
        'Affiliations': affiliations,
        'Abstract': abstracts,
        'Identifier': ids,
        'Data Type': [[]]*len(data)
    })

    if auth_name is None:
        df['Input Author'] = f_auth
    return df

def format_year(year):
    if isinstance(year, (int, np.integer)):
        startd = str(year - 1)
        endd = str(year + 4)
        return f'[{startd} TO {endd}]'
    elif isinstance(year, float):
        year = int(year)
        startd = str(year - 1)
        endd = str(year + 4)
        return f'[{startd} TO {endd}]'
    elif isinstance(year, str):
        if len(year) == 4:
            startd = str(int(year) - 1)
            endd = str(int(year) + 4)
            return f'[{startd} TO {endd}]'
        elif year.startswith("[") and year.endswith("]") and " TO " in year:
            return year  # Return the string as is if it's a year range
        else:
            return year
    else:
        raise ValueError("Year must be an integer, float, or a string representing a year or a year range.")

def ads_search(name=None, institution=None, year=None, refereed='property:notrefereed OR property:refereed', \
               token=None, stop_dir=None):
    
    final_df = pd.DataFrame()
    value = 0
    if name:
        value = value + 1
    if institution:
        value = value + 2
    if year:
        value = value + 4

    if value == 0:
        print("You did not give me enough to search on, please try again.")
        return final_df

    # Simplified query construction
    query = ""
    if name:
        query += f'author:"^{name}"'
    if institution:
        query += f'pos(institution:"{institution}",1)' if query else f'pos(institution:"{institution}",1)' 
    if year:
        years = format_year(year)
        query += f', pubdate:{years}' if query else f'pubdate:{years}' 

    print(f"I will search for papers matching the following criteria:\n{query}\n")

    encoded_query = urlencode({
        "q": query,
        "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword, identifier",
        "fq": "database:astronomy," + str(refereed),
        "rows": 3000,
        "sort": "date desc"
    })

    try:
        print('I am now querying ADS.\n')
        results = requests.get(
            "https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query),
            headers={'Authorization': 'Bearer ' + token}
        )
        data = results.json()["response"]["docs"]
    except:
        print('Ooops, something went wrong.\n')

    df = do_search(name, institution, token, encoded_query)

    # Try affiliation instead of institution if the DataFrame is empty
    if df.empty:
        print('DataFrame is empty! Trying affiliation instead of institution.')
        if institution:
            query = 'pos(aff:"{}",1)'.format(institution)
            if year:
                query += ', pubdate:{}'.format(years)
            print(f"Trying alternative search: {query}")
            encoded_query = urlencode({
                "q": query,
                "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword,identifier",
                "fq": "database:astronomy," + str(refereed),
                "rows": 3000,
                "sort": "date desc"
            })
            df = do_search(name, institution, token, encoded_query)

    # Run further analysis if the DataFrame is not empty
    if not df.empty:
        data2 = data_type(df)
        data3 = merge(data2)
        data4 = n_grams(data3, stop_dir)
        return data4
    else:
        print("No results found.")
        return final_df

def data_type(df):
    journals = ['ApJ', 'MNRAS', 'AJ', 'Nature', 'Science', 'PASP', 'AAS', 'arXiv', 'SPIE', 'A&A']

    for index, row in df.iterrows():
        flag = 0
        if any(journal in row['Bibcode'] for journal in journals):
            data_type_label = 'Clean'
        else:
            flag = flag + 1
        if row['First Author'].lower() == row['Input Author'].lower():
            data_type_label = 'Clean'
        else:
            flag = flag + 2
        df.at[index, 'Data Type'] = data_type_label if flag == 0 else 'Dirty'
    return df

def merge(df):
    df['Publication Date'] = df['Publication Date'].astype(str)
    df['Abstract'] = df['Abstract'].astype(str)
    df['Keywords'] = df['Keywords'].apply(lambda keywords: keywords if keywords else []) # <- Fix for Keywords
    df['Title'] = df['Title'].apply(lambda titles: titles if titles else []) 
    df['Identifier'] = df['Identifier'].apply(lambda ids: ids if ids else []) 
    df.fillna('None', inplace=True)

    merged = df.groupby('Input Author').aggregate({'Input Institution': ', '.join,
                                                 'First Author': ', '.join,
                                                 'Bibcode': ', '.join,
                                                 'Title': lambda x: list(itertools.chain.from_iterable(x)), 
                                                 'Publication Date': ', '.join,
                                                 'Keywords': lambda x: list(itertools.chain.from_iterable(x)), # <- Fix for Keywords
                                                 'Affiliations': ', '.join,
                                                 'Abstract': ', '.join,
                                                 'Data Type': ', '.join,
                                                 'Identifier': lambda x: list(itertools.chain.from_iterable(x))  
                                                 }).reset_index()
    return merged

def n_grams(df, directorypath):
    top10Dict = {'Top 10 Words': [],
                 'Top 10 Bigrams': [],
                 'Top 10 Trigrams': []}

    for i in df.values:
        abstracts = i[8]
        top10words = TA.topwords(abstracts, directorypath)
        top10bigrams = TA.topbigrams(abstracts, directorypath)
        top10trigrams = TA.toptrigrams(abstracts, directorypath)
        top10Dict['Top 10 Words'].append(top10words)
        top10Dict['Top 10 Bigrams'].append(top10bigrams)
        top10Dict['Top 10 Trigrams'].append(top10trigrams)

    top10Df = df
    top10Df['Top 10 Words'] = top10Dict['Top 10 Words']
    top10Df['Top 10 Bigrams'] = top10Dict['Top 10 Bigrams']
    top10Df['Top 10 Trigrams'] = top10Dict['Top 10 Trigrams']

    top10Df = top10Df[['Input Author', 'Input Institution', 'First Author', 'Bibcode', 'Title', 'Publication Date',
                       'Keywords', 'Affiliations', 'Abstract', 'Identifier', 'Top 10 Words', 'Top 10 Bigrams',
                       'Top 10 Trigrams', 'Data Type']]
    return top10Df


def get_user_input(dataframe):
    """
    Gets user input for searching a dataframe, compatible with Jupyter notebooks.
    
    Args:
        dataframe: pandas DataFrame containing the data to search
        
    Returns:
        dict: Dictionary containing search parameters and search type
    """
    # Detect available columns and build search options
    available_searches = []
    if 'Name' in dataframe.columns:
        available_searches.append(("name", "Name Search - search by author name"))
    if 'Institution' in dataframe.columns:
        available_searches.append(("institution", "Institution Search - search by institution"))
    if all(col in dataframe.columns for col in ['Name', 'Institution', 'Fellowship Year']):
        available_searches.append(("fellow", "Fellow Search - search by name, institution, and year"))

    # Display available options
    print("\nAvailable columns:", ", ".join(dataframe.columns))
    print("\nAvailable search types:")
    for search_id, description in available_searches:
        print(f"- {description}")

    # Get search type
    while True:
        try:
            search_type = input("\nWhich search would you like to perform? (name/institution/fellow): ").lower()
            if any(search_type == s[0] for s in available_searches):
                break
            print("Invalid search type. Please choose from the available options.")
        except NameError:
            print("Error getting input. Please try again.")

    # Get relevant column names based on search type
    search_params = {'search_type': search_type}
    
    if search_type == 'name':
        search_params['name_column'] = input("Name column [Name]: ") or "Name"
        search_params['year_range'] = '[2003 TO 2030]'  # Default year range
        
    elif search_type == 'institution':
        search_params['institution_column'] = input("Institution column [Institution]: ") or "Institution"
        search_params['year_range'] = '[2003 TO 2030]'  # Default year range
        
    elif search_type == 'fellow':
        search_params['name_column'] = input("Name column [Name]: ") or "Name"
        search_params['institution_column'] = input("Institution column [Institution]: ") or "Institution"
        search_params['year_column'] = input("Year column [Fellowship Year]: ") or "Fellowship Year"

    return search_params

def run_file_search(filename, token, stop_dir):
    """
    Runs search based on user's choice of search type.

    Args:
        filename (str): Path to the input CSV file
        token (str): ADS API token
        stop_dir (str): Path to stopwords file

    Returns:
        pandas.DataFrame: Search results
    """
    dataframe = pd.read_csv(filename)
    final_df = pd.DataFrame()
    count = 0

    # Get user's search preferences
    search_params = get_user_input(dataframe)
    search_type = search_params['search_type']

    for i in range(len(dataframe)):
        if search_type == 'name':
            # Name-only search
            name = dataframe[search_params['name_column']][i]
            data1 = ads_search(
                name=name,
                institution=None,
                year=search_params['year_range'],
                token=token,
                stop_dir=stop_dir
            )
            search_identifier = f"name: {name}"

        elif search_type == 'institution':
            # Institution-only search
            institution = dataframe[search_params['institution_column']][i]
            data1 = ads_search(
                name=None,
                institution=institution,
                year=search_params['year_range'],
                token=token,
                stop_dir=stop_dir
            )
            data1['Input Institution'] = institution
            search_identifier = f"institution: {institution}"

        elif search_type == 'fellow':
            # Fellow search (name + institution + year)
            name = dataframe[search_params['name_column']][i]
            institution = dataframe[search_params['institution_column']][i]
            year = str(int(dataframe[search_params['year_column']][i]))
            
            data1 = ads_search(
                name=name,
                institution=institution,
                year=year,
                token=token,
                stop_dir=stop_dir
            )
            data1['Input Institution'] = institution
            search_identifier = f"fellow: {name} at {institution} in {year}"

        # Process results if found
        if not data1.empty:
            data2 = data_type(data1)
            data3 = merge(data2)
            data4 = n_grams(data3, stop_dir)
            final_df = pd.concat([final_df, data4], ignore_index=True)
            count += 1
            print(f"Completed {count} searches - Processed {search_identifier}")
        else:
            print(f"No results found for {search_identifier}")

    return final_df

# def get_user_input(dataframe):
#     """
#     Gets user input for searching a dataframe, compatible with Jupyter notebooks
#     and all system types using raw_input() instead of input().
    
#     Args:
#         dataframe: pandas DataFrame containing the data to search
        
#     Returns:
#         dict: Dictionary containing search parameters and column names
#     """
#     # Columns we are concerned with for search
#     name_column = "Name"
#     institution_column = "Institution"
#     year_column = "Year"

#     # Detect and display available columns
#     print("I detected the following columns in your CSV:")
#     for column in dataframe.columns:
#         print(f"- {column}")

#     # Suggest search types based on available columns
#     possible_searches = []
#     if 'Name' in dataframe.columns:
#         possible_searches.append("Name Search - Type: Name")
#     if 'Institution' in dataframe.columns:
#         possible_searches.append("Institution Search - Type: Institution")    
#     if 'Name' in dataframe.columns and 'Institution' in dataframe.columns and 'Fellowship Year' in dataframe.columns:
#         possible_searches.append("Fellow Search - Type: Fellow")

#     print("\nBased on your data, you can perform the following searches:")
#     for search in possible_searches:
#         print(f"- {search}")

#     # Get user input using raw_input()
#     try:
#         # Python 2
#         search_type = raw_input("Which search would you like to perform? ").lower()
#     except NameError:
#         # Python 3
#         search_type = input("Which search would you like to perform? ").lower()

#     # Prompt the user to confirm or correct column names
#     if search_type == 'name':
#         try:
#             name_column = raw_input(f"Name column (detected: Name): ") or "Name"
#         except NameError:
#             name_column = input(f"Name column (detected: Name): ") or "Name"
            
#     if search_type == 'institution':
#         try:
#             institution_column = raw_input(f"Institution column (detected: Institution): ") or "Institution"
#         except NameError:
#             institution_column = input(f"Institution column (detected: Institution): ") or "Institution"
            
#     if search_type == "fellow":
#         try:
#             name_column = raw_input(f"Name column (detected: Name): ") or "Name"
#             institution_column = raw_input(f"Institution column (detected: Institution): ") or "Institution"
#             year_column = raw_input(f"Year column (detected: Fellowship Year): ") or "Fellowship Year"
#         except NameError:
#             name_column = input(f"Name column (detected: Name): ") or "Name"
#             institution_column = input(f"Institution column (detected: Institution): ") or "Institution"
#             year_column = input(f"Year column (detected: Fellowship Year): ") or "Fellowship Year"

#     # Return the user's input
#     return {
#         'name_column': name_column,
#         'institution_column': institution_column,
#         'year_column': year_column,
#         'search_type': search_type,
#         'default_year_range': None
#     }


# def run_file_search(filename, token, stop_dir, **kwargs):
#     """
#     Combined function for fellows, institutions, and names searches.

#     Args:
#         filename (str): Path to the input CSV file.
#         token (str): Your ADS API token.
#         stop_dir (str): Path to the stopwords file.
#         **kwargs: Optional keyword arguments for columns:
#             - name_column (str): Column name for author names. Defaults to 'Name'.
#             - institution_column (str): Column name for institutions. Defaults to 'Institution'.
#             - year_column (str): Column name for years. Defaults to 'Fellowship Year'.

#     Returns:
#         pandas.DataFrame: Dataframe containing search results.
#     """

#     dataframe = pd.read_csv(filename)
#     final_df = pd.DataFrame()
#     count = 0

#     user_input = get_user_input(dataframe)

#     name_column = user_input['name_column']
#     institution_column = user_input['institution_column']
#     year_column = user_input['year_column']

#     # name_column = kwargs.get('name_column', 'Name')
#     # institution_column = kwargs.get('institution_column', 'Institution')
#     # year_column = kwargs.get('year_column', 'Fellowship Year')

#     # Check if 'Fellowship Year' column exists (for fellows search)
#     if year_column in dataframe.columns:
#         # Handle the fellows search scenario
#         for i in range(dataframe.shape[0]):
#             name = dataframe[name_column][i]
#             inst = dataframe[institution_column][i]
#             year = dataframe[year_column][i]

#             # Convert year to string if necessary
#             if isinstance(year, (int, float)):
#                 year = str(int(year))

#             data1 = ads_search(name=name, institution=inst, year=year, token=token, stop_dir=stop_dir)
#             data1['Input Institution'] = inst

#             if not data1.empty:
#                 data2 = data_type(data1)
#                 data3 = merge(data2)
#                 data4 = n_grams(data3, stop_dir)
#                 final_df = pd.concat([final_df, data4], ignore_index=True)
#                 count += 1
#                 print(str(count) + ' iterations done')
#             else:
#                 print(f"No results found for {name} at {inst} in {year}.")
#     else:
#         # Handle the names or institutions search scenario
#         for i in range(dataframe.shape[0]):
#             if name_column in dataframe.columns:
#                 name = dataframe[name_column][i]
#                 inst = None  # Institution is not specified
#                 year = '[2003 TO 2030]'  # Default year range
#             elif institution_column in dataframe.columns:
#                 name = None  # Name is not specified
#                 inst = dataframe[institution_column][i]
#                 year = '[2003 TO 2030]'  # Default year range

#             data1 = ads_search(name=name, institution=inst, year=year, token=token, stop_dir=stop_dir)
#             data1['Input Institution'] = inst

#             if not data1.empty:
#                 data2 = data_type(data1)
#                 data3 = merge(data2)
#                 data4 = n_grams(data3, stop_dir)
#                 final_df = pd.concat([final_df, data4], ignore_index=True)
#                 count += 1
#                 print(str(count) + ' iterations done')
#             else:
#                 print(f"No results found for {name} at {inst} in {year}.")

#     return final_df
 
# ________________________________________________________Deprecated functions for testing purposes____________________________________________________________

#Depracted Function (Doesn't work with Jupyter Cells, Only Terminals)
def get_user_input_depracted(dataframe):
    
    # Create a prompt session
    session = PromptSession(auto_suggest=AutoSuggestFromHistory())
    
    # Columns we are concerned with for search
    name_column = "Name"
    institution_column = "Institution"
    year_column = "Year"

    # Detect and display available columns
    print("I detected the following columns in your CSV:")
    for column in dataframe.columns:
        print(f"- {column}")

    # Suggest search types based on available columns
    possible_searches = []
    if 'Name' in dataframe.columns:
        possible_searches.append("Name Search - Type: Name")
    if 'Institution' in dataframe.columns:
        possible_searches.append("Institution Search - Type: Institution")    
    if 'Name' in dataframe.columns and 'Institution' in dataframe.columns and 'Fellowship Year' in dataframe.columns:
        possible_searches.append("Fellow Search - Type: Fellow")
        

    print("\nBased on your data, you can perform the following searches:")
    for search in possible_searches:
        print(f"- {search}")

    # Get user input
    search_type = input("Which search would you like to perform? ").lower()

    # Prompt the user to confirm or correct column names
    if search_type == 'name':
        name_column = session.prompt(f"Name column (detected: Name): ", default = "Name")
    if search_type == 'institution':
        institution_column = session.prompt(f"Institution column (detected: Institution): ", default = "Institution")
    if search_type == "fellow":
        name_column = session.prompt(f"Name column (detected: Name): ", default = "Name" )
        institution_column = session.prompt(f"Institution column (detected: Institution): ", default = "Institution")
        year_column = session.prompt(f"Year column (detected: Year): ", default = "Fellowship Year")
    # Return the user's input
    return {
        'name_column': name_column,
        'institution_column': institution_column,
        'year_column': year_column,
        'search_type': search_type,
        'default_year_range': None  # No need for a default year range if 'Year' is in the CSV
    }


#Deprecated Fucntions (Combined Functionality Above)
def run_file_fellows_deprecated(filename, token, stop_dir ):
  dataframe= pd.read_csv(filename)
  try:
    institutions= dataframe['Current Institution']
  except:
    institutions= dataframe['Institution']
  names= dataframe['Name']
  start_years= dataframe['Fellowship Year']
  #host_insts= dataframe['Host Institution']

  final_df= pd.DataFrame()
  count= 0
  for i in range(dataframe.shape[0]):

    inst= institutions[i]
    name= names[i]
    year= int(start_years[i])

    #data1= ads_search(name, institution= inst, year_range=year)
    data1= ads_search(name=name, institution=inst, \
           year= year, token=token, stop_dir=stop_dir)
    '''
    if data1.empty:
        data1= ads_search(name, year_range=year)

    if data1.empty:
        data1= ads_search(name, year_range='general') #general year range added in ads_search function
    '''
    data1['Input Institution']=inst

    data2= data_type(data1)
    data3= merge(data2)
    data4= n_grams(data3, stop_dir)

    final_df = pd.concat([final_df, data4], ignore_index=True) # Changed pandas.dataframe.append() to pandas.concat()
    count+=1
    print(str(count)+' iterations done')
  return final_df

#Deprecated Fucntions (Combined Functionality Above)
def run_file_insts_deprecated(filename, token, stop_dir ):
  dataframe= pd.read_csv(filename)
  try:
    institutions= dataframe['Current Institution']
  except:
    institutions= dataframe['Institution']
  #names= dataframe['Name']
  #start_years= dataframe['Fellowship Year']
  #host_insts= dataframe['Host Institution']

  final_df= pd.DataFrame()
  count= 0
  for i in np.arange(len(dataframe)):

    inst= institutions[i]
    #name= names[i]
    #year= start_years[i]

    #data1= ads_search(name, institution= inst, year_range=year)
    data1= ads_search(institution=inst, \
            token=token, stop_dir=stop_dir)
    '''
    if data1.empty:
        data1= ads_search(name, year_range=year)

    if data1.empty:
        data1= ads_search(name, year_range='general') #general year range added in ads_search function
    '''
    data1['Input Institution']=inst

    #data2= data_type(data1)
    #data3= merge(data2)
    #data4= n_grams(data3, stop_dir)

    final_df = pd.concat([final_df, data1], ignore_index=True) # Changed pandas.dataframe.append() to pandas.concat()
    count+=1
    print(str(count)+' iterations done')
  return final_df

#Deprecated Fucntions (Combined Functionality Above)
def run_file_names_deprecated(filename, token, stop_dir):
  # just a file with a list of names. Format is a single column "Last, First"
  print('I will go through each name in the list. Name should be formatted in a single column called "Last, First".\
  We will search by default any pubblication between 2003 and 2030 by these authors, independently of the institutions they were\
  affiliated to. \n')

  dataframe= pd.read_csv(filename)
  #print(dataframe['Name'])
  #institutions= dataframe['Current Institution']
  names= dataframe['Name']
  #print(type(names[0]))
  final_df= pd.DataFrame()
  count= 0
  for i in np.arange(len(dataframe)):
    print(names[i])
    data1= ads_search(name=names[i],  \
           year='[2003 TO 2030]', token=token,stop_dir=stop_dir)
    
    
    final_df = pd.concat([final_df, data1], ignore_index=True) # Changed pandas.dataframe.append() to pandas.concat()
    count+=1
    print(str(count)+' iterations done')
  return final_df

#Deprecated Function (new function above)
def ads_search_deprecated(name=None, institution=None, year= None, refereed= 'property:notrefereed OR property:refereed', \
               token=None, stop_dir=None):
#editing query input here
  final_df= pd.DataFrame()
  value=0
  if name:
      value=value+1
  if institution:
      value=value+2
  if year:
    value=value+4

  # Block only name
  if value==1:
    query = 'author:"^{}", pubdate:[2008 TO 2030]'.format(name)
    print("I will search for any first author publications by %s in the last 15 years.\n" % name)

  # Block Only institution name
  if value==2:
    query = 'pos(institution:"{}",1), pubdate:[2008 TO 2030]'.format(institution)
    print("I will search for every paper who first authors is afiliated with %s and published in the past 15 years.\n" % institution)
    #print(query)

  # Block institution + name
  if value==3:
    #print('Value=3')
    query = 'pos(institution:"{}",1), author:"^{}", pubdate:[2008 TO 2030]'.format(institution, name)
    print("I will search for every paper published by %s and afiliated with %s  in the past 15 years.\n" %(institution, name) )
    #print(query)

  # Block just year, so nothing really
  if value==4:
    print("You did not give me enough to search on, please try again.")

  #Block. Name + year
  # Simplified query construction using f-strings and format_year function
  if value==5:
    query = f'author:"^{name}"'
    years = format_year(year)
    query += f', pubdate:{years}'
    print(f"I will search for every paper whose first author is {name} and has published between {years}.\n")
  
  # Block institution + year
  # Consolidated query construction and year formatting
  if value==6:
     years = format_year(year)
     query = f'pos(institution:"{institution}",1), pubdate:{years}'
     print(f"I will search for every paper whose first author is affiliated with {institution} and has published between {years}.\n")

  # Block name+institution + year
  # Simplified query construction and ensured all parameters are included
  if value==7:
     years = format_year(year)
     query = f'pos(institution:"{institution}",1), author:"^{name}", pubdate:{years}'
     print(f"I will search for every paper published by {name} and affiliated with {institution} between {years}.\n")


  #making and sending query to ADS


  encoded_query = urlencode({
        "q": query,
        "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword, identifier",
        "fq": "database:astronomy,"+str(refereed),
        "rows": 3000,
        "sort": "date desc"
    })

  try:
    print('I am now querying ADS.\n')
    #print(encoded_query)
    results = requests.get(
          "https://api.adsabs.harvard.edu/v1/search/query?{}".format(encoded_query),
         headers={'Authorization': 'Bearer ' + token}
      )
    data = results.json()["response"]["docs"]
  except:
    print('Ooops, something went wrong.\n')



  #extract results into each separate detail

  pdates = [d['pubdate'] for d in data]
  affiliations = [d['aff'][0] for d in data]
  bibcodes = [d['bibcode'] for d in data]
  f_auth = [d['first_author'] for d in data]
  keysw = [d.get('keyword', []) for d in data]
  titles = [d.get('title', '') for d in data]
  abstracts = [d.get('abstract', '') for d in data]
  ids= [d.get('identifier', []) for d in data]
  #define data frame

  df = pd.DataFrame({
        'Input Author': [name] * len(data),
        'Input Institution': [institution] * len(data),
        'First Author': f_auth,
        'Bibcode': bibcodes,
        'Title': titles,
        'Publication Date': pdates,
        'Keywords': keysw,
        'Affiliations': affiliations,
        'Abstract': abstracts,
        'Identifier': ids,
        'Data Type': [[]]*len(data)
    })

  if name==None:
        df['Input Author']= f_auth

  ##############################
  ############# Checking if the DATAFRAME is EMPTY and trying affiliation instead of institution
  #############
  if df.empty:
    print('DataFrame is empty! Something is wrong with the institution')
    if value==2:
      print('I am querying ADS in a different way, stay tuned!/n')

      query = 'pos(aff:"{}",1), pubdate:[2008 TO 2030]'.format(institution)
      print("I will search for every paper who first authors is afiliated with %s and published in the past 15+ years.\n" % institution)
      #print(query)

      #making and sending query to ADS

      encoded_query = urlencode({
        "q": query,
        "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword, identifier",
        "fq": "database:astronomy,"+str(refereed),
        "rows": 3000,
        "sort": "date desc"
        })

      df=do_search(name, institution, token, encoded_query)
    if value==6:
      print('I am at the alternative 6')

      refereed='property:notrefereed OR property:refereed'
      query = 'pos(aff:"{}",1)'.format(institution)
      if len(year)==4:
        startd=str(int(year)-1)
        endd=str(int(year)+4)
        years='['+startd+' TO '+endd+']'
        print("I will search for every paper who first authors is %s and has published between %s and %s. /n" % (name,str(startd),str(endd)))

      else:
        years=year
        #query += ', pubdate:{}'.format(years) #input year in function
        print("I will search for every paper who first authors is %s and has published between %s and %s. /n" % (name,year[1:5],year[9:14]))
    
      query = 'pos(institution:"{}",1)'.format(institution)
      query += ', pubdate:{}'.format(years) #input year in function
    
      #print(query)
      #making and sending query to ADS

      encoded_query = urlencode({
        "q": query,
        "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword,identifier",
        "fq": "database:astronomy,"+str(refereed),
        "rows": 3000,
        "sort": "date desc"
        })

      df=do_search(name, institution, token, encoded_query)

    if value==7:
      #print('I am at the alternative 7')
      print('I am querying ADS in a different way, stay tuned!/n')

      refereed='property:notrefereed OR property:refereed'

      query = 'pos(aff:"{}",1), author:"^{}"'.format(institution, name)
      if len(str(year))==4: #<- Attempted to take len() of int. Change to Str() before taking len().
        startd=str(int(year)-1)
        endd=str(int(year)+4)
        years='['+startd+' TO '+endd+']'
        print("I will search for every paper who first authors is %s and has published between %s and %s. /n" % (name,str(startd),str(endd)))

      # Attempts to Access a Int as a indexable object
      else:
        years=year
        #query += ', pubdate:{}'.format(years) #input year in function
        print("I will search for every paper who first authors is %s and has published between %s and %s. /n" % (name,year[1:5],year[9:14]))
    
      query = 'pos(institution:"{}",1)'.format(institution)
      query += ', pubdate:{}'.format(years) #input year in function
    

      print("I will search for every paper published by %s and affiliated with %s  \
          between %s and %s.\n" %(name, institution,startd,endd) )
      #print(query)
      encoded_query = urlencode({
        "q": query,
        "fl": "title, first_author, bibcode, abstract, aff, pubdate, keyword,identifier",
        "fq": "database:astronomy,"+str(refereed),
        "rows": 3000,
        "sort": "date desc"
        })
      #print(encoded_query)
      df=do_search(name, institution, token, encoded_query)
      df

    ######################## Block that runs the other functions to get the N-grams
  data2= data_type(df)
  data3= merge(data2)
  data4= n_grams(data3, stop_dir)

  #final_df= df.append(data4, ignore_index= True)
  return data4