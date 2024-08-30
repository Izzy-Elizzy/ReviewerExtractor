import requests
from urllib.parse import urlencode, quote_plus
import numpy as np
import TextAnalysis as TA

#We designed the code to work with Pandas 1.5.3

import pandas as pd
#print(pd. __version__)
#If the Pandas version differs from 1.5.3, run the following:
#pip install pandas==1.5.3 --user

#edit the following string pointing to the directory where the stopwords.txt file is

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
  ids= [d.get('identifier', []) for d in data]
  
  #define data frame

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

  if auth_name==None:
        df['Input Author']= f_auth
  return df

def run_file_fellows(filename, token, stop_dir ):
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
  for i in np.arange(len(dataframe)):

    inst= institutions[i]
    name= names[i]
    year= start_years[i]

    #data1= ads_search(name, institution= inst, year_range=year)
    data1= ads_search(name=name, institution=inst, \
           year= year, token=token)
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

    final_df= final_df.append(data4, ignore_index= True)
    count+=1
    print(str(count)+' iterations done')
  return final_df


def run_file_insts(filename, token, stop_dir ):
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

    final_df= final_df.append(data1, ignore_index= True)
    count+=1
    print(str(count)+' iterations done')
  return final_df

def run_file_names(filename, token, stop_dir):
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
    
    
    final_df= final_df.append(data1, ignore_index= True)
    count+=1
    print(str(count)+' iterations done')
  return final_df


def merge(df):

    df['Publication Date']= df['Publication Date'].astype(str)
    df['Abstract']= df['Abstract'].astype(str)
    df['Keywords'] = df['Keywords'].apply(lambda keywords: ', '.join(keywords) if keywords else '')
    df['Title'] = df['Title'].apply(lambda titles: ', '.join(titles) if titles else '')
    df['Identifier'] = df['Identifier'].apply(lambda ids: ', '.join(ids) if ids else '')
    
# if the dataframe is missing any information it is labeled as "None"

    df.fillna('None', inplace=True)
    
    merged= df.groupby('Input Author').aggregate({'Input Institution':', '.join,
                                                  'First Author':', '.join,
                                                  'Bibcode':', '.join,
                                                 'Title':', '.join,
                                                 'Publication Date':', '.join,
                                                 'Keywords':', '.join,
                                                 'Affiliations':', '.join,
                                                'Abstract':', '.join,
                                                'Data Type':', '.join,
                                                 'Identifier':', '.join}).reset_index()

    return merged


def n_grams(df, directorypath): #directory path should lead to TextAnalysis.py
    top10Dict = {'Top 10 Words':[],
                 'Top 10 Bigrams':[],
                 'Top 10 Trigrams':[]}

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
             'Keywords', 'Affiliations', 'Abstract', 'Identifier','Top 10 Words', 'Top 10 Bigrams', 'Top 10 Trigrams', 'Data Type']]

    return top10Df


def data_type(df):

    journals = ['ApJ', 'MNRAS', 'AJ', 'Nature', 'Science', 'PASP', 'AAS', 'arXiv', 'SPIE', 'A&A']

    for index, row in df.iterrows():

        flag= 0

# Journal check
        if any(journal in row['Bibcode'] for journal in journals):
            data_type_label = 'Clean'
        else:
            flag= flag+1

#Author check
        if row['First Author'].lower() == row['Input Author'].lower():
            data_type_label = 'Clean'
        else:
            flag=flag+2


# Update the 'Data Type' column
        if flag==0:
            data_type_label = 'Clean'
        else:
            data_type_label = 'Dirty'

        df.at[index, 'Data Type'] = data_type_label

    #this lets the user know what aspect of the data made it 'dirty'- uncomment to see what the "dirty" aspect is
    #print(flag)

#flag= 1 just the journal aspect is dirty,
#flag= 2 just the author aspect is dirty,
#flag=3 the author and journal are dirty,
#flag=4 just the inst is dirty, etc.

    return df

def ads_search(name=None, institution=None, year= None, refereed= 'property:notrefereed OR property:refereed', \
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
  if value==5:
    #print('Value=5')
    #print(type(year))
    query = 'author:"^{}"'.format(name)
    if len(year)==4:
      startd=str(int(year)-1)
      endd=str(int(year)+4)
      years='['+startd+' TO '+endd+']'
      print("I will search for every paper who first authors is %s and has published between %s and %s.\n" % (name,str(startd),str(endd)))
      #query += ', pubdate:{}'.format(years) #input year in function
    else:
      years=year
      print("I will search for every paper who first authors is %s and has published between %s and %s. \n" % (name,year[1:5],year[9:13]))
    query += ', pubdate:{}'.format(years) #input year in function
    
  # Block institution + year
  if value==6:
    #print('Value=6')
    if len(year)==4:
      startd=str(int(year)-1)
      endd=str(int(year)+4)
      years='['+startd+' TO '+endd+']'
      print("I will search for every paper who first authors is %s and has published between %s and %s. \n" % (name,str(startd),str(endd)))

    else:
      years=year
      #query += ', pubdate:{}'.format(years) #input year in function
      print("I will search for every paper who first authors is %s and has published between %s and %s. \n" % (name,year[1:5],year[9:13]))
    
    query = 'pos(institution:"{}",1)'.format(institution)
    query += ', pubdate:{}'.format(years) #input year in function
    
    #print(query)

  # Block name+institution + year
  if value==7:
    #print("Value=7 \n")
    query = 'pos(institution:"{}",1), author:"^{}"'.format(institution, name)
    if len(year)==4:
      startd=str(int(year)-1)
      endd=str(int(year)+4)
      years='['+startd+' TO '+endd+']'
      print("I will search for every paper who first authors is %s and has published between %s and %s. /n" % (name,str(startd),str(endd)))

    else:
      years=year
      #query += ', pubdate:{}'.format(years) #input year in function
      print("I will search for every paper who first authors is %s and has published between %s and %s. /n" % (name,year[1:5],year[9:13]))
    
    query = 'pos(institution:"{}",1)'.format(institution)
    query += ', pubdate:{}'.format(years) #input year in function

    #print(query)


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
    

      print("I will search for every paper published by %s and affiliated with %s  \
          between %s and %s.\n" %(name, institution,year[1:5],year[9:14]) )
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
