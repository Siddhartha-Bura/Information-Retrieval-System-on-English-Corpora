#!/usr/bin/env python
# coding: utf-8

# In[12]:


from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk import flatten
from collections import Counter
from sys import argv


# In[13]:


import ntpath
import pickle
import ast
import math


# In[14]:


test_file = argv[1] # taking query file name through command line argument


# In[16]:


print(test_file)


# ## Utility functions for Preprocessing

# In[17]:


def porter_stemmer(x):  # using porter stemmer for stemming
    if type(x) is not list:
        x = x.split()
    ps = PorterStemmer()
    ans = [ps.stem(i) for i in x]
    return ans


# In[18]:


def extract_path_name(path): # utility function to extract name of the file from path
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


# In[19]:


def remove_non_ascii(string): # Removes non ascii characters
    return string.encode("ascii", "ignore").decode()


# In[20]:


def scrub_special_characters(string,with_brackets = True,with_space = False):   # removes special characters from the string
    replace_char = '@"+-=\\_!#$%^.,&*()<>?/\|}{~:;[]' if with_brackets else '@"+-=\\_!#$%^.,&*<>?/\|}{~:;[]'
    join_char = ' ' if with_space else ''
    return ''.join(x if not x in replace_char else join_char for x in string )


# In[21]:


def remove_stop_words(x,with_boolean_connectives=True): # Removes stop words
    if type(x) is not list:
        x = x.split()
    stop_words = set(stopwords.words('english')) if with_boolean_connectives else set(stopwords.words('english'))-{'and','or','not'}
    ans = [i for i in x if not i in stop_words]
    return ans


# In[22]:


def white_space_tokenizer(string): # Tokenize the string
    tk = WhitespaceTokenizer()
    return tk.tokenize(string)


# In[23]:


def query_preprocess(string, with_stop_words=False): # Preprocessing
    cleaned_string = remove_non_ascii(string)  # removes non-ascii characters
    cleaned_string= cleaned_string.replace("'s"," ")   
    uncleaned_tokenized_list = white_space_tokenizer(cleaned_string) # tokenization
    #uncleaned_tokenized_list = [i.lower() for i in uncleaned_tokenized_list]
    uncleaned_tokenized_list = [scrub_special_characters(i)  for i in uncleaned_tokenized_list if scrub_special_characters(i)]
    cleaned_token_list = remove_stop_words(uncleaned_tokenized_list) if not with_stop_words else uncleaned_tokenized_list
    cleaned_token_list = [porter_stemmer(i) for i in cleaned_token_list if porter_stemmer(i)]  # case lower
    return list(flatten(cleaned_token_list)) # put into list


# ## Reading query data and Preprocessing

# In[27]:


#reading the queries
query_file = open(test_file, 'r')
queries = query_file.readlines()


# In[28]:


#seperating the queryid and query
queries_list=[]
for query in queries:
    queries_list.append(query.split("\t",1))


# In[29]:


#removing last element of the list if it contains only '/n'
if len(queries_list[-1])==1:
    queries_list=queries_list[:-1]


# In[30]:


processed_queries={}
for i in queries_list:
    processed_queries[i[0]]=query_preprocess(i[1])


# In[31]:


processed_queries


# ## Loading all relevant dictionaries

# In[32]:


tf = pickle.load(open('tf.p', "rb"))
df = pickle.load(open('df.p', "rb"))
idf = pickle.load(open('idf.p', "rb"))
tf_idf = pickle.load(open('tf_idf.p', "rb"))
idf_bm25 = pickle.load(open('idf_bm25.p', "rb"))
doc_lens = pickle.load(open('doc_lens.p', "rb"))


# ## Boolean Search

# In[33]:


def boolean_search(input_dir = "english-corpora/",count = 5):
    output_dictionary_file_name = extract_path_name(input_dir)+"_dictionary.p"  # paths
    output_postings_file_name = extract_path_name(input_dir)+"_postings.txt"
    dictionary = pickle.load(open(output_dictionary_file_name, "rb"))
    postings_file = open(output_postings_file_name, "r")
    with open(test_file,'r') as query_file:
        lines = query_file.readlines()
    query_file.close()  
    output = ''
    for line in lines:
        docs = set()
        doc_score = {}
        query_list = [p.strip() for p in line.split(None,1)]  
        if len(query_list)==2:
            query_id = query_list[0]
            query = query_list[1]
        else:
            break   # if reached to end
        tokens = query_preprocess(query,True)
        tokens = list(set(tokens))
        for token in tokens:
            if token in dictionary:
                byteOffset = dictionary[token][0]   # taking offset for postings list
                pf = postings_file
                pf.seek(byteOffset)
                postList = ast.literal_eval(pf.readline().rstrip())
                for i in postList:
                    doc = i
                    if doc in docs:
                        doc_score[doc]+=1 # if already present add 1
                    else:
                        docs.add(doc)
                        doc_score[doc]=1  # if not make the value 1
        ans = [[k, v] for k, v in doc_score.items()] 
        ans.sort(key=lambda x: x[1],reverse=True)   # sort in descending order
        ans = [x[0] for x in ans]
        ans = ans[:count]
        for i in ans:
            output += query_id + ",1," + i + ",1" + '\n'
    with open('boolean_qrel_output.txt', 'w+') as fh:
        fh.write(output)
    fh.close()


# In[34]:


boolean_search()


# ## TF-IDF Search

# In[67]:


def tfidf_search(count = 5):
    
    output = ''
    for query_id in processed_queries:
        query_tokens=processed_queries[query_id]
        query_dictionary = {}       # query tokens as keys and its tfidf values as values to the dictionary
        for k,v in Counter(query_tokens).items():   # calculating tfidf values for query given
            if k in idf:
                query_dictionary[k] = (v/len(query_tokens))*(idf[k])
        ans = cosine_similarity(tf_idf,query_dictionary)
        ans = ans[:count]
        for i in ans:
            output += query_id + ",1," + i + ",1" + '\n'
    with open('tfidf_qrel_output.txt', 'w+') as fh:
        fh.write(output)
    fh.close()
    
    
def cosine_similarity(dictionary,query_dictionary): # calculates cosine similarity iteratively between a set of document and the query as a document
    denominator1 = math.sqrt(sum([query_dictionary[k]*query_dictionary[k] for k in query_dictionary.keys()]))
    ans = []
    for doc in dictionary.keys():
        numerator = float(0.0)
        for token in query_dictionary.keys():
            if token in dictionary[doc].keys():
                numerator += query_dictionary[token]*dictionary[doc][token]
        denominator2 = math.sqrt(sum([dictionary[doc][k]*dictionary[doc][k] for k in dictionary[doc].keys()]))
        ans.append([doc,numerator/(denominator1*denominator2)])
    ans.sort(key=lambda x: x[1],reverse=True)
    ans = [x[0] for x in ans]
    return ans


# In[68]:


tfidf_search()


# ## BM25 Search

# In[69]:


def bm25_search(count = 5):
    b = 0.75
    k_1 = 1.2
    output = ''
    avgdl=(sum(list(doc_lens.values())))/len(doc_lens)
    for query_id in processed_queries:
        bm25_scores=[]
        query_tokens=processed_queries[query_id]
        for doc in tf:
            curr_score=0
            for token in query_tokens :
                if token in tf[doc]:
                    curr_score+=(idf_bm25[token]*tf[doc][token]*doc_lens[doc]*(k_1+1))/((tf[doc][token]*doc_lens[doc])+k_1*(1-b+b*(doc_lens[doc]/avgdl)))
            bm25_scores.append([curr_score,doc])
        bm25_scores.sort(reverse=True) # sorting in descending order
        doc_ids=[]
        for i in range(count):  # returns top desired no of elements
            doc_ids.append(bm25_scores[i][1])
        for i in doc_ids:
            output += query_id + ",1," + i + ",1" + '\n'
    with open('bm25_qrel_output.txt', 'w+') as fh:
        fh.write(output)
    fh.close()


# In[70]:


bm25_search()


# In[ ]:




