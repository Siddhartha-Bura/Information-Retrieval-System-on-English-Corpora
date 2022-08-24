**USING PYTHON 3.8.10 IS RECOMMENDED**
**INSTALL JUPYTER NOTEBOOK**
**KINDLY RE-RUN THE ENTIRE NOTEBOOK IF YOU FACE DIFFICULTIES, ALL AT ONCE UNLIKE CELL BY CELL EXECUTION, BECAUSE RUNNING A CELL TWICE AT
SOME POINT OF TIME MAY CAUSE INCONSISTENCIES**
**INSTALL JUPYTER NBCONVERT APP BY EXECUTING "sudo apt install jupyter-nbconvert"**
**BEFORE EXECUTING ANY SHELL SCRIPT, FIRST MAKE IT EXECUTABLE BY COMMAND "chmod +x <file_name>.sh"**
**FOR MORE SPECIFIC INFORMATION OF A QUESTION,PLEASE REFER THE COMMENTS IN 'build_system.ipynb' or 'search.ipynb' FILE**

REQUIREMENTS:
You are requested to install below packages before executing scripts
	1.nltk
	2.collections
	3.ntpath
	4.pickle
	5.ast
	6.math
	7.os
	8.sys
	9.re
	10.io
If you don't have stopwords in the nltk package downloaded, please run the command "nltk.download('stopwords')" in jupyter notebook
to download.
	
ASSUMPTIONS:
	* I assume you have "english-corpora" folder inside the running system folder.
	* Everytime Preprocessing happens before sending document text to any model and also on the query.
	* Output will be in qurels text format where each field is separated by a comma (,).
	* Make sure you have pickle binary files present in the same directory before executing make file.

(Q1)PREPROCESSING:
	1. Scrub all non-ascii characters present in the string.
	2. Replace all occurances of "'s" with a single space.
	3. Using White space tokenizer to tokenize the text
	4. Scrub all special characters like '@"+-=\\_!#$%^.,&*()<>?/\|}{~:;[]'
	5. Removed stop words
	6. Using Porter stemmer to stem the token to its root 
	7. Finally, a list containing tokens are returned.

(Q2)	
BOOLEAN RETRIEVAL MODEL:
	1. Receive tokens as list of each document from preprocessing and we iterate over all the documents in the corpus.
	2. We build the model in the form of postings file and dictionary file and save them.
	3. Postings file contains the first line as list of all the documents in the corpus and in the rest contains list of all 
	   documents that contain the corresponding token.
	4. Dictionary file contains token as key and corresponding value as a tuple of two elements, which in turn contains 
	   offset of the list corresponding to this token in postings file and another value as frequency of the token in all documents.
	5. Frequncy of a token here refers to either '1' if it is present in a document no matter how many times else '0' if not
	   present in any of the document in corpus.
	6. For searching, we count the occurances of tokens in query in a document and iterate over all the documents. Same rule (4)
	   applies here too.
	7. Sort the count of each document in descending order and retrieve the first 5 documents.
	
TF-IDF RETRIEVAL MODEL:
	1. Receive tokens as list of each document from preprocessing and we iterate over all the documents in the corpus.
	2. We build the model in the form of dictionary file and inverse document frequency file and save them.
	3. Dictionary file contains documents as keys and value as another dictionary i.e. token as a key and tfidf value corresponding 
           to it as value.
      4. idf file contains the token as key and inverse document frequency corresponding to it as value.
      5. Term Frequency calculation normalizes the frequency with the length of the document.
      6. Inverse Document Frequency is calculated for a term is log(no of total documents/(no of documents the term is present + 1))
      7. For every term in each document we calculate the tfidf value by multiplying term frequency with inverse document frequency.
      8. For searching, we calculate the tfdif values for each token in query with the same formulae and sum them.
      9. we then calculate the cosine similarity of the query with all the documents in the corpus.
      10. Sort the cosine similarity values in descending order for all the documents and retrieve first 5 documents.
        
BM25 RETRIEVAL MODEL:
	1. Receive tokens as list of each document from preprocessing and we iterate over all the documents in the corpus.
	2. We build the model in the form of dictionary file and save it.
	3. Dictionary file contains documents as keys and value is a dictionary with token as key and corresponding term frequency as 
	   value.
	4. For searching, we calculate the score corresponding to each document with query as another document.
	5. Sort the scores in descending order and retrieve first 5 documents.   
	Ref: https://en.wikipedia.org/wiki/Okapi_BM25
	
   
HOW TO RUN:
	make run testfile = "query_file_name.txt" command is used without double quotes to run the search module without indexing from 	 scratch.
	buildsys.sh is used to run indexing from scratch.
         
           

**** THE END ****	
