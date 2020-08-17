"""
Written by Nadia Isiboukaren
"""

import pandas as pd
import Levenshtein
import ngram
from sklearn.utils import shuffle
########################
### IMPORTING FILES ####
########################
# importing the suppliers file
def import_files():
    data = pd.read_csv('/Users/anass/Desktop/SuppliersApp1/SuppliersApp1.csv', engine='python', delimiter=";",
                   error_bad_lines=False, dtype='str',
                   names=['CustomerId', 'SupplierReference', 'SupplierName', 'SupplierAddress', 'SupplierPostcode',
                          'SupplierTown', 'SupplierCountry',
                          'SupplierWebAddress', 'SupplierEmail', 'SupplierKVK', 'SupplierPhone', 'SupplierFax',
                          'SupplierTypeId', 'SupplierStatus', 'SupplierGroupId'])

    # importing the cora file with references to computer science papers
    cora = pd.read_csv('/Users/anass/Documents/Thesis/Datasets/Cora/cora.csv')

    #importing restaurant files
    res1 = pd.read_csv('/Users/anass/Documents/Thesis/Datasets/Fodors_Zagat/fodors.csv')
    res2 = pd.read_csv('/Users/anass/Documents/Thesis/Datasets/Fodors_Zagat/zagats.csv')
    res = pd.concat([res1, res2])

    return

# import csv file with entity linkage
coraLabeled = pd.read_csv('/Users/anass/Documents/Thesis/Datasets/Cora/candset_ids_only.csv')
coraLabeled = coraLabeled[coraLabeled.gold == 1]
coraLabeled.ltable_id.value_counts() #number of duplicates = 803

# copy the file to another dataframe.
test = data.copy()

### CLEAN SUPPLIERS DATASET ###
def cleanSupplier(test):
    # drop the irrelevant columns
    test = test.drop(
        columns=['CustomerId', 'SupplierReference', 'SupplierWebAddress', 'SupplierEmail', 'SupplierKVK',
                 'SupplierPhone',
                 'SupplierFax', 'SupplierTypeId', 'SupplierStatus', 'SupplierGroupId'])
    # change everything to strings, in case some values are not classified as strings.
    test['SupplierName'] = test['SupplierName'].astype('str')
    test['SupplierAddress'] = test['SupplierAddress'].astype('str')
    test['SupplierPostcode'] = test['SupplierPostcode'].astype('str')
    test['SupplierTown'] = test['SupplierTown'].astype('str')
    test['SupplierCountry'] = test['SupplierCountry'].astype('str')
    # change every value to lower-case characters.
    test['SupplierName'] = test['SupplierName'].str.lower()
    test['SupplierAddress'] = test['SupplierAddress'].str.lower()
    test['SupplierPostcode'] = test['SupplierPostcode'].str.lower()
    test['SupplierTown'] = test['SupplierTown'].str.lower()
    test['SupplierCountry'] = test['SupplierCountry'].str.lower()
    # strip these columns of the redundant whitespaces before and after the actual value.
    test['SupplierName'] = test['SupplierName'].str.strip()
    test['SupplierAddress'] = test['SupplierAddress'].str.strip()
    test['SupplierPostcode'] = test['SupplierPostcode'].str.strip()
    test['SupplierTown'] = test['SupplierTown'].str.strip()
    test['SupplierCountry'] = test['SupplierCountry'].str.strip()
    # removes white space in zip code
    test['SupplierPostcode'] = test['SupplierPostcode'].replace('\s', "", regex=True)
    # removes double white spaces present in the addresses
    test['SupplierAddress'] = test['SupplierAddress'].replace('\s+', ' ', regex=True)
    # remove the records with empty or unknown values.
    test = test[test.SupplierName != "nan"]
    test = test[test.SupplierName != "onbekend"]
    # shuffle the suppliersdata
    test = shuffle(test)
    test.reset_index(inplace=True, drop=True)

    return

### CLEAN CORA DATASET ###
def cleanCora(cora):
    cora = cora.drop(columns=['id', 'venue', 'address', 'publisher', 'editor', 'date', 'volume', 'pages'])
    cora['author'] = cora['author'].astype(str)
    cora['title'] = cora['title'].astype(str)
    # remove 20 records without a title filled, results into 973 records
    cora = cora[cora.title != "nan"]
    # address really general and most of the time not filled
    cora.reset_index(inplace=True, drop=True)
    return

### CLEAN RESTAURANT (FODORS) DATASET ###
def cleanRes(res):
    #drop irrelevant columns
    res = res.drop(columns=['id', 'class', 'phone', 'type'])
    # every columns needs to be classified as strings
    res['name'] = res['name'].astype(str)
    res['addr'] = res['addr'].astype(str)
    res['city'] = res['city'].astype(str)

    res.reset_index(inplace=True, drop=True)
    #remove backlashes: \' is written instead of '
    res['name'] = res['name'].replace("\\", "")
    return

####################
####################
####################
#set of 1000,
test1 = test[0:1000].reset_index()
# export this sample to a csv, since it needs to be manually labelled.
test1.to_csv("/Users/anass/Documents/Thesis/test1.csv", index=False, encoding='utf8')

############################
### JACCARD DEFINED ########
############################

#for words
def compute_jaccard_similarity_score(x, y):
    """
    Jaccard Similarity J (A,B) = | Intersection (A,B) | /
                                    | Union (A,B) |
    """
    intersection_cardinality = len(set(x).intersection(set(y)))
    union_cardinality = len(set(x).union(set(y)))
    return intersection_cardinality / float(union_cardinality)

###########################
###### SCORING FUCTION ####
###########################
# this function will calculate our address scores for record i and j, given the address, zip code and town scores.

def scoring_function(suppAdd=None, suppPost=None, suppTown=None):
    if suppAdd != None and suppPost !=None and suppTown != None:
        score = (0.4*suppAdd)+(0.4*suppPost)+(0.2*suppTown)
    elif suppAdd != None and suppTown != None and suppPost == None:
        score = (suppAdd * 0.8) + (suppTown * 0.2)
    else:
        score = None
    return score

########################################
##### DUPLICATE DETECTION FUNCTIONS ####
########################################

# this function will classify duplicates in the given suppliers dataset.
def validate_testset(dataset):
    counter = 0
    clusters = {}
    # in the for loops, every record i will be compared to every record j which comes after record i.
    for i in range(len(dataset)):
        # the counter's purpose is so the runner will know how far the programme is in the dataset.
        counter += 1
        print(counter)
        # for every record i, a dictionary will be made for their duplicates
        key_string = dataset['SupplierName'][i], dataset['SupplierAddress'][i], dataset['SupplierPostcode'][i], dataset['SupplierTown'][i], dataset['SupplierCountry'][i]
        clusters[key_string] = []
        for j in range(i+1, len(dataset)):
            triAdd = None
            triPost = None
            triTown = None
            levAdd = None
            levPost = None
            levTown = None
            # jacAdd = None
            # jacPost = None
            # jacTown = None
            # calculate the trigram and levenshtein edit distance values for the name of record i and j.
            trigram = ngram.NGram.compare(dataset['SupplierName'][i], dataset['SupplierName'][j], N=3)
            lev = Levenshtein.ratio(dataset['SupplierName'][i], dataset['SupplierName'][j])
            #jac = compute_jaccard_similarity_score(dataset['SupplierName'][i], dataset['SupplierName'][j])
            # only calculate the scores for the address, zip code and town if they are both filled.
            if (dataset['SupplierAddress'][i] != "nan") and (dataset['SupplierAddress'][j] != "nan"):
                triAdd = ngram.NGram.compare(dataset['SupplierAddress'][i], dataset['SupplierAddress'][j], N=3)
                levAdd = Levenshtein.ratio(dataset['SupplierAddress'][i], dataset['SupplierAddress'][j])
                # jacAdd = compute_jaccard_similarity_score(dataset['SupplierAddress'][i], dataset['SupplierAddress'][j])
            if (dataset['SupplierPostcode'][i] != "nan") and (dataset['SupplierPostcode'][j] != "nan"):
                triPost = ngram.NGram.compare(dataset['SupplierPostcode'][i], dataset['SupplierPostcode'][j], N=3)
                levPost = Levenshtein.ratio(dataset['SupplierPostcode'][i], dataset['SupplierPostcode'][j])
                # jacPost = compute_jaccard_similarity_score(dataset['SupplierPostcode'][i], dataset['SupplierPostcode'][j])
            if (dataset['SupplierTown'][i] != "nan") and (dataset['SupplierTown'][j] != "nan"):
                triTown = ngram.NGram.compare(dataset['SupplierTown'][i], dataset['SupplierTown'][j], N=3)
                levTown = Levenshtein.ratio(dataset['SupplierTown'][i], dataset['SupplierTown'][j])
                # jacTown = compute_jaccard_similarity_score(dataset['SupplierTown'][i], dataset['SupplierTown'][j])

            #calculating the address score of record i and j.
            triscore = scoring_function(triAdd, triPost, triTown)
            levscore = scoring_function(levAdd, levPost, levTown)
            #jacscore = scoring_function(jacAdd, jacPost, jacTown)

            # if calculated then both supplier name and supplier address should be above threshold.
            if triscore != None and levscore != None:
                if (trigram >= 0.66 and triscore >= 0.9) or (levscore >= 0.98 and lev >= 0.46):
                    # the duplicate j will be added to the dictionary of i.
                    clusters[key_string].append([dataset['SupplierName'][j], dataset['SupplierAddress'][j],
                                                 dataset['SupplierPostcode'][j], dataset['SupplierTown'][j],
                                                 trigram, lev, triscore, levscore])
            # if there is no address score, only look at the name value
            elif (trigram >= 0.66):
                # the duplicate j will be added to the dictionary of i.
                clusters[key_string].append([dataset['SupplierName'][j], dataset['SupplierAddress'][j],
                                             dataset['SupplierPostcode'][j], dataset['SupplierTown'][j],
                                             trigram])
        # if the dictionary of i turns out to be empty, remove the dictionary entirely since it has no puropse.
        if len(clusters[key_string]) == 0:
            del clusters[key_string]
    # return our results.
    return clusters

# the restaurant dataset requires a new function, because the attributes are not identical.
def validate_resset(res):
    clusters = {}
    counter = 0
    # for loop
    for i in range(len(res)):
        key_string = res['name'][i], res['addr'][i], res['city'][i]
        clusters[key_string] = []
        counter += 1
        print(counter)
        for j in range(i + 1, len(res)):
            #compute name similarity scores
            trigram = ngram.NGram.compare(res['name'][i], res['name'][j], N=3)
            # lev = Levenshtein.ratio(res['name'][i], res['name'][j])
            # jac = compute_jaccard_similarity_score(res['name'][i], res['name'][j])
            #compute address similarity scores
            triAdd = ngram.NGram.compare(res['addr'][i], res['addr'][j], N=3)
            # levAdd = Levenshtein.ratio(res['addr'][i], res['addr'][j])
            # jacAdd = compute_jaccard_similarity_score(res['addr'][i], res['addr'][j])

            if (trigram >= 0.45 and triAdd >= 0.65) or (trigram >= 0.9 and triAdd >= 0.3) or (trigram >= 0.3 and triAdd >= 0.9):
                clusters[key_string].append([res['name'][j], res['addr'][j],
                                              res['city'][j], trigram, triAdd])
        if len(clusters[key_string]) == 0:
            del clusters[key_string]
    return clusters

# classifying function for the cora dataset
def validate_coraset(cora):
    clusters = {}
    counter = 0
    # for loop
    for i in range(len(cora)):
        key_string = cora['author'][i], cora['title'][i]
        clusters[key_string] = []
        counter += 1
        print(counter)
        for j in range(i+1, len(cora)):
            triTitle = ngram.NGram.compare(cora['title'][i], cora['title'][j], N=3)
            # levTitle = Levenshtein.ratio(cora['title'][i], cora['title'][j])
            jacTitle = compute_jaccard_similarity_score(cora['title'][i], cora['title'][j])
            if triTitle >= 0.45 and jacTitle >= 0.7:
                clusters[key_string].append([cora['author'][j], cora['title'][j],
                                                 triTitle])
            #cora.drop(cora.index[j], inplace=True)
        if len(clusters[key_string]) == 0:
            del clusters[key_string]
    return clusters






