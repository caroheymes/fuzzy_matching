#!/usr/bin/env python
# coding: utf-8

# In[5]:


from ast import literal_eval
import numpy as np
import math
import re
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from difflib import SequenceMatcher
from collections import Counter
from itertools import groupby


# In[ ]:


cd Downloads/fuzzy_matching


# # Original

# In[25]:


#original
file = 'org_norm_candidates.csv'

with open(file, encoding='utf-8-sig') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        input_dict = eval(line)
        output_dict = {}       
        #print(input_dict)
        curr_parent_org = input_dict['parent_data']['parent_org_name']
        curr_parent_sk = input_dict['parent_data']['parent_sk']
        if curr_parent_org in tmp_output_dict_lkp:
            if tmp_output_dict_lkp[curr_parent_org]['is_parent']:
                top_parent_org = curr_parent_org
            else:
                top_parent_org = tmp_output_dict_lkp[curr_parent_org]['parent_org']
            if input_dict['child_data_list']:
                for child_data in input_dict['child_data_list']:
                    curr_child_org = child_data['child_org_name']
                    if curr_child_org in tmp_output_dict_lkp:
                        pass
                    else:
                        tmp_output_dict_lkp[curr_child_org] = {"is_parent":False, 'parent_org':top_parent_org, 'org_sk':child_data['child_sk']}
                        #print(output_dict_dedup[top_parent_org])
                        try:
                            #print(child_data)
                            output_dict_dedup[top_parent_org]['child_data_list'].append(child_data)
                        except KeyError:
                            output_dict_dedup[top_parent_org]['child_data_list'] = []
                            output_dict_dedup[top_parent_org]['child_data_list'].append(child_data)
        else:
            output_dict_dedup[curr_parent_org] = {'dedup_sk': curr_parent_sk, 'child_data_list': input_dict['child_data_list'] }
            tmp_output_dict_lkp[curr_parent_org] = {"is_parent":True, 'parent_org': None, 'org_sk' : input_dict['parent_data']['parent_sk']}
            for child_data in input_dict['child_data_list']:
                curr_child_org = child_data['child_org_name']
                if curr_child_org not in tmp_output_dict_lkp:
                    tmp_output_dict_lkp[curr_child_org] = {"is_parent":False, 'parent_org':curr_parent_org, 'org_sk':child_data['child_sk']}


# # Levenshtein Function

# In[64]:


#levenshtein version
def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    # Creates a matrix of zeros the size of which
    # is 1 + the length of each string
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])


# In[65]:


#levenshtein test run
Str1 = 'MYLLC'
Str2 = 'MYLIUS S.R.L.'

Distance = levenshtein_ratio_and_distance(Str1.lower(),Str2.lower())
print(Distance)

Ratio = levenshtein_ratio_and_distance(Str1,Str2,ratio_calc = True)
print(Ratio)

# reference = 'larry'
# value_list = ['lar','lair','larrylamo']
# get_top_matches(reference, value_list)


# # Testing Jaro Functions

# In[106]:


def _score(first, second):
    shorter, longer = first.lower(), second.lower()

    if len(first) > len(second):
        longer, shorter = shorter, longer

    m1 = _get_matching_characters(shorter, longer)
    m2 = _get_matching_characters(longer, shorter)

    if len(m1) == 0 or len(m2) == 0:
        return 0.0

    return (float(len(m1)) / len(shorter) +
            float(len(m2)) / len(longer) +
            float(len(m1) - _transpositions(m1, m2)) / len(m1)) / 3.0

_score('max','andi')


# In[110]:


def _get_matching_characters(first, second):
    common = []
    limit = math.floor(min(len(first), len(second)) / 2)

    for i, l in enumerate(first):
        left, right = int(max(0, i - limit)), int(
            min(i + limit + 1, len(second)))
        if l in second[left:right]:
            common.append(l)
            second = second[0:second.index(l)] + '*' + second[
                                                       second.index(l) + 1:]
            return i, l
            
_get_matching_characters('max','andimx')


# In[112]:


def _get_prefix(first, second):
    if not first or not second:
        return ""

    index = _get_diff_index(first, second)
    if index == -1:
        return first

    elif index == 0:
        return ""

    else:
        return first[0:index]

_get_prefix('max','andi')


# In[113]:


def _get_diff_index(first, second):
    if first == second:
        pass

    if not first or not second:
        return 0

    max_len = min(len(first), len(second))
    for i in range(0, max_len):
        if not first[i] == second[i]:
            return i

    return max_len

_get_diff_index('max','andi')


# In[114]:


def _transpositions(first, second):
    return math.floor(
        len([(f, s) for f, s in zip(first, second) if not f == s]) / 2.0)

_transpositions('max','andi')


# In[115]:


def get_jaro_distance(first, second, winkler=True, winkler_ajustment=True,
                      scaling=0.1, sort_tokens=True):
    """
    :param first: word to calculate distance for
    :param second: word to calculate distance with
    :param winkler: same as winkler_ajustment
    :param winkler_ajustment: add an adjustment factor to the Jaro of the distance
    :param scaling: scaling factor for the Winkler adjustment
    :return: Jaro distance adjusted (or not)
    """
    if sort_tokens:
        first = sort_token_alphabetically(first)
        second = sort_token_alphabetically(second)

    if not first or not second:
        raise JaroDistanceException(
            "Cannot calculate distance from NoneType ({0}, {1})".format(
                first.__class__.__name__,
                second.__class__.__name__))

    jaro = _score(first, second)
    cl = min(len(_get_prefix(first, second)), 4)

    if all([winkler, winkler_ajustment]):  # 0.1 as scaling factor
        return round((jaro + (scaling * cl * (1.0 - jaro))) * 100.0) / 100.0

    return jaro

get_jaro_distance('max','andi')


# # All Jaro Functions

# In[326]:


#jaro version
def sort_token_alphabetically(word):
    token = re.split('[,. ]', word)
    sorted_token = sorted(token)
    return ' '.join(sorted_token)

def get_jaro_distance(first, second, winkler=True, winkler_ajustment=True,
                      scaling=0.1, sort_tokens=True):
    """
    :param first: word to calculate distance for
    :param second: word to calculate distance with
    :param winkler: same as winkler_ajustment
    :param winkler_ajustment: add an adjustment factor to the Jaro of the distance
    :param scaling: scaling factor for the Winkler adjustment
    :return: Jaro distance adjusted (or not)
    """
    if sort_tokens:
        first = sort_token_alphabetically(first)
        second = sort_token_alphabetically(second)

    if not first or not second:
        raise JaroDistanceException(
            "Cannot calculate distance from NoneType ({0}, {1})".format(
                first.__class__.__name__,
                second.__class__.__name__))

    jaro = _score(first, second)
    cl = min(len(_get_prefix(first, second)), 4)

    if all([winkler, winkler_ajustment]):  # 0.1 as scaling factor
        return round((jaro + (scaling * cl * (1.0 - jaro))) * 100.0) / 100.0

    return jaro

def _score(first, second):
    shorter, longer = first.lower(), second.lower()

    if len(first) > len(second):
        longer, shorter = shorter, longer

    m1 = _get_matching_characters(shorter, longer)
    m2 = _get_matching_characters(longer, shorter)

    if len(m1) == 0 or len(m2) == 0:
        return 0.0

    return (float(len(m1)) / len(shorter) +
            float(len(m2)) / len(longer) +
            float(len(m1) - _transpositions(m1, m2)) / len(m1)) / 3.0

def _get_diff_index(first, second):
    if first == second:
        pass

    if not first or not second:
        return 0

    max_len = min(len(first), len(second))
    for i in range(0, max_len):
        if not first[i] == second[i]:
            return i

    return max_len

def _get_prefix(first, second):
    if not first or not second:
        return ""

    index = _get_diff_index(first, second)
    if index == -1:
        return first

    elif index == 0:
        return ""

    else:
        return first[0:index]

def _get_matching_characters(first, second):
    common = []
    limit = math.floor(min(len(first), len(second)) / 2)

    for i, l in enumerate(first):
        left, right = int(max(0, i - limit)), int(
            min(i + limit + 1, len(second)))
        if l in second[left:right]:
            common.append(l)
            second = second[0:second.index(l)] + '*' + second[
                                                       second.index(l) + 1:]

    return ''.join(common)

def _transpositions(first, second):
    return math.floor(
        len([(f, s) for f, s in zip(first, second) if not f == s]) / 2.0)

def get_top_matches(reference, value_list, max_results=None):
    scores = []
    if not max_results:
        max_results = len(value_list)
    for val in value_list:
        score_sorted = get_jaro_distance(reference, val)
        score_unsorted = get_jaro_distance(reference, val, sort_tokens=False)
        scores.append((val, max(score_sorted, score_unsorted)))
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:max_results]

class JaroDistanceException(Exception):
    def __init__(self, message):
        super(Exception, self).__init__(message)


# # Execute Jaro Functions on new df

# In[596]:


#jaro test run
df = pd.read_csv('org_norm_final.csv')
df['parent_org_name']=df['parent_org_name'].str.replace(',','')
# df.head(5).to_dict()

df['children_org_name_list'] = df.children_org_name_list.apply(literal_eval)
df['jaro_func_results'] = df.agg(lambda x: get_top_matches(*x), axis=1)
df['jaro_func_results'] = df['jaro_func_results'].apply(sorted)

#create ge cols
df['bt_80_89'] = df.jaro_func_results.apply(lambda x: [val for val in x if val[1] >= 0.8 and val[1] <= 0.89])
df['ge_75'] = df.jaro_func_results.apply(lambda x: [val for val in x if val[1] >= 0.75])
df['bt_70_79'] = df.jaro_func_results.apply(lambda x: [val for val in x if val[1] >= 0.7 and val[1] <= 0.79])
df['bt_60_69'] = df.jaro_func_results.apply(lambda x: [val for val in x if val[1] >= 0.6 and val[1] <= 0.69])
df['ge_50'] = df.jaro_func_results.apply(lambda x: [val for val in x if val[1] >= 0.50])
df['ge_25'] = df.jaro_func_results.apply(lambda x: [val for val in x if val[1] >= 0.25])

#create scores col
df['jaro_func_score'] = df.jaro_func_results.apply(lambda x: [val[1] for val in x])

df.head(20)


# In[607]:


data = {'col1':  ['MAX', 'Sam', 'Larry'],
        'col2': ["['MAX', 'amx', 'akd']", "['Sam','sammy','samsam']", "['lar','lair','larrylamo']"],
#         'func_results': ["[('MAX',1.0),('amx',0.89),('akd',0.56)]", "[('Sam',1.0),('sammy',0.91), ('samsam',0.88)]", "[('lar',0.91),('larrylamo',0.91), ('lair',0.83)]"]
        }

# df1 = pd.DataFrame (data, columns = ['col1','col2','func_results'])
df1 = pd.DataFrame (data, columns = ['col1','col2'])

df1['col2'] = df1.col2.apply(literal_eval)
df1['func_results'] = df1.agg(lambda x: get_top_matches(*x), axis=1)
df1


# In[614]:


data = {'col1':  ['abc co', 'kdj', 'bac'],
        'col2': ['AAP akj', 'fuj ddd', 'ADO asd']
        }

df3 = pd.DataFrame (data, columns = ['col1','col2'])

# df3['col1'] = df3['col1'].agg(list)
df3['func_results'] = df3.agg(lambda x: get_top_matches(*x), axis=1)

df3


# # Create Additional Fuzzy Matching Features

# In[597]:


def total_chars(row):
    
    results0 = []
    results1 = []
    results2 = []
    results3 = []
    results4 = []
    results5 = []
    results6 = []
    results7 = []
    
    p = row['parent_org_name'].lower()
    for i in row['children_org_name_list']:
        
        t = i.lower()
        
        dict1 = Counter(p)
        dict2 = Counter(t)
        
        commonDict = dict1 & dict2
        
        if len(commonDict) == 0:
            print -1
            return
    
        commonDict = list(set(dict1 & dict2)) #commonChars
        commonChars = list(set(dict1 + dict2)) #totalUniqueChars
        allChars = len(dict1) + len(dict2)
        
        results0.append((i, len(commonDict))[1]) #countCommonChars
        results1.append((i, len(commonChars))) #totalUniqueChars
        results2.append((i, allChars)[1]) #countTotalUniqueChars
        results3.append((i, sum(int(k==v) for k,v in zip(t, p)))) #samePosition
        results4.append((i, sum(int(k==v) for k,v in zip(t, p)))[1]) #countSamePosition
        results5.append((i, SequenceMatcher(None, t, p).find_longest_match(0, len(t), 0, len(p))[2])) #consecutiveMatchingChars
        results6.append((i, round(SequenceMatcher(None, t, p).ratio(),2))) #consecutiveCharsRatio #slightly off?
        results7.append((i, round(SequenceMatcher(None, t, p).ratio(),2))[1]) #sequence_matcher_ratio

    return pd.Series([results0, results1, results2, results3, results4, results5, results6, results7])

df[['count_common_chars'
    , 'total_unique_chars'
    , 'count_total_unique_chars'
    , 'same_location'
    , 'count_same_position'
    , 'consecutive_chars'
    , 'consecutive_chars_ratio'
    , 'sequence_matcher_ratio'
   ]]  = df.apply(total_chars, axis=1)

df['percentMatched'] = df.apply(lambda x: [np.round(x['count_common_chars'][i]/x['count_total_unique_chars'][i], 2) 
                                           for i in range(len(x['count_common_chars']))], axis = 1)
df['percentSamePosition'] = df.apply(lambda x: [np.round(x['count_same_position'][i]/x['count_total_unique_chars'][i], 2) 
                                                for i in range(len(x['count_common_chars']))], axis = 1)

df['jaro_sequencer_diff'] = df.apply(lambda x: [np.round(
    x['jaro_func_score'][i] - 
    x['sequence_matcher_ratio'][i],2) for i in range(len(x['jaro_func_score']))
], axis = 1)

df['jaro_sequencer_diff_sorted'] = [sorted(l) for l in df['jaro_sequencer_diff']]

df['jaro_min'] = df.jaro_func_score.apply(min)
df['jaro_max'] = df.jaro_func_score.apply(max)

df['sequence_min'] = df.sequence_matcher_ratio.apply(min)
df['sequence_max'] = df.sequence_matcher_ratio.apply(max)

df['jaro_sequencer_diff_max'] = df.jaro_sequencer_diff_sorted.apply(max)
df['jaro_sequencer_diff_min'] = df.jaro_sequencer_diff_sorted.apply(min)

df = df[['parent_org_name', 'children_org_name_list', 'jaro_func_results',
       'ge_25', 'ge_50', 'ge_75', 'bt_60_69', 'bt_70_79', 'bt_80_89', 
       'jaro_func_score', 'jaro_min', 'jaro_max',
       'sequence_matcher_ratio', 'sequence_min', 'sequence_max',
       'jaro_sequencer_diff', 'jaro_sequencer_diff_sorted',
       'jaro_sequencer_diff_min', 'jaro_sequencer_diff_max',
       'count_common_chars', 'count_total_unique_chars', 'count_same_position',
       'percentMatched', 'percentSamePosition',
       'total_unique_chars', 'same_location', 'consecutive_chars', 'consecutive_chars_ratio']]

df.head(20)


# In[615]:


df2 = df[['parent_org_name'
    , 'children_org_name_list'
    , 'jaro_func_score'
    , 'jaro_min'
    , 'jaro_max'
    , 'sequence_matcher_ratio'
    , 'sequence_min'
    , 'sequence_max'
    , 'jaro_sequencer_diff'
    , 'jaro_sequencer_diff_sorted'
    , 'jaro_sequencer_diff_min'
    , 'jaro_sequencer_diff_max'
    , 'percentSamePosition'
    , 'percentMatched'
   ]]
df2


# # Create scores/bins df

# In[30]:


#create dataframe out of scores
scores = [st for row in df.func_score for st in row]
scores_df = pd.DataFrame()
scores_df['scores'] = scores

#create value counts col
count_scores = scores_df['scores'].value_counts()
scores_df['count_scores'] = scores_df['scores'].map(count_scores)
scores_df = scores_df.sort_values(by='scores',ascending=False)

scores_df


# In[31]:


def score_bins(scores_df):
    if scores_df['scores'] >= .9:
        return '.9'
    elif scores_df['scores'] >= .8 and scores_df['scores'] < .9:
        return '.8'
    elif scores_df['scores'] >= .7 and scores_df['scores'] < .8:
        return '.7'
    elif scores_df['scores'] >= .6 and scores_df['scores'] < .7:
        return '.6'
    elif scores_df['scores'] >= .5 and scores_df['scores'] < .6:
        return '.5'
    elif scores_df['scores'] >= .4 and scores_df['scores'] < .5:
        return '.4'
    elif scores_df['scores'] >= .3 and scores_df['scores'] < .4:
        return '.3'
    elif scores_df['scores'] >= .2 and scores_df['scores'] < .3:
        return '.2'
    elif scores_df['scores'] >= .1 and scores_df['scores'] < .2:
        return '.1'
    else:
        return '0'
    
scores_df['scores_bins'] = scores_df.apply(score_bins, axis=1)
scores_df


# # Visualize Jaro Score Bin Breakdown

# In[40]:


#scores_df['scores'].value_counts(normalize=True) * 100
fig = px.pie(scores_df
             , values='scores'
             , names='scores'
             , title='Scores Breakdown')

layout = go.Layout(
                 legend={'traceorder':'normal'}
)

# fig.update_traces(textposition='inside'
#                   , textinfo='percent+label')

fig.show()


fig = px.pie(scores_df
             , values='scores_bins'
             , names='scores_bins'
             , title='Scores Bins Breakdown')

layout = go.Layout(
                 legend={'traceorder':'normal'}
)

fig.update_traces(textposition='inside'
                  , textinfo='percent+label')

fig.show()


# In[181]:


#plot distribution
fig = px.scatter(scores_df
                 , x="scores"
                 , y="count_scores"
                 , color="scores"
                 , size='count_scores'
                 , hover_data=['count_scores','scores']
                )

fig.update_layout(title='<b>Function Score Distribution</b>')

fig.show()


# # Notes and Recommendations

# ## Fuzzy Matching Metrics
# ##### ~60% of fuzzy matches are greater than or equal to a Jaro score of .7
# ##### ~30% are greater than or equal to a Jaro score of .9
# ##### Only ~18% of fuzzy matches are less than or equal to a Jaro score of .5
# 
# ## Jaro .6 Scores
# ##### .6 - .69 Jaro scores appear to only have 2-3 letters in common
# ##### No matches appear to be correct within this bin thus no opportunities observed
# 
# ## Jaro .7 Scores
# ##### .7 - .79 Jaro scores appear to only have 4-5 letters in common and seem to disregard being in exact order
# ##### Several matches appear to be actual matches within this bin (3/20)
# 
# ## .7 - .79 Potential Matches: 
# ##### NK-NET OOO,	[('LLC NK-NET', 0.78)]
# ##### SORAIRE MARKETING,	[('SOLAISE CAPITAL ECINET', 0.74)
# ##### TYROLIT AUSTRALIA PTY LTD,	[('REASSIGN TO TYROLIT ASIA PACIFIC LTD.', 0.77)]
# 
# ## Jaro .8 Scores
# ##### .8 - .89: matches appear to be pretty decent - over 5-7 letter matches, similar name lengths; over 50% coule be matches; difficult for humans to even predict accuracy without research.
# ##### .8 - .89 some inputs could be automated spelling corrections (apple, andriod, os etc.):
# 
# ## .8 - .89 Potential matches:
# ##### DURAMETAL LTDA,	[('DURAND DO BRASIL LTDA', 0.82)]
# ##### GODOSOFT,	[('GOLDSOFT S.A.', 0.81)]
# ##### NK-NET OOO,	[('NK-NETT AS', 0.87)]
# 
# ## Opportunities
# ##### .7 - .79: words with 5-7 consecutive letters to be given higher scores
# ##### .7 - .79: matches with more than 1 complete word match to be given higher scores
# ##### .7 - .79: matches with dramatically different lengths to be given lower scores
# 
# ## Opportunities based on Jaro Score vs. Sequencer Ratio
# ##### Jaro rewards one word matches, regardless of string length difference; when jaro is much greater than sequencer, typically it's due to a word matching, but very different lengths
# ##### Example: HPCL	['HPCL', 'HPCL MITTAL ENERGY LIMITED VILLAGE PHULLO KHARI RAMAN DISTT BATHINDA', 'HPCL-MITTAL ENERGY LTD.'] Jaro: [1.0, 0.81, 0.83]	Sequencer:	[1.0, 0.11, 0.3] (second element)
# ##### Sequencer seems to reward better when matches are in the middle of the strings
# ##### Example: NHOTELS	['D A NHOTEL SP', 'NHOTELS', 'STIFTUNG NHTLZ BBC ARENA'] Jaro: [0.0, 1.0, 0.39] Sequencer: [0.7, 1.0, 0.26] (first element)
# 

# # Jaro Test Code

# In[200]:


#jaro test with small dataframe
df1 = pd.read_csv('test_csv.csv')
df1['children'] = df1.children.apply(literal_eval)

# df1['children'] = df1['children'].map(lambda x: x.lstrip('"\'').rstrip('"\''))

# new_list = list(df1['children'])
# df1['children'] = new_list

df1['func_results'] = df1.agg(lambda x: get_top_matches(*x), axis=1)

df1

df1.head(5).to_dict()

print(df1.applymap(type))

a = (df1.applymap(type) == list).all()
print(a)

df1.children.apply(literal_eval)


# # Test Common Characters Function

# In[292]:


# Test Function:
# Total Characters
# Characters in Same Position
# % Same Position
# Common Characters
# % Characters Matched
# Consecutive Matching Characters
# Consecutive Characters Ratio --> ge .6 eq interesting match
# Matching Characters

from collections import Counter
from itertools import groupby

def commonChars(first,second):
    dict1 = Counter(first)
    dict2 = Counter(second)

    totalChars = len(set(first + second))
    
    samePosition = sum(1 if c1 == c2 else 0 for c1, c2 in zip('first', 'second'))
    
    consecutiveMatchingChars = SequenceMatcher(None, first, second).find_longest_match(0, len(first), 0, len(second))
    
    consecutiveCharsRatio = SequenceMatcher(None, first, second).ratio()

    commonDict = dict1 & dict2

    if len(commonDict) == 0:
        print -1
        return

    commonChars = list(commonDict.elements())
    commonChars = sorted(commonChars)

    countCommonChars = len(commonChars)
    percentMatched = countCommonChars / totalChars
    percentSamePosition = samePosition / totalChars
    
    print ('Total Unique Characters: ', totalChars)
    print ('Characters in Same Position: ', samePosition)
    print ('% Same Position: ', percentSamePosition)
    print ('Count Common Characters: ', countCommonChars)
    print ('% Characters Matched: ', round(percentMatched,2))
    print ('Consecutive Matching Characters: ', consecutiveMatchingChars[2])
    print ('Consecutive Characters Ratio: ', round(consecutiveCharsRatio,2))
    print ('Matching Characters: ', ''.join(commonChars))


# # Code Testing

# In[293]:


commonChars('max','max llc')


# In[451]:


def total_chars(row):
    results0 = []
    results1 = []
    results2 = []
    results3 = []
    results4 = []
    results5 = []
    results6 = []
    p = row['parent_org_name'].lower()
    for i in row['children_org_name_list']:
        t = i.lower()
        
        dict1 = Counter(p)
        dict2 = Counter(t)
        
        commonDict = dict1 & dict2
        
        if len(commonDict) == 0:
            print -1
            return
        
#         commonChars = list(commonDict.elements())
#         commonChars = list(set(commonChars))
    
        commonDict = list(set(dict1 & dict2)) #commonChars
        commonChars = list(set(dict1 + dict2)) #totalUniqueChars
        allChars = len(dict1) + len(dict2)
        
        results0.append((i, len(commonDict))[1]) #countCommonChars
#         results1.append((i, len(set(t + p)))) #totalUniqueChars
        results1.append((i, len(commonChars))) #totalUniqueChars
#         results2.append((i, len(set(t + p)))[1]) #countTotalUniqueChars
        results2.append((i, allChars)[1]) #countTotalUniqueChars
    
        results3.append((i, sum(int(k==v) for k,v in zip(t, p)))) #samePosition
        results4.append((i, sum(int(k==v) for k,v in zip(t, p)))[1]) #countSamePosition
        results5.append((i, SequenceMatcher(None, t, p).find_longest_match(0, len(t), 0, len(p))[2])) #consecutiveMatchingChars
        results6.append((i, round(SequenceMatcher(None, t, p).ratio(),2))) #consecutiveCharsRatio #slightly off? 

    return pd.Series([results0, results1, results2, results3, results4, results5, results6])

df[['count_common_chars'
    , 'total_chars'
    , 'count_total_chars'
    , 'same_location'
    , 'count_same_location'
    , 'consecutive_chars'
    , 'consecutive_chars_ratio'
   ]]  = df.apply(total_chars, axis=1)

df['percentMatched'] = df.apply(lambda x: [np.round(x['count_common_chars'][i]/x['count_total_chars'][i], 2) 
                                           for i in range(len(x['count_common_chars']))], axis = 1)
df['percentSamePosition'] = df.apply(lambda x: [np.round(x['count_same_location'][i]/x['count_total_chars'][i], 2) 
                                                for i in range(len(x['count_common_chars']))], axis = 1)

df


# In[586]:


round(SequenceMatcher(None, 'IEM,','IEM').ratio(),2)


# In[404]:


df[['parent_org_name','children_org_name_list','count_common_chars','count_total_chars']]


# In[443]:


dict1 = Counter('YDEA S.R.L')
dict2 = Counter('YD CONFECCOES LTDA')

# results0.append((i, len(commonChars))[1]) #countCommonChars
# results1.append((i, len(set(t + p)))) #totalUniqueChars
# results2.append((i, len(set(t + p)))[1]) #countTotalUniqueChars
        
commonDict = list(set(dict1 & dict2)) #commonChars
commonChars = list(dict1 + dict2) #totalUniqueChars
allChars = len(dict1) + len(dict2)

print(dict1)
print(dict2)

print('common chars: ', commonDict)
print('length common chars: ', len(commonDict))

print('total unique chars: ', commonChars)
print('count total unique chars: ', len(commonChars))

print(allChars)


# In[431]:


del(commonChars)


# In[403]:


df.head(20)


# In[276]:


#consecutive chars ratio
round(SequenceMatcher(None, 'LTD YURIA-PHARM','LTD YURIA-PHARM').ratio(),2)


# In[263]:


#consecutive chars
SequenceMatcher(None, 'MJN ENTERPRISES', 'MJM INTERANTIONAL INC').find_longest_match(0, len('MJN ENTERPRISES'), 0, len('MJM INTERANTIONAL INC'))[2]


# In[243]:


#same position
sum(1 if c1 == c2 else 0 for c1, c2 in zip('HYONIX', 'HYMAX TALKING SOLUTIONS'))


# In[386]:


#distinct chars
len(set('JAARBEURS B.V.' + 'JAARBEURS B.V.'))


# In[283]:


len(sorted('max'))


# In[282]:


set('max')


# In[493]:


df.applymap(type)


# In[593]:


#New Logic 2020-08-14
from collections import defaultdict

lkp_dict = {'COLUMBUS':1,'CHICAGO':1,'SCHOOL':0.1,'DISTRICT':0.1}

lkp_dict_def = defaultdict(float,lkp_dict )

score_list = []

for child_org_name in child_org_name_list:
    for child_token in child_org_name.split(' '):
        temp_score_list = []
        max_score = 0
        for parent_token in parent_org_name.split(' '):
            print(child_token, parent_token)
            if lkp_dict_def[child_token] < 0.5 and lkp_dict_def[parent_token] < 0.5:
                temp_score = 0.1
            else:
                #weight = lkp_dict_def[child_token]
                weight = 1
                temp_score = SequenceMatcher(None, parent_token,child_token).ratio() * weight            
            print(temp_score)
            max_score = max(max_score,temp_score)
           
        score_list.append(max_score)

def mean_func(lst):
    score_sum, count = 0, 0
    for score in lst:
        if score >= 0.4:
            count += 1
            score_sum += score
 
    if count > 0:
        return score_sum / count
    else:
        return 0   


# In[590]:


df2.to_csv('org_norm_results_new.csv')


# In[591]:


for child_org_name in child_org_name_list:
    for child_token in child_org_name.split(' '):
        temp_score_list = []
        max_score = 0
        for parent_token in parent_org_name.split(' '):
            print(child_token, parent_token)
            if lkp_dict_def[child_token] < 0.5 and lkp_dict_def[parent_token] < 0.5:
                temp_score = 0.1
            else:
                #weight = lkp_dict_def[child_token]
                weight = 1
                temp_score = SequenceMatcher(None, parent_token,child_token).ratio() * weight            
            print(temp_score)
            max_score = max(max_score,temp_score)
           
        score_list.append(max_score)


# In[ ]:




