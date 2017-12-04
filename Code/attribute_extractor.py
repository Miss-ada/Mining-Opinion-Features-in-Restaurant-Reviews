import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import json
import nltk
from nltk.tree import Tree
from collections import Counter, OrderedDict


MIN_ATTRI_CT = 5

def extract_attribute_opinion_pair(file):
    """
    Extract attribute_opinion with count in following form
    {attri1: 
        {opinion1: 1, opinion2: 1}
     attrib2: 
        {opinion1: 1, opinion2: 1, opinion3: 2}
      ....
        }
    """
    candidates = []
    nps = []
    attri_opinion = {}
    with open(file) as f:
        for par in f.read().split('\n\n'):
            if len(par.strip()) != 0:
                tree = Tree.fromstring(par)
                noun_phrases = extract_noun_phrases(tree)
                for np in noun_phrases:
                    if len(np) != 0:
                        opinion = extract_opinion_word(np, tree)
                        if len(opinion) != 0:
                            nps.append(np)
                            build_dict(attri_opinion, np, opinion)
    # Sort opinion according to its count in reversed order
    for key, value in attri_opinion.items():
        attri_opinion[key] = OrderedDict(sorted(value.items(), key=lambda d: d[1], reverse=True))

    nps_count = Counter(nps)
    # Sorted attribute in reversed order
    nps_count = nps_count.most_common()
    # print(nps_count)
    print('Length of all np is: ', len(nps_count))
    for attri, ct in nps_count:
        if ct >= MIN_ATTRI_CT:
            candidates.append((attri, attri_opinion[attri]))
    print('Length of candidates attributes is: ', len(candidates))
    return candidates


def build_dict(dictionary, np, opinion):
    """
     Construct nested dict with np, opinion and opinion count
    """
    if np not in dictionary:
        dictionary[np] = {}
        dictionary[np][opinion] = 1
    elif opinion not in dictionary[np]:
        dictionary[np][opinion] = 1
    else:
        dictionary[np][opinion] += 1



def extract_opinion_word(np, tree):
    """
    Find opinion word closed to np
    """
    sentence = ' '.join(token.lower() for token in tree.leaves())
    np_index = sentence.find(np)

    stop = set(stopwords.words('english'))
    desired_np = ''
    min_dist = len(sentence)
    for subtree in tree.subtrees(filter=jj_filter):
        answer = " ".join(token.lower() for token in subtree.leaves())

        # # remove stopwords
        if (len(word_tokenize(answer)) == 1 and answer.lower() in stop):
            continue
        opinion_index = sentence.find(answer)
        if opinion_index > np_index:
            np_index = np_index - len(answer)
        else:
            opinion_index = opinion_index - len(np)
        dist = abs(np_index - opinion_index)
        if min_dist > dist:
            min_dist = dist
            desired_np = answer
    return desired_np


def extract_noun_phrases(tree):
    """
    Extract noun phrases using parsed tree
    """
    stop = set(stopwords.words('english'))
    np = []
    for subtree in tree.subtrees(filter=np_filter):
        answer = " ".join(token.lower() for token in subtree.leaves())

        # remove stopwords
        if (len(word_tokenize(answer)) == 1 and answer.lower() in stop):
            continue
        np.append(answer)
    return np

# def chunk_sentence(document):
#     sentences = nltk.sent_tokenize(document)
#     sentences = [nltk.word_tokenize(sent) for sent in sentences]
#     sentences = [nltk.pos_tag(sent) for sent in sentences]
#     return sentences

def np_filter(subtree):
    return subtree.label() == 'NP'

def jj_filter(subtree):
    return subtree.label() == 'JJ'

# def pair_features(sentence):
#     pairs = []
#     i = 1
#
#     while i < len(sentence):
#         if sentence[i][1] == 'NN':
#             if sentence[i-1][1] == 'JJ':
#                 pairs.append([sentence[i-1][0], sentence[i][0]])
#         i += 1
#     return pairs

if __name__ == '__main__':
    parsed_file = '../Data/Table_17_subjective_parser.json'
    candidates = extract_attribute_opinion_pair(parsed_file)

    with open('../Data/candidates_attributes_subjective.json', 'w') as f:
        json.dump(candidates, f)


