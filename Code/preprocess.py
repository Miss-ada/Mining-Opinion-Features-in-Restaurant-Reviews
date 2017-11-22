from pycorenlp import StanfordCoreNLP
from nltk.tree import Tree
import pandas as pd
import os


def extract_noun_phrases(tree):
    np = []
    for subtree in tree.subtrees(filter=np_filter):
        answer = " ".join(token for token in subtree.leaves())
        np.append(answer)
    return np


def np_filter(subtree):
    return subtree.label() == 'NP'

if __name__ == '__main__':
    corenlp = StanfordCoreNLP('http://localhost:9000')

    # Read input
    data_dir = '../'
    input_file = 'Table\ 17'
    df = pd.read_csv


    text = 'A boy is looking out with a telescope'
    # Define your output here
    output = corenlp.annotate(text,
                          properties={'annotators': 'tokenize,pos,parse,sentiment',
                          'outputFormat': 'json'})

    # extract noun phrase
    parsed_string = output['sentences'][0]['parse']
    parsed_tree = Tree.fromstring(parsed_string)
    np = extract_noun_phrases(parsed_tree)
    print('noun phrases are: ', np)

    # extract others
    sentiment_value = output['sentences'][0]['sentimentValue']
    sentiment = output['sentences'][0]['sentiment']
    # tokens = output['sentences'][0]['tokens']
    # print('sentiment: {0}\nsentiment value: {1}\ntokens: {2}\n'.format(sentiment, sentiment_value, tokens))
