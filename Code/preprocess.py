from pycorenlp import StanfordCoreNLP
from nltk.tree import Tree
import pandas as pd
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize


def extract_noun_phrases(tree):
    stopWords = set(stopwords.words('english'))
    np = []
    for subtree in tree.subtrees(filter=np_filter):
        answer = " ".join(token for token in subtree.leaves())
        # Remove stopwords if only one word
        if len(word_tokenize(answer)) == 1 and answer.lower() in stopWords:
            continue
        np.append(answer)
    return np


def np_filter(subtree):
    return subtree.label() == 'NP'

if __name__ == '__main__':
    corenlp = StanfordCoreNLP('http://localhost:9000')

    # Read input
    data_dir = '../Data/'
    input_file = os.path.join(data_dir, 'Table_17_sample.csv')
    df = pd.read_csv(input_file, encoding='latin-1')
    reviews = df['review'].values

    np_list = []
    for review in reviews:
    # Define your output here
        output = corenlp.annotate(review,
                              properties={'annotators': 'tokenize,pos,parse,sentiment',
                              'outputFormat': 'json'})

        # extract noun phrase
        parsed_string = output['sentences'][0]['parse']
        parsed_tree = Tree.fromstring(parsed_string)
        np_list += extract_noun_phrases(parsed_tree)

        # extract others
        # sentiment_value = output['sentences'][0]['sentimentValue']
        # sentiment = output['sentences'][0]['sentiment']
        # tokens = output['sentences'][0]['tokens']
        # print('sentiment: {0}\nsentiment value: {1}\ntokens: {2}\n'.format(sentiment, sentiment_value, tokens))
    np_count = Counter(np_list)
    print(np_count.most_common(10))