from stanfordcorenlp import StanfordCoreNLP
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import json
import nltk


def extract_noun_phrases(tree):
    stop = set(stopwords.words('english'))
    np = []
    for subtree in tree.subtrees(filter=np_filter):
        answer = " ".join(token for token in subtree.leaves())

        # remove stopwords
        if (len(word_tokenize(answer)) == 1 and answer.lower() in stop):
            continue
        np.append(answer)
    return np

def chunk_sentence(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def np_filter(subtree):
    return subtree.label() == 'NP'

def pair_features(sentence):
    pairs = []
    i = 1

    while i < len(sentence):
        if sentence[i][1] == 'NN':
            if sentence[i-1][1] == 'JJ':
                pairs.append([sentence[i-1][0], sentence[i][0]])
        i += 1
    return pairs

if __name__ == '__main__':
    filepath = "./pairs.txt"
    adj_np_pairs = open(filepath, "w")

    corenlp = StanfordCoreNLP('http://localhost')
    # Input
    df = pd.read_csv('Table17_sample.csv', encoding = 'latin-1')
    reviews = df['review'].values


    pairs = []


    # Define your output here
    for review in reviews:

        output = corenlp.annotate(review,
                          properties={'annotators': 'tokenize,pos,parse,sentiment',
                          'outputFormat': 'json'})
        output = json.loads(output)
        # extract noun phrase
        sentences = chunk_sentence(review)
        # parsed_string = output['sentences'][0]['parse']
        # parsed_tree = Tree.fromstring(parsed_string)
        grammar = "NP: {<JJ> <NN>}"
        cp = nltk.RegexpParser(grammar)
        for sentence in sentences:
            pairs.extend(pair_features(sentence))

    adj_np_pairs.write(pairs + '\n')


    # # extract others
    #     dic['sentiment_value'].append(output['sentences'][0]['sentimentValue'])
    #     dic['sentiment'].append(output['sentences'][0]['sentiment'])
    #     dic['token'].append(output['sentences'][0]['tokens'])
    # #print('sentiment: {0}\nsentiment value: {1}\ntokens: {2}\n'.format(sentiment, sentiment_value, tokens))
    #
    # result_frame = pd.DataFrame(dic)