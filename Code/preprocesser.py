import json, os
import subprocess
import pandas as pd
import nltk
"""
author: Wen Cui
Date: 11/30/17
1. turn REVIEW.CSV into file with a sentence per line save to REVIEW_TXT
2. run standford pos tagger and parser on REVIEW_TXT and save to json files
"""

cwd = os.getcwd()

PARSER_DIR = "/Users/Daisy/dev-tools/corenlp/stanford-parser-full-2017-06-09/"
TAGGER_DIR = "/Users/Daisy/dev-tools/corenlp/stanford-postagger-full-2017-06-09/"
PROJECT_DIR = os.path.dirname(cwd)
DATA_DIR = os.path.join(PROJECT_DIR, 'Data/')
REVIEW_CSV = os.path.join(DATA_DIR, 'Table_17_sample.csv')
REVIEW_SENT = os.path.join(DATA_DIR, 'Table_17_subjective.csv')
OUTPUT_POSTAG = os.path.join(DATA_DIR, 'Table_17_subjective_postag.json')
OUTPUT_PARSED_TREE = os.path.join(DATA_DIR, 'Table_17_subjective_parser.json')

#  1. Tokenize review sentences and write to a txt file
# if not os.path.isfile(REVIEW_SENT):
df = pd.read_csv(REVIEW_CSV, encoding='latin-1')
reviews = df['review'].values
review_sents = []

for review in reviews:
    sentences = nltk.sent_tokenize(review)
    for sent in sentences:
        if len(sent.strip()) != 0:
            review_sents.append(sent.strip())
df1 = pd.DataFrame(review_sents)
df1.to_csv(REVIEW_SENT, index=False, header=None)

# 2. Run pos tag and save to a file
subprocess.call("java -mx1500m -cp " + TAGGER_DIR + "stanford-postagger.jar" + \
   ": edu.stanford.nlp.tagger.maxent.MaxentTagger -sentenceDelimiter newline " + \
   "-model " + TAGGER_DIR + "models/wsj-0-18-bidirectional-distsim.tagger -textFile " + \
   REVIEW_SENT + " -outputFile " + OUTPUT_POSTAG, shell=True)

# 2. Run parser and save to a file
subprocess.call(PARSER_DIR + "lexparser.sh " + REVIEW_SENT + " > " + OUTPUT_PARSED_TREE, shell=True)#