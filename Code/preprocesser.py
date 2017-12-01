import json, os
import subprocess
import pandas as pd
import nltk

cwd = os.getcwd()

PARSER_DIR = "/Users/Daisy/dev-tools/corenlp/stanford-parser-full-2017-06-09/"
TAGGER_DIR = "/Users/Daisy/dev-tools/corenlp/stanford-postagger-full-2015-04-20/"
PROJECT_DIR = os.path.dirname(cwd)
DATA_DIR = os.path.join(PROJECT_DIR, 'Data/')
REVIEW_CSV = os.path.join(DATA_DIR, 'Table_17_sample.csv')
REVIEW_TXT = os.path.join(DATA_DIR, 'Table_17_sent.txt')
OUTPUT_POSTAG = os.path.join(DATA_DIR, 'postag.json')
OUTPUT_PARSED_TREE = os.path.join(DATA_DIR, 'parse_trees.json')

# Tokenize review sentences and write to a txt file
if not os.path.isfile(REVIEW_TXT):
    df = pd.read_csv(REVIEW_CSV, encoding='latin-1')
    reviews = df['review'].values
    with open(os.path.join(DATA_DIR, REVIEW_TXT), 'w') as f:
        for review in reviews:
            sentences = nltk.sent_tokenize(review)
            for sent in sentences:
                f.write(sent + '\n')

# Run pos tag and save to a file
# subprocess.call("java -mx300m -cp " + TAGGER_DIR + "stanford-postagger.jar" + \
#   ": edu.stanford.nlp.tagger.maxent.MaxentTagger -sentenceDelimiter newline " + \
#   "-model " + TAGGER_DIR + "models/wsj-0-18-bidirectional-distsim.tagger -textFile " + \
#   REVIEW_TXT + " -outputFile " + OUTPUT_POSTAG, shell=True)

# Run parser and save to a file
# subprocess.call("java -mx1500m -cp \"*\"" + \
#    " edu.stanford.nlp.parser.lexparser.LexicalizedParser " + \
#   "-outputFormat \"penn\" -sentences newline " + \
#   "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz -textFile " + \
#   REVIEW_TXT + " -outputfile " + OUTPUT_PARSED_TREE, shell=True)

subprocess.call(PARSER_DIR + "lexparser.sh " + REVIEW_TXT + " > " + OUTPUT_PARSED_TREE, shell=True)
