import nltk
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def chunk_sentence(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def pair_features(sentence):
    pairs = []
    i = 1

    while i < len(sentence):
        if sentence[i][1] == 'NN':
            if sentence[i-1][1] == 'JJ':
                pairs.append([sentence[i-1][0], sentence[i][0]])
        i += 1
    return pairs

def robust_decode(bs):
    '''Takes a byte string as param and convert it into a unicode one.
First tries UTF8, and fallback to Latin1 if it fails'''
    cr = None
    try:
        cr = bs.decode('utf8')
    except UnicodeDecodeError:
        cr = bs.decode('latin1')
    return cr


if __name__ == '__main__':
    filepath = "./pairs.txt"
    adj_np_pairs = open(filepath, "w")
    pairs = []


    # Input
    with open('Table_17_subjective.txt') as f:
        read_data = f.read()
        sentences = chunk_sentence(read_data)

        for sentence in sentences:
            pairs.extend(pair_features(sentence))

    for pair in pairs:
        stringPair = pair[0]+ " "+pair[1]
        adj_np_pairs.write(stringPair + '\n')

    adj_np_pairs.close()