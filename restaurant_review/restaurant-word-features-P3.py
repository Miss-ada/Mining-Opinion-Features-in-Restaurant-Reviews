import re, nltk, random, pickle, sys
from nltk.corpus import stopwords

def get_score(review):
    return int(re.search(r'Overall = ([1-5])', review).group(1))


def get_text(review):
    return re.search(r'Text = "(.*)"', review).group(1)


# Normalized data and tag
def process_reviews(file_name):
    file = open(file_name, "rb")
    raw_data = file.read().decode("latin1")
    file.close()
    # print(raw_data)
    positive_texts = []
    negative_texts = []
    first_sent = None
    for review in re.split(r'\.\n', raw_data):
        overall_score = get_score(review)
        # print(overall_score)
        review_text = get_text(review)
        # print(review_text)
        if overall_score > 3:
            # label = 'positive'
            positive_texts.append(review_text)
        elif overall_score < 3:
            # label = 'negative'
            negative_texts.append(review_text)
            # if first_sent == None:
            #    sent = nltk.sent_tokenize(review_text)
            #    if (len(sent) > 0):
            #        first_sent = sent[0]

    # There are 150 positive reviews and 150 negative reviews.
    # print(positive_texts)
    # Your code goes here

    # Normailze data
    nor_pos_words = nor_texts(positive_texts)
    nor_neg_words = nor_texts(negative_texts)

    # Label positive reviews
    pos_label_review = label_review(positive_texts, nor_pos_words, "positive")
    neg_label_review = label_review(negative_texts, nor_neg_words, "negative")

    # reviews = []
    reviews = pos_label_review + neg_label_review
    #print(reviews)
    return reviews


def label_review(text, words, label):
    reviews = []
    for i in range(0, len(text)):
        reviews.append([text[i], words[i], label])
    return reviews


# Normalized data: lowercased , remove stopwords, remove words with 1 char
# Return a list of list, each list is a review with words
def nor_texts(texts):
    stopwords = nltk.corpus.stopwords.words('english')
    word_list = []
    for sents in texts:
        words = nltk.word_tokenize(sents)
        word_str = ' '.join(w for w in words)
        pat = r'\w+' # remove words with only 1 char
        nor_word_list = re.findall(pat, word_str)
        word_list.append([w.lower() for w in nor_word_list if w.lower() not in stopwords])
        #word_list.append([w.lower() for w in nor_word_list])
    # print(word_list)

    return word_list


# Adds word_features and word_pos_features(optional)
# Return the feature vector
def lexical_features(review_text, review_words, vocabulary):
    feature_vector = {}

    # remove punc and lower the whole sentence in each review for bigrams
    lower_review_text = ''.join(word.lower() for word in review_text)
    pat = r'\w+'
    nor_review_text = re.findall(pat, lower_review_text)


    # add word_features on unigrams, bigrams and trigrams
    uni_dist = nltk.FreqDist(review_words)

    bi_list = list(nltk.bigrams(nor_review_text))
    bi_dist = nltk.FreqDist(bi_list)

    tri_list = list(nltk.trigrams(nor_review_text))
    tri_dist = nltk.FreqDist(tri_list)

    ngrams = list(set(review_words) | set(bi_list) | set(tri_list))

    # Modify L102-106 here to add word_features
    #add_word_features_absolute(uni_dist, bi_dist, tri_dist, feature_vector)

    add_word_features_binary(ngrams, vocabulary, feature_vector)

    #add_word_features_relative(uni_dist, bi_dist, tri_dist, feature_vector)

    #print(feature_vector)
    return feature_vector


def bin(freq):
    # Just a wild guess on the cutoff
    return freq if freq < 3 else 5


# Improved relative frequency using binning
def add_word_features_relative(uni_dist, bi_dist, tri_dist, feature_vector):
    # add unigram features with relative freq
    for word, freq in uni_dist.items():
        fname = "UNI_{0}".format(word)
        feature_vector[fname] = float(bin(freq)) / uni_dist.N()

    # add bigram features with relative freq
    for word, freq in bi_dist.items():
        fname = "BIGRAM_" + word[0] + "_" + word[1].format(word)
        feature_vector[fname] = float(bin(freq)) / bi_dist.N()

    # add trigram features with relative freq
    for word, freq in tri_dist.items():
        fname = "TRIGRAM_" + word[0] + "_" + word[1] + "_" + word[2].format(word)
        feature_vector[fname] = float(bin(freq)) / tri_dist.N()


# Add unigrams, bigrams, trigrams with absolute count
def add_word_features_absolute(uni_dist, bi_dist, tri_dist, feature_vector):
    # add unigram features with absoulute freq
    for word, freq in uni_dist.items():
        fname = "UNI_{0}".format(word)
        feature_vector[fname] = freq

    # add bigram features with absolute freq
    for word, freq in bi_dist.items():
        fname = "BI_" + word[0] + "_" + word[1].format(word)
        feature_vector[fname] = freq

    # add trigram features with absolute freq
    for word, freq in tri_dist.items():
        fname = "TRI_" + word[0] + "_" + word[1] + "_" + word[2].format(word)
        feature_vector[fname] = freq


# Get binary feature of words (high accuracy)
# def add_word_features_binary(ngrams, vocabulary, feature_vector):
#     for word in ngrams:
#         fname = "CONTAINS({})".format(word)
#         feature_vector[fname] = (word in vocabulary)


# Get binary feature of words, required by assignment (low accuracy)
def add_word_features_binary(ngrams, vocabulary, feature_vector):
    for word in vocabulary:
        fname = "CONTAINS({})".format(word)
        feature_vector[fname] = (word in ngrams)



# Create uni+bi+tri grams vocabulary
def get_vocabulary(reviews):
    unigrams = []
    bigrams = []
    trigrams = []
    for (i, tuple) in enumerate(reviews):
        unigrams += tuple[1]
        sent_lower = ''.join(word.lower() for word in tuple[0])
        pat = r'\w+'
        sent_nor = re.findall(pat, sent_lower)
        bigrams += nltk.bigrams(sent_nor)
        trigrams += nltk.trigrams(sent_nor)
    voc = set(unigrams) | set(bigrams) | set(trigrams)
    return voc


def evaluate(model, target_file, output_file, vocabulary):
    # Output file uses append mode for continuing writing
    fh = open(output_file, 'a', encoding = 'utf-8')

    # Load trained model
    f = open(model, 'rb')
    classifier = pickle.load(f)
    f.close()

    # Write in target_file name
    fh.write("{0} classifier on {1}\n\n".format(model, target_file))

    # Get featuresets of target file
    reviews_test = process_reviews(target_file)
    reference_text = [review[0] for review in reviews_test]
    test_data = [(lexical_features(review_text, review_words, vocabulary), label)
                for (review_text, review_words, label) in reviews_test]


    accuracy = nltk.classify.accuracy(classifier, test_data)
    print(accuracy)
    fh.write("{0:10s} {1:8.5f}\n\n".format("Accuracy", accuracy))

    # Write in confusion matrix
    features_only = []
    reference_labels = []
    for feature_vectors, category in test_data:
        features_only.append(feature_vectors)
        reference_labels.append(category)

    predicted_labels = classifier.classify_many(features_only)

    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)
    fh.write(str(confusion_matrix))
    fh.write('\n\n')

    # Write the predications
    for reference, predicated, text in zip(
            reference_labels, predicted_labels, reference_text):
        #if reference != predicated:
            fh.write("reference_{0} predicated_{1}\n{2}\n\n".format(reference, predicated, text))



if __name__ == '__main__':
    # Get the vocabulary based on training data
    file_name_train = "restaurant-training.data"
    reviews_train = process_reviews(file_name_train)
    vocabulary = get_vocabulary(reviews_train)

    # Classify reviews using trained models, command line arg L39-241 Or 243-247
    # First arg = model (one of restaurant-word features-model-P1.pickle or restaurant-word pos features-model-P1.pickle)
    # The second should be the file with the reviews in it to be classified.
    # The third should be the output file to write the classification results.

    classifier_model = sys.argv[1]
    target_file = sys.argv[2]
    output_file = sys.argv[3]

    #classifier_model = "nb-word_features-absolute-model-P3.pickle"
    #classifier_model = "nb-word_features-binary-model-P3.pickle"
    #classifier_model = "nb-word_features-relative-improved-model-P3.pickle"
    #target_file = "restaurant-development.data"
    #output_file = "out-dev-word_features-binary.txt"


    evaluate(classifier_model, target_file, output_file, vocabulary)


