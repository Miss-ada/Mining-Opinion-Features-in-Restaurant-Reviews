import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from subprocess import call
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, LabelBinarizer
import pickle, os

"""
Author: Wen Cui
Date: 12/3/17
Implement Decision Tree Attribute classification 
Table 17 - subjective/not:  514/1200, att: 21/49
"""

def encode_label(labels, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(labels)
    encoded_y = encoder.transform(labels)
    # # class_names = encoder.inverse_transform(label_list)
    # # encoded_Ytest = encoder.transform(y_test)
    # # convert integers to dummy variables (i.e. one hot encoded)
    # y = MultiLabelBinarizer().fit_transform(list(encoded_y))
    # labels = to_categorical(np.asarray(labels))
    return encoded_y, encoder


def decode_label(y, encoder):
    return encoder.inverse_transform(y)


def vectorize_text(texts, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer(decode_error='ignore', stop_words='english', min_df=2)
        X = vectorizer.fit_transform(texts).toarray()
    else:
        X = vectorizer.transform(texts).toarray()
    X = tfidf_transform(X)
    return X, vectorizer


def tfidf_transform(counts):
    transformer = TfidfTransformer(norm='l2',smooth_idf=False)
    tfidf = transformer.fit_transform(counts).toarray()
    return tfidf

def process_text(file):
    df = pd.read_csv(file)
    attr = df['attribute']
    label = df['label']
    return attr, label


if __name__ == '__main__':
    # Define directory
    data_dir = '../Data/labelled_data'
    model_dir = '../Data/classification_result'
    train_file = os.path.join(data_dir, 'train.csv')
    out_vec_file = os.path.join(model_dir, 'vectorizer.pkl')
    out_encoder_file = os.path.join(model_dir, 'encoder.pkl')
    out_model_file = os.path.join(model_dir, 'decision_tree.pkl')
    report_file = os.path.join(model_dir, 'dt_report.txt')

    # turn text into matrix X and vectors y
    attributes, labels = process_text(train_file)
    X, vectorizer = vectorize_text(attributes)
    y, encoder = encode_label(labels)

    # Single split
    # X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1, test_size=0.2)

    # 10 fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=1)

    accuracies = []
    f1s = []
    best_accuracy = 0
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]


        # Training
        model = tree.DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_predict = model.predict(X_val)


        # Get result
        accuracy = accuracy_score(y_val, y_predict)
        f1 = f1_score(y_val, y_predict, average='weighted')
        accuracies.append(accuracy)
        f1s.append(f1)


        if accuracy > best_accuracy:
            # Save best model
            with open(out_vec_file, 'wb') as vec:
                pickle.dump(vectorizer, vec, protocol=pickle.HIGHEST_PROTOCOL)
            with open(out_encoder_file, 'wb') as le:
                pickle.dump(encoder, le, protocol=pickle.HIGHEST_PROTOCOL)
            with open(out_model_file, 'wb') as ml:
                pickle.dump(model, ml, protocol=pickle.HIGHEST_PROTOCOL)
            with open(report_file, 'w') as f:
                f.write('\nClassification_report: \n')
                f.write(classification_report(y_val, y_predict, target_names=encoder.classes_))

    avg_accuracy = str(np.sum(accuracies)/len(accuracies))
    avg_f1 = str(np.sum(f1s)/len(f1s))
    with open(report_file, 'a') as f:
        f.write('Average accuracy: ')
        f.write(avg_accuracy)
        f.write('\nAverage f1: ')
        f.write(avg_f1)