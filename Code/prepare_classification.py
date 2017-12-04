import json, re
import pandas as pd

"""
prepare file for classification
"""

input_file = '../Data/candidates_attributes_subjective.json'
output_file = '../Data/labelled_data/attributes_subjective.csv'
f = open(input_file)
candidates_attributes = json.load(f)
attr = []
for tuple in candidates_attributes:
    attr.append(tuple[0])
df = pd.DataFrame(attr)
df.to_csv(output_file, header=['attribute'], index=False)

# input_file = '/Users/Daisy/Desktop/Fall2017/TIM245/yelp-dataset-challenge-master/data/labeled_attributes/develop_set'
# output_file = '../Data/labelled_data/train.csv'
# attributes = []
# labels = []
# with open(input_file, 'r') as f:
#     for line in f:
#         line_seperate = re.search('(^\w)(\w+)', line)
#         label = line_seperate.group(1)
#         labels.append(label)
#         attr = line_seperate.group(2)
#         attributes.append(attr)
# df = pd.DataFrame({'attribute': attributes, 'label': labels})
# df.to_csv(output_file, index=False)