1. extract_restaurant_data.py

- Extract all restaurant data from original yelp dataset

- DOWNLOAD DATASET here: https://www.yelp.com/dataset/challenge

- RESULTING FILE: restaurant.json, restaurant_review.csv (too big, not included)

- REVIEWS COUNT:

* TOTAL RESTAURANTS: 51613

* TOTAL REVIEWS: 2927731

2. preprocesser.py:

- Run standfor tagger and parser on review data

- DOWNLOAD

** TAGGER & PARSE: https://nlp.stanford.edu/software/tagger.shtml#Download
https://nlp.stanford.edu/software/lex-parser.shtml#Download

3. extract_restaurant_data.py:

- Reading in tagger and parser of reviews, generate candidate (attribute, opinion) pair

