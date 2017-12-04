System pipeline:

1. extract_restaurant_data.py

- Extract all restaurant data from original yelp dataset

- Download dataset here: https://www.yelp.com/dataset/challenge

- Resulting files: restaurant.json, restaurant_review.csv (too big, not included)

- REVIEWS COUNT:

* TOTAL RESTAURANTS: 51613

* TOTAL REVIEWS: 2927731

2. preprocesser.py:

- Run standfor tagger and parser on review data

- Download standford tagger and parser files


3. attribute_extractor.py:

- Reading parsed file and extract attributes

- Resulting file: /Data/candidates_attributes.json

4. prepare_classification.py:

- Extract all attributes from 3 and manually label

- Prepare input file for classification

5. attribute_classification.py:

- Classify attributes into Food, Service, Decor, and Other

- Decision Tree model
