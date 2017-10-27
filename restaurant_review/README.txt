*******************This file including***************************
-Python Scripts:	train_model-P3.py
			restaurant-word-features-P3.py

-Review Data: 		restaurant-training.data
			restaurant-testing.data
			restaurant-development.data

-Model pickle files:    nb-word_features-absolute-model-P3.pickle
			nb-word_features-binary-model-P3.pickle
			nb-word_features-relative-improved-model-P3.pickle

-Output of features:	features-training-word_features-absolute.txt
			features-training-word_features-binary.txt
			features-training-word_features-relative-improved.txt

-Output of most informative:
			nb-word_features-absolute-most-informative-features.txt
			nb-word_features-binary-most-informative-features.txt
			nb-word_features-relative-improved-most-informative-features.txt
			
-Output of evaluation:  out-dev-word_features-absolute.txt
			out-test-word_features-absolute.txt
			out-dev-word_features-binary.txt
			out-test-word_features-binary.txt
			out-dev-word_features-relative-improved.txt
			out-test-word_features-relative-improved.txt
			all-results.txt

-README

********************Usage*****************************************
train_model-P3.py:
	-used for output model
	-need to include Line 102 for adding word_features with absolute frequency 
	-need to include Line 104 for adding word binary features
	-need to include line 106 for adding word features with relative frequency

restaurant-word-features-P3.py is used for output evaluation
	-can take 3 arguments: model_file classify_review output_file
	-or modify Line 238-240 to save the typing
	-need to include Line 102 for adding word_features with absolute frequency 
	-need to include Line 104 for adding word binary features
	-need to include line 106 for adding word features with relative frequency