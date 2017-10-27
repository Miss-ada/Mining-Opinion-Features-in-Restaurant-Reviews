*******************This file including***************************
-Python Scripts:	train_model-P3.py
			restaurant-word-features-P3.py

-Review Data: 		restaurant-training.data
			restaurant-testing.data
			restaurant-development.data

-Evaluation:
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
