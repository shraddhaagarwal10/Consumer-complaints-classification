# -*- coding: utf-8 -*-
"""
This file is used to generate test data class labels.
"""

from project import data, txt_cleaning, preprocessing

### for testing set
tst_data = pd.read_csv('test_data.csv', header = None)
tst_data.rename(columns = {0:"Complaints"}, inplace = True)

# Pre-processing test data
tst_cln = txt_cleaning(tst_data.Complaints)
tst_cln, tokens = preprocessing(tst_cln)
    
count_vec = CountVectorizer(tokenizer = word_tokenize, token_pattern=None, ngram_range = (1,3))
trn=count_vec.fit_transform(data.Complaints)
tst=count_vec.transform(tst_cln)

# Predicting text class labels
lr = LogisticRegression(random_state=0,max_iter=200)
lr.fit(trn, data.Products)
preds = lr.predict(tst)

predictions = preds.reshape(4061, 1)

# Saving test class labels
text_file = open("testing_data_labelfile.txt", "w")
for i in range(len(predictions)):
    text_file.write(str(*predictions[i]) + "\n")
text_file.close()