# -*- coding: utf-8 -*-

import numpy as np                                 # linear algebra
import pandas as pd                                # data processing
import matplotlib.pyplot as plt                    # visulaization
import re
import string
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # bags of words and TF IDF
from sklearn.metrics import classification_report, confusion_matrix          # classification Metrics
from sklearn.model_selection import GridSearchCV                             # for tuning parameters                          
from sklearn.model_selection import train_test_split                         # splitting dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import itertools

# Loading training data
data = pd.read_csv('training_data.csv', header = None)
data.rename(columns = {0:"Complaints", 1:"Products"}, inplace = True)

# Fixing null complaints
data = data.dropna()
data = data.reset_index(drop="True")
comp = pd.DataFrame(data.groupby(["Products"]).count())
print(comp)

# Cleaning the data
def txt_cleaning(text):
    text = [re.sub(r'@\S+', '', t) for t in text ]
    text = [re.sub(r'#', '', t) for t in text ]
    text = [re.sub(r"https?\S+", '', t) for t in text ] 
    text = [t.lower() for t in text]
    text = [ re.sub(r"\d*", '', t) for t in text ]    
    text = [ re.sub(r"[+|-|*|%]", '', t) for t in text ]  
    text = [ re.sub(r"[^^(éèêùçà)\x20-\x7E]", '', t) for t in text]
    return text
train_cln = txt_cleaning(data.Complaints)

# Pre-processing the data
def preprocessing(cleaned_txt):
    stop_words = set(stopwords.words('english'))
    lemmatizer    = WordNetLemmatizer() 
    txt2words     = [word_tokenize(sent)  for sent in cleaned_txt ]
    stopwords_rmv = [[t for t in l if t not in stop_words ] for l in txt2words ]
    lemmatize_verbs = [[lemmatizer.lemmatize(t, pos ="v") for t in l if t not in stop_words ] for l in stopwords_rmv ]
    lemmatize_nouns = [[lemmatizer.lemmatize(t, pos ="n") for t in l if t not in stop_words ] for l in lemmatize_verbs ]
    text = [[t.translate(str.maketrans('', '', string.punctuation)) for t in l if len(t)>1] for l in lemmatize_nouns ]
    stopwords_rmv = [" ".join([(t) for t in l if t not in stop_words and len(t)>0]) for l in text ]
    return stopwords_rmv, txt2words
train_cln, txt2words = preprocessing(train_cln)

for i in range(len(data)):
    data["Complaints"][i] = train_cln[i]
    
# Visualizaton of count of complaints in different products
last_token = list(itertools.chain(*txt2words))  
bow_simple = Counter(last_token).most_common(15)
labels = [item[0] for item in bow_simple]
number = [item[1] for item in bow_simple]
nbars = len(bow_simple)

plt.figure(figsize=(15,5))
plt.bar(np.arange(nbars), number, tick_label=labels,color ='maroon',zorder=3)
plt.xticks(rotation=45,fontsize=16)
plt.yticks(fontsize=16)
plt.show()

plt.figure(figsize=(8,8))
plt.pie(data.Products.value_counts()/len(data),
        labels =data['Products'].unique(),
        autopct='%1.2f%%',shadow=True,explode=(0.1, 0.1, 0.1,0.1,0.1))

plt.show()

# Spliting the data into training and cross validation
X_train, X_test, Y_train, Y_test = train_test_split(data["Complaints"], data["Products"], test_size = 0.20, random_state = 42,stratify = data["Products"], shuffle = True)

from collections import Counter
Y_train = [str(x) for x in Y_train]
print('\n Training Data Class  Names:\t['+','.join(list(Counter(Y_train).keys()))+']\n')
clas_labels = [str(x) for  x in list(Counter(Y_train).values())]
print(' Instances in Individual Classes: '+','.join(clas_labels))
Y_test = [str(x) for x in Y_test]
print('\n Test Data Class  Names:\t['+','.join(list(Counter(Y_test).keys()))+']\n')
clas_labels = [str(x) for  x in list(Counter(Y_test).values())]
print(' Instances in Individual Classes: '+','.join(clas_labels))

def feature_selection(feat_sel, X_train, X_test):
    if feat_sel == 'bow':
        count_vec = CountVectorizer(tokenizer = word_tokenize, token_pattern=None)
    if feat_sel == 'tf-idf':
        count_vec = TfidfVectorizer(tokenizer = word_tokenize, token_pattern=None)
    if feat_sel == 'ngram13':
        count_vec = CountVectorizer(tokenizer = word_tokenize, token_pattern=None, ngram_range = (1,3))
    if feat_sel == 'ngram12':
        count_vec = CountVectorizer(tokenizer = word_tokenize, token_pattern=None, ngram_range = (1,2))
    if feat_sel == 'ngram23':
        count_vec = CountVectorizer(tokenizer = word_tokenize, token_pattern=None, ngram_range = (2,3))
 
    count_vec.fit(X_train)
    xtrain = count_vec.transform(X_train)
    xtest = count_vec.transform(X_test)
    return xtrain, xtest

tf_train, tf_test = feature_selection('tf-idf', X_train, X_test)
bow_train, bow_test = feature_selection('bow', X_train, X_test)
ngram_train13, ngram_test13 = feature_selection('ngram13', X_train, X_test)
ngram_train12, ngram_test12 = feature_selection('ngram12', X_train, X_test)
ngram_train23, ngram_test23 = feature_selection('ngram23', X_train, X_test)

class data_classification():
    def __init__(self, clf_opt='lr'):
        self.clf_opt = clf_opt
     #   self.feat_sel = feat_sel

    # Selection of classifiers  
    def classification_pipeline(self, clf_opt='lr'):       
    # Decision Tree
        if self.clf_opt=='dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=40) 
            clf_parameters = {
                'criterion':('gini', 'entropy'), 
                'max_features':('sqrt', 'log2'),
                'max_depth':(10,25,50),
                'ccp_alpha':(0.001,0.005,0.01)
            } 
    # Logistic Regression 
        elif self.clf_opt=='lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
            clf_parameters = {
                'random_state':(0,10),
                 'max_iter':(100,200),
            }  
    # Multinomial Naive Bayes
        elif self.clf_opt=='nb':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            clf = MultinomialNB(fit_prior=True, class_prior=None)  
            clf_parameters = {
                'alpha':(0.001,0.01,0.1,0.5),
            }            
    # Random Forest 
        elif self.clf_opt=='rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None,class_weight='balanced')
            clf_parameters = {
                'criterion':('entropy','gini'),       
                'n_estimators':(30,50),
                'max_depth':(20, 40),
                'max_features':('sqrt', 'log2')
            }
    # k-Nearest Neighbors
        elif self.clf_opt=='knn':
            print('\n\t### Training k-nearest Neighbor Classifier ### \n')
            clf = KNeighborsClassifier()  
            clf_parameters = {
                'n_neighbors':(5,15,50),
                'weights':('uniform', 'distance'),
                'algorithm':('auto', 'kd_tree'),
                'leaf_size':(10,30,80)
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)           
        return clf, clf_parameters

    
    def classification(self,fs,Y_train,Y_test):
        if fs == 'tf-idf':
            X_train = tf_train
            X_test = tf_test
        if fs == 'bow':
            X_train = bow_train
            X_test = bow_test
        if fs == 'ngram12':
            X_train = ngram_train12
            X_test = ngram_test12
        if fs == 'ngram13':
            X_train = ngram_train13
            X_test = ngram_test13
        if fs == 'ngram23':
            X_train = ngram_train23
            X_test = ngram_test23
            
        clf,clf_parameters = self.classification_pipeline()
        grid = GridSearchCV(clf,clf_parameters,scoring='accuracy',cv=5)          
        grid.fit(X_train, Y_train)     
        clf= grid.best_estimator_
        predicted = clf.predict(X_test)
        print(grid.best_params_) 
                
    # Evaluation
        class_names=list(Counter(Y_test).keys())
        class_names = [str(x) for x in class_names] 
        print('\n The classes are: ')
        print(class_names) 

        print('\n *************** Confusion Matrix ***************  \n')
        print (confusion_matrix(Y_test, predicted))     
        print('\n ***************  Scores on Test Data  *************** \n ')
        print(classification_report(Y_test, predicted, target_names=class_names))

#models=['nb','dt','lr','knn','rf']
#feat_sel=['tf-idf','bow','ngram12','ngram13','ngram23']
        
#for i in range(len(models)):
#    for j in range(len(feat_sel)):
#        print("\nFeature Selection : "+ feat_sel[j])
#        start = time.time()
#        clf = data_classification(clf_opt=models[i])
#        clf.classification(feat_sel[j], Y_train, Y_test)
#        end = time.time()
#        print("The time of execution of above program is :", (end-start), "sec")
        
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