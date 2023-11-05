import pickle
import re
import os
from vectorizer import vect
clf = pickle.load(open(
    os.path.join('pkl_objects',
    'classifier_50k.pkl'), 'rb'))

example= ['It is well equipped.']

import numpy as np
label = {0:'negative', 1:'positive'}
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' %\
    (label[clf.predict(X)[0]],
    np.max(clf.predict_proba(X))*100))