from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


file = open("/home/banafsheh/Desktop/h/code/all-c")
file.readline()
data = np.loadtxt(file,delimiter=',')
X_train = data[:, 0:54]
Y_train = data[:, 55]

anova_filter = SelectKBest(f_regression, k=5)

clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X_train, Y_train)

prediction = anova_svm.predict(X_train)
anova_svm.score(X_train, Y_train)#0.97082658022690438
