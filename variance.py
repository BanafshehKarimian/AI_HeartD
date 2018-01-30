##neural networks
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
file = open("/home/banafsheh/Desktop/h/code/all-c")
file.readline()
data = np.loadtxt(file,delimiter=',')
X_train = data[0:400, 0:54]
Y_train = data[0:400, 55]
X_test = data[400:, 0:54]
Y_test = data[400:, 55]
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.6 * (1 - .6)))
sel.fit_transform(X_train)
sel.fit_transform(X_test)


#X_train, X_test, y_train, y_test = train_test_split(data[0:800, 0:12].reshape(data.shape[1:]).tranpose(), data[0:800, 13], test_size=0.33, #random_state=42)

#X = data[0:810, 0:12]
#y = data[0:810, 13]

# Add noisy features
#random_state = np.random.RandomState(0)
#n_samples, n_features = X.shape
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

#X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],test_size=.5,random_state=random_state)

nuralnet = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(2, 2), random_state=1)
nuralnet.fit(X_train, Y_train)
accuracy_score(nuralnet.predict(X_test),Y_test)	#0.9662921348314607    0.0092165898617511521    0.92165898617511521??0.17050691244239632
#.................................................................................................#
##Regression+linearSVC+SVC
from sklearn import svm
from sklearn.metrics import accuracy_score
svr_clf = svm.SVR()
svr_clf.fit(X_train, Y_train)
accuracy_score(svr_clf.predict(X_test).round(),Y_test)  #0.4157303370786517   0.004608294930875576   0.0001
#pre.append(precision_score(Y_test, svr_clf.predict(X_test), average='macro'))
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, Y_train)
accuracy_score(lin_clf.predict(X_test).round(),Y_test) #0.5056179775280899    0.66359447004608296   0.94930875576036866
#pre.append(precision_score(Y_test, lin_clf.predict(X_test), average='macro'))
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(X_train, Y_train)
accuracy_score(Y_test,rbf_svc.predict(X_test).round()) #0.4157303370786517   0.0092165898617511521  0.33640552995391704
#pre.append(precision_score(Y_test, rbf_svc.predict(X_test), average='macro'))
sigmoid_svc = svm.SVC(kernel='sigmoid')
sigmoid_svc.fit(X_train, Y_train)
accuracy_score(Y_test,sigmoid_svc.predict(X_test).round()) #0.5617977528089888   0.027649769585253458  0.16589861751152074
#pre.append(precision_score(Y_test, sigmoid_svc.predict(X_test), average='macro'))
#.................................................................................................#
## Nearest Centroid Classifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
ncc_clf = NearestCentroid()
ncc_clf.fit(X_train, Y_train)
accuracy_score(ncc_clf.predict(X_test),Y_test) #0.5842696629213483    0.72811059907834097   0.3686635944700461
#.................................................................................................#
##Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
accuracy_score(gnb.fit(X_train, Y_train).predict(X_test),Y_test) #0.5955056179775281   0.26728110599078342  0.25345622119815669
#.................................................................................................#
##DecisionTreeClassifier
from sklearn import tree
DT_clf = tree.DecisionTreeClassifier()
DT_clf = DT_clf.fit(X_train, Y_train)
accuracy_score(DT_clf.predict(X_test),Y_test) #0.6292134831460674   0.45161290322580644   0.44239631336405533
#pre.append(precision_score(Y_test, DT_clf.predict(X_test), average='macro'))
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("Heart D-tree") 
DTR_clf = tree.DecisionTreeRegressor()
DTR_clf = DTR_clf.fit(X_train, Y_train)
accuracy_score(DTR_clf.predict(X_test),Y_test) #0.6292134831460674    0.86635944700460832  0.45161290322580644
#pre.append(precision_score(Y_test, DTR_clf.predict(X_test), average='macro'))

#.................................................................................................#

