##neural networks
>>> from sklearn.neural_network import MLPClassifier
>>> import numpy as np
>>> from sklearn.metrics import accuracy_score
>>> file = open("/home/banafshbts/Desktop/hosh/76/all")
>>> file.readline()
>>> data = np.loadtxt(file,delimiter=',')
>>> data = np.loadtxt(file,delimiter=',')
>>> X_train = data[0:810, 0:12]
>>> Y_train = data[0:810, 13]
>>> X_test = data[810:, 0:12]
>>> Y_test = data[810:, 13]
>>> clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(2, 2), random_state=1)
>>> clf.fit(X_train, Y_train)
>>> accuracy_score(clf.predict(X_test),Y_test)	#0.9662921348314607
#.................................................................................................#
##Regression+linearSVC+SVC
>>> from sklearn import svm
>>> import numpy as np
>>> from sklearn.metrics import accuracy_score
>>> file = open("/home/banafshbts/Desktop/hosh/76/all")
>>> file.readline()
>>> data = np.loadtxt(file,delimiter=',')
>>> data = np.loadtxt(file,delimiter=',')
>>> X_train = data[0:810, 0:12]
>>> Y_train = data[0:810, 13]
>>> X_test = data[810:, 0:12]
>>> Y_test = data[810:, 13]
>>> clf = svm.SVR()
>>> clf.fit(X_train, Y_train)
>>> accuracy_score(clf.predict(X_test).round(),Y_test)  #0.4157303370786517
>>> lin_clf = svm.LinearSVC()
>>> lin_clf.fit(X_train, Y_train)
>>> accuracy_score(lin_clf.predict(X_test).round(),Y_test) #0.5056179775280899
>>> rbf_svc = svm.SVC(kernel='rbf')
>>> rbf_svc.fit(X_train, Y_train)
>>> accuracy_score(Y_test,rbf_svc.predict(X_test).round()) #0.4157303370786517
>>> sigmoid_svc = svm.SVC(kernel='sigmoid')
>>> sigmoid_svc.fit(X_train, Y_train)
>>> accuracy_score(Y_test,sigmoid_svc.predict(X_test).round()) #0.5617977528089888
#.................................................................................................#
## Nearest Centroid Classifier
>>> from sklearn.neighbors.nearest_centroid import NearestCentroid
>>> import numpy as np
>>> file = open("/home/banafshbts/Desktop/hosh/76/all")
>>> file.readline()
>>> data = np.loadtxt(file,delimiter=',')
>>> data = np.loadtxt(file,delimiter=',')
>>> X_train = data[0:810, 0:12]
>>> Y_train = data[0:810, 13]
>>> X_test = data[810:, 0:12]
>>> Y_test = data[810:, 13]
>>> clf = NearestCentroid()
>>> clf.fit(X_train, Y_train)
>>> accuracy_score(clf.predict(X_test),Y_test) #0.5842696629213483
#.................................................................................................#
##Gaussian Naive Bayes
>>> from sklearn.naive_bayes import GaussianNB
>>> import numpy as np
>>> file = open("/home/banafshbts/Desktop/hosh/76/all")
>>> file.readline()
>>> data = np.loadtxt(file,delimiter=',')
>>> data = np.loadtxt(file,delimiter=',')
>>> X_train = data[0:810, 0:12]
>>> Y_train = data[0:810, 13]
>>> X_test = data[810:, 0:12]
>>> Y_test = data[810:, 13]
>>> gnb = GaussianNB()
>>> accuracy_score(gnb.fit(X_train, Y_train).predict(X_test),Y_test) #0.5955056179775281
#.................................................................................................#
##DecisionTreeClassifier
>>> from sklearn import tree
>>> import numpy as np
>>> file = open("/home/banafshbts/Desktop/hosh/76/all")
>>> file.readline()
>>> data = np.loadtxt(file,delimiter=',')
>>> data = np.loadtxt(file,delimiter=',')
>>> X_train = data[0:810, 0:12]
>>> Y_train = data[0:810, 13]
>>> X_test = data[810:, 0:12]
>>> Y_test = data[810:, 13]
>>> clf = tree.DecisionTreeClassifier()
>>> clf = clf.fit(X_train, Y_train)
>>> accuracy_score(clf.predict(X_test),Y_test) #0.6292134831460674
>>> dot_data = tree.export_graphviz(clf, out_file=None) 
>>> graph = graphviz.Source(dot_data) 
>>> graph.render("Heart D-tree") 
#.................................................................................................#
##regretion
>>> from sklearn import tree
>>> import numpy as np
>>> file = open("/home/banafshbts/Desktop/hosh/76/all")
>>> file.readline()
>>> data = np.loadtxt(file,delimiter=',')
>>> data = np.loadtxt(file,delimiter=',')
>>> X_train = data[0:810, 0:12]
>>> Y_train = data[0:810, 13]
>>> X_test = data[810:, 0:12]
>>> Y_test = data[810:, 13]
>>> clf = tree.DecisionTreeRegressor()
>>> clf = clf.fit(X_train, Y_train)
>>> accuracy_score(clf.predict(X_test),Y_test) #0.6292134831460674
