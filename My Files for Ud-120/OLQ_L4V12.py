from sklearn import tree
from sklearn.metrics import accuracy_score

tre=tree.DecisionTreeClassifier()
clf=tre.fit(features_train,labels_train)
pred=clf.predict(features_test)

acc_min_samples_split_2=accuracy_score(labels_test,pred)

tre2=tree.DecisionTreeClassifier(min_samples_split=50)
clf=tre2.fit(features_train,labels_train)
pred=clf.predict(features_test)

acc_min_samples_split_50=accuracy_score(labels_test,pred)
