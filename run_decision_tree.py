import pandas
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pandas.read_csv("dataset.csv")

# print(dataset)

target = dataset.iloc[:,30].values
data = dataset.iloc[:,0:30].values

print(target)
print(data)

kfold_object = KFold(n_splits = 4)
kfold_object.get_n_splits(data)


test_case = 0
for training_index, test_index in kfold_object.split(data):
	print(test_case)
	test_case = test_case+1
	print("training: ", training_index)
	print("test: ", test_index)
	data_training = data[training_index]
	data_test = data[test_index]
	target_training = target[training_index]
	target_test = target[test_index]
	machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=3)
	machine.fit(data_training, target_training)
	new_target = machine.predict(data_test)
	print("Accuracy score: ", metrics.accuracy_score(target_test,new_target))
	print("Confusion Matrix: \n", metrics.confusion_matrix(target_test,new_target))



