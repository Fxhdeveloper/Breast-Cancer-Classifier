from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#load dataset
breast_cancer_data = load_breast_cancer()
#split train and validation data
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

#classifier initialization
classifier = KNeighborsClassifier(n_neighbors = 3)

#Train classifier
classifier.fit(training_data, training_labels)
#Check accuracy using the validation set
print(classifier.score(validation_data,validation_labels ))

#checking for a better k coefficient
for k in range(1,101):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data, training_labels)
    print(classifier.score(validation_data,validation_labels ))