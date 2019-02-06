from sklearn.datasets         import load_breast_cancer
from sklearn.model_selection  import train_test_split
from sklearn.neighbors        import KNeighborsClassifier
from matplotlib               import pyplot as plt 

#load dataset
breast_cancer_data = load_breast_cancer()
#split train and validation data
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

"""Running the classifier"""
#Training the classifier and finding the best k coefficient
accuracies = []
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data,validation_labels ))
k_list = range(1,101)

#plot accuracies for different k coefficients
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()