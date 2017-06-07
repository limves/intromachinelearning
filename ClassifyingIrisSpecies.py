# A First Application: Classifying Iris Species
import numpy as np 

from sklearn.datasets import load_iris 
iris_dataset = load_iris ()

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split (
     iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train )

# Making Predictions
X_new = np.array([[5, 2.9, 1, 0.2]]) 
print("X_new.shape: {}".format(X_new.shape ))

prediction = knn.predict(X_new) 
print("Prediction: {}".format(prediction)) 
print("Predicted target name: {}".format (iris_dataset['target_names'][prediction ]))

# Evaluating the Mode
y_pred = knn.predict(X_test ) # This is where the test set that we created earlier comes in
print("Test set predictions:\n {}".format(y_pred ))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test )))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test )))
