import pprint as pp
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score

def load_mnist():
    """
    Load the MNIST dataset from OpenML.

    Returns:
        X: array-like, shape (70000, 784)  two dimensional array containing the flattened images of handwritten digits.
            The input data, where each row is a flattened 28x28 image. 
        y: array-like, shape (70000,) one-dimensional array containing the labels for the images. example '5'
            since there are 70000 samples, the target values are also 70000. so len(y) = 70000
            The target values, where each value is a digit from 0 to 9.
    """
    mnist = fetch_openml('mnist_784', version=1)
    print("MNIST dataset loaded successfully.")
    print(f"Keys: {mnist.keys()}")
    X, y = mnist.data, mnist.target
    
    return X, y

X,y = load_mnist()
# Convert target values to integers, any ml model will not accept string values as target values, 
# so we need to convert them to integers.
y = y.astype(int)  
"""
MNIST datasets are split into training and test sets. The first 60,000 samples are used for training, 
and the remaining 10,000 samples are used for testing.

The datasets are already shuffled, so we can simply split them based on the index.
since the datasets are shuffled, all the cross-validation folds will have a good representation of the data.
you don't want some folds to have only 0s and 1s, and some folds to have only 8s and 9s.
the shuffled datasets ensure that each fold has a good representation of the data, 
which is important for training and evaluating machine learning models.
"""

x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
pp.pprint(x_train.shape)
pp.pprint(y_train.shape)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits
y_test_5 = (y_test == 5)    # True for all 5s, False for all other digits
print(f"y_train_5: {y_train_5[:10]}")
print(f"y_test_5: {y_test_5[:10]}")

"""
train a binary classifier to detect the digit 5. The target values are converted to boolean values,
where True indicates that the digit is 5, and False indicates that the digit is not


"""

# use Stocastic Gradient Descent (SGD) classifier to train the model

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)
prdiction_of_5 = sgd_clf.predict([X.iloc[10].to_numpy()])  # predict the first sample in the dataset
print(f"Prediction of the first sample: {prdiction_of_5}")

# use cross-validation to evaluate the model's performance

""" 
cross validation is a technique used to evaluate the performance of a machine learning model 
by splitting the dataset into multiple folds.
In this case, we are using 3-fold cross-validation, which means that the dataset is split into 3 equal parts. 
The model is trained on 2 parts and tested on the remaining part.
"""


score = cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy")

print(f"Cross-validation scores: {score}")
print(f"Mean accuracy: {score.mean()}")


"""
Much better way to evaluate a classifer model. The general idea is to count the number of times
the model classifes a class A instance as class B, and vice versa. 

To compute the confusion matrix, we need to make predictions first, so predictions result can be compared 
with the actual target values.

"""

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3) # cv=3 means 3-fold cross-validation

# now confusion matrix can be calculated by passing target classes and predicted classes to 
# confusion_matrix function

confusionmatrix = confusion_matrix(y_train_5, y_train_pred)

print(f"Confusion matrix:\n{confusionmatrix}")

"""
the result of the confusion matrix looks for y_predict_5
53057 - instances are classifed correctly as non-5 classes -- TN -- True Negatives
1522  - instances are classified wrongly as 5 -- FP -- False Positives
1325  - instances are classified wrongly as non-5 -- FN -- False Negatives
4096  - instances are correctly classifed as 5  -- TP -- True Positives

[
 TN      FP
[53057, 1522],   -- first row negative class 
[1325, 4096 ]    -- second row positive class   
  FN   TP
]

confusion matrix on perfect predicted classes have only values for TP and TN , and 0 value for FP and FN
"""

# from the confusion matrix, we can calculate precision and recall

from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_train_5, y_train_pred)
recall = recall_score(y_train_5, y_train_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")

# F1 score is the harmonic mean of precision and recall, which gives a single metric to evaluate the model's performance.

f1_score = 2 * (precision * recall) / (precision + recall)
print(f"F1 score: {f1_score}")

# or simply use the f1_score function from sklearn.metrics

from sklearn.metrics import f1_score as f1_score_func
f1_score = f1_score_func(y_train_5, y_train_pred)

print(f"F1 score: {f1_score}")


"""

precision and recall are often in tension with each other. 
Increasing precision reduces recall and vice versa.
For example, if you want to increase precision, you can set a higher threshold for classifying 
a sample as positive. This means that the model will be more conservative in predicting positive samples,
which will reduce the number of false positives and increase precision. However, this will also 
increase the number of false negatives, which will reduce recall. 

recall = ( TP / (TP + FN)) if you increase the threshold, you will have more FN, which will reduce recall.


Conversely, if you want to increase recall, you can set a lower threshold for classifying a sample as positive. 
This means that the model will be more liberal in predicting positive samples, 
which will reduce the number of false negatives and increase recall. 
However, this will also increase the number of false positives, which will reduce precision.  


"""







"""
# uncomment the following lines to visualize a sample digit from the dataset

pp.pprint(X.shape)
some_digit = X.iloc[98].to_numpy()
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary", interpolation="nearest")
plt.axis("off")
plt.show()
pp.pprint(y.iloc[98])

"""
