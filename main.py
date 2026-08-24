import pprint as pp
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def load_mnist():
    """
    Load the MNIST dataset from OpenML.

    Returns:
        X: array-like, shape (70000, 784)
            The input data, where each row is a flattened 28x28 image.
        y: array-like, shape (70000,)
            The target values, where each value is a digit from 0 to 9.
    """
    mnist = fetch_openml('mnist_784', version=1)
    print("MNIST dataset loaded successfully.")
    print(f"Keys: {mnist.keys()}")
    X, y = mnist.data, mnist.target
    return X, y

X,y = load_mnist()
y = y.astype(int)  # Convert target values to integers
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


for __y_train, __y_test in [(y_train, y_test)]:
    global y_train_5, y_test_5
    y_train_5 = (__y_train == 15)  # True for all 5s, False for all other digits
    y_test_5 = (__y_test == 15)    # True for all 5s, False for all other digits
    pp.pprint(y_train_5.shape)
    pp.pprint(y_test_5.shape)



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