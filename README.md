## General

This program implements a simple one layer perceptron algorithm.                    
It's an Machine Learning algorithm for supervised learning of binary classifiers.

Idea : Given set of vectors of the same dimension, which are binary classified, we want to create an algorithm
which can learn to classify vectors of the dimension.

https://en.wikipedia.org/wiki/Perceptron

!!!WORKS FOR ONLY LINEARLY SEPARABLE DATA!!!
The algorithm won't terminate if the data isn't linearly separable.

https://en.wikipedia.org/wiki/Linear_separability

The user can enter a training set for which the algorithm will return a predictor.    
You can enter / generate a sample solution to calculate an error.
You're also able visualize the result in a special case.

## Definition of the training set
The training set is a list of the type [(x,y) , ... ] where every first element of one tuple is an np.array of 1 dimensional shape with the same size.
The second element of the tuples is either 1 or -1 (the classification).










