Binary-Classification
=====================

The Naïve Bayes classifier used in this classification framework is based on Bayes’ Theorem.

  Laplacian correction is applied to the data to help solve the zero-probability problem by adding 1 to each possible case count and updating the total counts accordingly to maintain the probability estimates.

  In general, the training and test files are read into data structures in the program and the total number of attributes is determined.  A train function is then called to calculate the likelihood probabilities of each case as well as the probabilities for each class.  Finally, both training and test data are classified using a classify function which calculates the posteriori probabilities for each possible class.  Since this assignment only deals with binary classification, there are only two possible classes.  The posteriori probabilities are compared and a classification is made based on this comparison.  The Naïve Bayes classification is then compared with the actual classification and metrics are generated.

  In addition to the Naïve Bayes algorithm, a boosting algorithm is used in conjunction to optimize accuracy.  First, a weight is assigned to each data tuple in the training dataset.  The intial weights are set using the init_weights method, which initializes each weight to be 1/d, where d is the size of the training data set.  Then, a generate_classifiers method is called, which runs k times, creating k instances of the Naïve Bayes classifier.  These classifiers are then weighted, and the test data is run on each classifier.  The sum of the weighted "votes" of the classifiers constitutes the final classification.
