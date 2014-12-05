import sys

from random import random
from math import log

class NaiveBayes(object):
    def __init__(self, training, max_idx):
        self.train_data = training
        self.neg_probs = {}
        self.pos_probs = {}
        self.neg = 0
        self.pos = 0
        self.total = 0
        self.max_idx = max_idx

        self._train()

    def _train(self):
        neg = 0
        pos = 0

        # count negative and positive classes in training data 
        # update attribute count
        for data in self.train_data:
            for attribute in data[1:]:
                if data[0] == "-1":
                    if attribute not in self.neg_probs:
                        self.neg_probs[attribute] = 2
                    else:
                        self.neg_probs[attribute] += 1
                else:
                    if attribute not in self.pos_probs:
                        self.pos_probs[attribute] = 2
                    else:
                        self.pos_probs[attribute] += 1

            if data[0] == "-1":
                neg += 1
            else:
                pos += 1

        # lapacian correction, sets likelihood probablities to a minimum
        # of 1 / class count for each attribute 
        # need to add max_idx (number of attributes) to account for correction
        self.neg = (float(neg) + self.max_idx)
        self.pos = (float(pos) + self.max_idx)
        self.total = self.neg + self.pos

        # calculate likelihood probabilities
        self.neg_probs = dict(map(lambda(k, v): (k, v / self.neg), self.neg_probs.iteritems()))
        self.pos_probs = dict(map(lambda(k, v): (k, v / self.pos), self.pos_probs.iteritems()))

    def classify_train(self, train_data):
        data = train_data[1:]
        
        comb_likelihood_neg = 1
        comb_likelihood_pos = 1

        # calculate likelihood probabilities for set of attributes
        # if attribute does not exist in likelihood probability list,
        # use default 1 / total class count to eliminate zero-probability
        for value in data:
            if value in self.neg_probs:
                comb_likelihood_neg *= self.neg_probs[value]
            else:
                comb_likelihood_neg *= (1 / self.neg) 

            if value in self.pos_probs:
                comb_likelihood_pos *= self.pos_probs[value]
            else:
                comb_likelihood_pos *= (1/ self.pos) 
       
        # calculate posteriori probabilities
        posteriori_neg = comb_likelihood_neg * (self.neg / self.total)
        posteriori_pos = comb_likelihood_pos * (self.pos / self.total)

        # return 0 if correctly classed, 1 if misclassed
        if (posteriori_neg > posteriori_pos):
            if train_data[0] == "-1":
                return 0 

        else:
            if train_data[0] == "+1":
                return 0 

        return 1

    # same as above, return predicted class only
    def classify_test(self, test_data):
        data = test_data[1:]
        
        comb_likelihood_neg = 1
        comb_likelihood_pos = 1

        for value in data:
            if value in self.neg_probs:
                comb_likelihood_neg *= self.neg_probs[value]
            else:
                comb_likelihood_neg *= (1 / self.neg) 

            if value in self.pos_probs:
                comb_likelihood_pos *= self.pos_probs[value]
            else:
                comb_likelihood_pos *= (1/ self.pos) 
        
        posteriori_neg = comb_likelihood_neg * (self.neg / self.total)
        posteriori_pos = comb_likelihood_pos * (self.pos / self.total)

        if (posteriori_neg > posteriori_pos):
            return -1
        else:
            return 1

class NBAdaBoost(object):
    def __init__(self, training, test):
        self.train_file = training
        self.test_file = test
        self.neg_probs = {}
        self.pos_probs = {}
        self.true_total = 0
        self.max_idx = 0
        self.train_data = []
        self.test_data = []
        self.classifiers = []
        self.weights = []
        self.classifier_weights = []

        self._read_files()
        # generate specified number of weighted classifiers
        self._generate_classifiers(7)

    def _read_files(self):
        with open(self.train_file, 'r') as data_file:
            neg = 0
            pos = 0
            max_idx = 0

            for line in data_file.readlines():
                space_split = line.rstrip().split(' ')
                train_class = space_split[0]

                del space_split[0]

                member = [train_class]
                for attribute in space_split:
                    idx, val = map(int, attribute.split(':'))

                    if idx > max_idx:
                        max_idx = idx

                    member.append((idx, val))

                # count number of positive and negative members in data
                if train_class == "-1":
                    neg += 1
                else:
                    pos += 1

                self.train_data.append(tuple(member))

            self.max_idx = max_idx
            self.true_total = (neg + pos)

        with open(self.test_file, 'r') as data_file:
            for line in data_file.readlines():
                data = []
                space_split = line.rstrip().split(' ')
                data.append(space_split[0])

                del space_split[0]
                for attribute in space_split:
                    idx, val = map(int,attribute.split(':'))
                    data.append((idx, val))

                self.test_data.append(data)

    # set all weights to 1 / total size 
    def _init_weights(self, size):
        for idx in xrange(size):
            self.weights.append(float(1) / size)

    # normalize weights so they sum to 1
    def _normalize_weights(self):
        total = 0

        for weight in self.weights:
            total += weight

        self.weights = map(lambda(x): x / total, self.weights)

    # retrieve sample set of data based on weights and size
    def _get_sample(self, size):
        sample = []

        for idx in xrange(size):
            count = 0
            idx = 0
            rand = random()
        
            while rand > count:
                count += self.weights[idx]
                idx += 1

            sample.append(self.train_data[idx - 1])

        return sample

    # generates k classifiers through k iterations of the adaboost algorithm
    # updating training member weights and classifier weights
    def _generate_classifiers(self, k):
        self._init_weights(self.true_total)

        for idx in xrange(k):
            sample = self._get_sample(self.true_total)
            self.classifiers.append(NaiveBayes(sample, self.max_idx))
            err_rate = 0
            misclassified = [] 

            # intialize classifer and train with sample data, then calculate
            # error rate with original training data set
            for t_idx, member in enumerate(self.train_data):
                class_val = self.classifiers[idx].classify_train(member)
                err_rate += class_val * self.weights[t_idx]

                if class_val:
                    misclassified.append(True)
                else:
                    misclassified.append(False)

            # reduce weights for correctly classified members, 
            # paying more attention to incorrectly classified members
            for t_idx, value in enumerate(misclassified):
                if not value:
                    self.weights[t_idx] = self.weights[t_idx] * (err_rate / (1 - err_rate))

            # update classifier weights
            if err_rate:
                if err_rate > 0.5:
                    self.classifier_weights.append(-1 * log((1 - err_rate) / err_rate))
                else:
                    self.classifier_weights.append(log((1 - err_rate) / err_rate))
            else:
                self.classifier_weights.append(0)

            self._normalize_weights()

    # run training data against classifiers and calculated weighted final vote
    def classify_train(self):
        true_pos = 0
        false_neg = 0
        false_pos = 0
        true_neg = 0

        for member in self.train_data:
            votes = 0

            for idx, classifier in enumerate(self.classifiers):
                votes += classifier.classify_test(member) * self.classifier_weights[idx]

            if votes < 0:
                if member[0] == "-1":
                    true_neg += 1
                else:
                    false_neg += 1
            else:
                if member[0] == "+1":
                    true_pos += 1
                else:
                    false_pos += 1

        return (true_pos, false_neg, false_pos, true_neg)

    # same as above, but with test data
    def classify_test(self):
        true_pos = 0
        false_neg = 0
        false_pos = 0
        true_neg = 0

        for member in self.test_data:
            votes = 0

            for idx, classifier in enumerate(self.classifiers):
                votes += classifier.classify_test(member) * self.classifier_weights[idx]

            if votes < 0:
                if member[0] == "-1":
                    true_neg += 1
                else:
                    false_neg += 1
            else:
                if member[0] == "+1":
                    true_pos += 1
                else:
                    false_pos += 1

        return (true_pos, false_neg, false_pos, true_neg)

            
def main(argv):
    if len(argv) != 2:
        print "Incorrect number of parameters"
        print " Usage: python naive_payes.py <trainingFile> <testFile>"
        sys.exit(-1)

    training = argv[0]
    test = argv[1]

    nbadaboost = NBAdaBoost(training, test)
    training_output = nbadaboost.classify_train()
    test_output = nbadaboost.classify_test()

    print("%i %i %i %i" % (training_output[0], training_output[1], training_output[2], training_output[3]))

    print("%i %i %i %i" % (test_output[0], test_output[1], test_output[2], test_output[3]))

    #print_metrics(training_output)
    #print_metrics(test_output)
   
# calculate and print extra metrics
def print_metrics(data):
    total = float(0)

    for member in data:
        total += member

    accuracy = (data[0] + data[3]) / total
    error = 1 - accuracy
    sensitivity = float(data[0]) / (data[0] + data[1])
    specificity = float(data[3]) / (data[2] + data[3])
    precision = float(data[0]) / (data[0] + data[2])
    recall = sensitivity
    f_score = (2 * precision * recall) / (precision + recall)
    f05 = ((1 + 0.5 * 0.5) * precision * recall) / (0.5 * 0.5 * precision + recall)
    f2 = ((1 + 2 * 2) * precision * recall) / (2 * 2 * precision + recall) 

    print("%f %f %f %f" % (accuracy, error, sensitivity, specificity))
    print("%f %f %f %f" % (precision, f_score, f05, f2))
    
if __name__ == "__main__":
    main(sys.argv[1:])
