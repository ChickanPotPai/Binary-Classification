import sys

class NaiveBayes(object):
    def __init__(self, training, test):
        self.train_file = training
        self.test_file = test
        self.neg_train_data = []
        self.pos_train_data = []
        self.neg_probs = {}
        self.pos_probs = {}
        self.neg = 0
        self.pos = 0
        self.total = 0
        self.max_idx = 0
        self.test_data = []

        self._train()

    def _train(self):
        self._read_files()
        self._pre_calculate()

    # read files, separate training file by class
    # also counts attributes, sorted by class
    def _read_files(self):
        with open(self.train_file, 'r') as data_file:
            neg = 0
            pos = 0
            max_idx = 0

            for line in data_file.readlines():
                space_split = line.rstrip().split(' ')
                train_class = space_split[0]

                del space_split[0]

                data = []
                for attribute in space_split:
                    idx, val = map(int, attribute.split(':'))

                    if idx > max_idx:
                        max_idx = idx

                    data.append((idx, val))

                    # update attribute count by class
                    # starts at 2 due to lapacian correction
                    if train_class == "-1":
                        if (idx, val) not in self.neg_probs:
                            self.neg_probs[(idx,val)] = 2
                        else:
                            self.neg_probs[(idx, val)] += 1
                    else:
                        if (idx, val) not in self.pos_probs:
                            self.pos_probs[(idx, val)] = 2
                        else:
                            self.pos_probs[(idx, val)] += 1

                # update class count and sort data into arrays by class
                if train_class == "-1":
                    neg += 1
                    self.neg_train_data.append(tuple(data))
                else:
                    pos += 1
                    self.pos_train_data.append(tuple(data))    


            # lapacian correction, sets likelihood probablities to a minimum
            # of 1 / class count for each attribute 
            # need to add max_idx (number of attributes) to account for correction
            self.max_idx = max_idx
            self.neg = (float(neg) + max_idx)
            self.pos = (float(pos) + max_idx)
            self.total = self.neg + self.pos 

        with open(self.test_file, 'r') as data_file:
            for line in data_file.readlines():
                data = []
                space_split = line.rstrip().split(' ')
                data.append(space_split[0])

                del space_split[0]
                for attribute in space_split:
                    idx, val = map(int, attribute.split(':'))
                    data.append((idx, val))

                self.test_data.append(data)

    # calculate all likelihood probabilities based on count per class
    def _pre_calculate(self):
        self.neg_probs = dict(map(lambda(k, v): (k, v / self.neg), self.neg_probs.iteritems()))
        self.pos_probs = dict(map(lambda(k, v): (k, v / self.pos), self.pos_probs.iteritems()))

    def classify_training(self):
        true_pos = 0
        false_neg = 0
        false_pos = 0
        true_neg = 0

        # calculate training data separated by class
        for member in self.neg_train_data:
            comb_likelihood_neg = 1
            comb_likelihood_pos = 1
            
            # calculate likelihood probabilities for set of attributes
            # if attribute does not exist in likelihood probability list,
            # use default 1 / total class count to eliminate zero-probability
            for value in member:
                if value in self.neg_probs:
                    comb_likelihood_neg *= self.neg_probs[value]
                else:
                    comb_likelihood_neg *= (1 / self.neg) 

                if value in self.pos_probs:
                    comb_likelihood_pos *= self.pos_probs[value]
                else:
                    comb_likelihood_pos *= (1 / self.pos)
           
            # calculate posteriori probabilities
            posteriori_neg = comb_likelihood_neg * (self.neg / self.total)
            posteriori_pos = comb_likelihood_pos * (self.pos / self.total)

            # compare and update based on result
            if (posteriori_neg > posteriori_pos):
                true_neg += 1
            else:
                false_pos += 1

        # same as above, but for +1 class data
        for member in self.pos_train_data:
            comb_likelihood_neg = 1
            comb_likelihood_pos = 1

            for value in member:
                if value in self.neg_probs:
                    comb_likelihood_neg *= self.neg_probs[value]
                else:
                    comb_likelihood_neg *= (1 / self.neg)

                if value in self.pos_probs:
                    comb_likelihood_pos *= self.pos_probs[value]
                else:
                    comb_likelihood_pos *= (1 / self.pos)
            
            posteriori_neg = comb_likelihood_neg * (self.neg / self.total)
            posteriori_pos = comb_likelihood_pos * (self.pos / self.total)

            if (posteriori_pos > posteriori_neg):
                true_pos += 1
            else:
                false_neg += 1

        return (true_pos, false_neg, false_pos, true_neg)

    # same as above, but with combined data (unknown class)
    def classify_test(self):
        true_pos = 0
        false_neg = 0
        false_pos = 0
        true_neg = 0

        for member in self.test_data:
            data = member[1:]
            
            comb_likelihood_neg = 1
            comb_likelihood_pos = 1

            for value in data:
                if value in self.neg_probs:
                    comb_likelihood_neg *= self.neg_probs[value]
                else:
                    comb_likelihood_neg *= (1 /self.neg)

                if value in self.pos_probs:
                    comb_likelihood_pos *= self.pos_probs[value]
                else:
                    comb_likelihood_pos *= (1 / self.pos)
            
            posteriori_neg = comb_likelihood_neg * (self.neg / self.total)
            posteriori_pos = comb_likelihood_pos * (self.pos / self.total)

            if (posteriori_neg > posteriori_pos):
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

    naive_bayes = NaiveBayes(training, test)
    training_output = naive_bayes.classify_training()
    test_output = naive_bayes.classify_test()

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
