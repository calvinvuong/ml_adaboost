from mldata import *
import numpy as np
from calculations import input_weight_activation
from preprocessing import *
import random
import sys

import matplotlib.pyplot as plt

import input_arg_processing as iap
import common as cmn

CLASS_ATTR = -1

# Generalization function to run the logreg classification on an example and given the parametrs (learned)
def gen_classify(example, parameters):
    classification = classify(parameters, example)
    return 1 if classification > .5 else -1

# Generalization function to learn using logreg on a dataset, with given example weights and number of attributes.
def gen_learn(data_set, example_weights, num_attributes):
    step_size = 0.05
    debug_val = False
    lambd = .1
    
    return minimize_log_likelihood(
                    data_set,
                    example_weights,
                    step_size, 
                    lambd, 
                    num_attributes, 
                    debug=debug_val
                )

# Classifies an example given learned weights
# weights: The weight vector that was learned in training (with bias b at front)
# example: The specific example that we want to classify.
# return: p(y=1 | x) or in other words the confidence of this example given weights
def classify(weights, example):
    return input_weight_activation(weights, example)


# Returns an array of weights, with first value being the bias
# Set weights to random float between -1 and 1
def initialize_weights(num_attributes):
    weights = [random.random() * 2 - 1 for i in range(num_attributes)]
    return np.asarray(weights)

# Performs the standard gradient descent algorithm to minimize
# the negative log likelihood; iterates until stop
# lambd is the weight penalty hyperparameter
def minimize_log_likelihood(examples, ex_weights, step_size, lambd, num_attributes, debug=False):
    M = 100 # num iterations
    weights = initialize_weights(num_attributes)
    
    while M > 0:
        weights = gradient_descend(weights, ex_weights, examples, step_size, lambd)
        if debug and M % 10 == 0:
            print(evaluate_model(examples, weights)[0])
        M -= 1

    # print(weights)
    return weights

# Performs one iteration of gradient descent
# lambd: weight penalty hyperparameter
# Adjusts the weights by step size * gradient and returns new weights
def gradient_descend(weights, ex_weights, examples, step_size, lambd):
    # Compute the gradient
    #gradient = weights.copy() # initialize to existing weights
    gradient = weights * lambd  # initialize to existing weights times weight penalty

    for i, ex in enumerate(examples):
        # calculated predicted class
        predicted = input_weight_activation(weights, ex)
        true_label = 1 if ex[CLASS_ATTR] else 0
        for j in range(len(weights)):
            x_j = 1 if j == 0 else ex[j]
            gradient[j] += ex_weights[i] * partial_log_likelihood(x_j, predicted, true_label)

    # update weights with gradient
    weights = weights - step_size * gradient

    return weights

# Computes and returns the contribution of one example
# to the partial derivative of negative log likelihood
# w.r.t to weight j
def partial_log_likelihood(x_j, predicted, true):
    return (predicted - true) * x_j

# Returns the accuracy, precision, recall of the model evaluated on the example_set
# Also returns num of pos and num of neg examples
def evaluate_model(example_set, weights, debug=False):
    TP, TN, FP, FN, confidence_list = example_confidences(example_set, weights)
    accuracy = (TP + TN) / len(example_set)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    all_pos = TP + FN
    all_neg = TN + FP

    return accuracy, precision, recall, all_pos, all_neg, confidence_list

# calculates the confidences of each classification
# example_set: The set of examples we would like to classify
# weights: the learned weights that we will use for classification.
# return: A list of the confidences for each example with it's true label.
def example_confidences(example_set, weights):
    TP, TN, FP, FN = 0, 0, 0, 0
    THRESHOLD = .5
    CLASS_ATTR = -1

    confidence_list = []
    for example in example_set:
        confidence = classify(weights, example)

        predicted_label = confidence >= THRESHOLD
        true_label = example[CLASS_ATTR]

        confidence_list.append((confidence, true_label))

        TP, TN, FP, FN = cmn.increment_counts(true_label, predicted_label, TP, TN, FP, FN)

    return TP, TN, FP, FN, confidence_list

# Train Naive bayes model on whole example set
# Evaluate on whole example set, return accuracy
def train_evaluate_whole(dataset, step_size, lambd, debug=False, correlate=False, threshold=0, printout=True):
    schema = dataset.schema
    num_attributes = len(dataset[0]) -1

    if correlate:
        correlation(dataset, threshold)

    # learning
    example_weights = [1/len(dataset)] * len(dataset)
    weights = minimize_log_likelihood(dataset, example_weights, step_size, lambd, num_attributes)

    # evaluation
    accuracy, precision, recall, all_pos, all_neg, confidence_list = evaluate_model(dataset, weights, debug=debug)

    # print out mean, precision, recall, AUC
    return cmn.summarize_performance_measures(
                    [accuracy], 
                    [precision], 
                    [recall], 
                    all_pos, 
                    all_neg, 
                    confidence_list, 
                    debug=debug, 
                    printout = printout
                )

# Perform stratified k-fold cross validation to train and evaluate tree
# Return average accuracy over k-folds
def k_fold_cross_validation(
            dataset, 
            k, 
            step_size, 
            lambd, 
            debug=False, 
            correlate=False, 
            threshold=0,
            printout=True
        ):
    schema = dataset.schema    
    num_attributes = len(dataset[0]) -1

    if correlate:
        correlation(dataset, threshold)
    
    # Generate folds
    folds = cmn.stratified_folds(dataset, k)

    accuracy_list = []
    precision_list = []
    recall_list = []
    confidences = []
    all_pos, all_neg = 0, 0
    
    for i, holdout in enumerate(folds):
        # create the training set:
        training_set = ExampleSet([ex for fold in folds if fold != holdout for ex in fold])
        
        # Train on the data
        example_weights = [1/len(training_set)] * len(training_set)
        weights = minimize_log_likelihood(
                        training_set,
                        example_weights,
                        step_size, 
                        lambd, 
                        num_attributes, 
                        debug=debug
                    )
        # validate on holdout
        accuracy, precision, recall, pos, neg, confidence_list = evaluate_model(holdout, weights, debug=debug)

        # Retain accuracies etc:
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)

        confidences += confidence_list
        all_pos += pos
        all_neg += neg

    return cmn.summarize_performance_measures(
                    accuracy_list, 
                    precision_list, 
                    recall_list, 
                    all_pos, 
                    all_neg, 
                    confidences, 
                    debug=debug, 
                    printout=printout
                )

# Function which allows logreg to be run in notebooks. Runs the actual algorithm given arguments.
def run(args, print_flag=True):
    # Define Necessary parameters
    STEP_SIZE = 0.05 # used to be 0.005
    NUM_FOLDS = 5

    RNG_SEED = 12345
    random.seed(RNG_SEED)
    cmn.random.seed(RNG_SEED)

    # Set the thresholds to compute for correlation.
    thres = .75

    LAMBDA = args.lambd
    example_set = parse_c45(args.name, args.path)
    CORRELATE = args.correlate
    
    # Standardize and numerification of the examples (so it can be run through logreg)
    numerify_nominal(example_set)
    standardize_input(example_set)

    # Determine how the model should be trained.
    if args.full_sample:
        return train_evaluate_whole(
                        example_set, 
                        STEP_SIZE, 
                        LAMBDA, 
                        debug=False, 
                        correlate=CORRELATE, 
                        threshold=thres,
                        printout=print_flag
                    )
    else:
        return k_fold_cross_validation(
                        example_set, 
                        NUM_FOLDS, 
                        STEP_SIZE, 
                        LAMBDA, 
                        debug=False, 
                        correlate=CORRELATE, 
                        threshold=thres,
                        printout = print_flag
                    )

# Research extension
def correlation(dataset, threshold):
    ex_matrix = []
    for ex in dataset:
        ex_matrix.append(ex[1:-1])
    # Get correlation coefficients between different features
    corr_matrix = np.corrcoef(ex_matrix, rowvar=False)
    
    # zero diagonal
    np.fill_diagonal(corr_matrix, 0.0)
    corr_matrix = np.absolute(corr_matrix)

    # remove one feature for every pair with correlation > threshold
    removed_attributes = []
    max_coeff_index = np.unravel_index(corr_matrix.argmax(), corr_matrix.shape)
    while corr_matrix[max_coeff_index] > threshold:
        removed_attributes.append(max_coeff_index[0]+1)
        corr_matrix[max_coeff_index[0],:] = 0.0
        corr_matrix[:,max_coeff_index[0]] = 0.0

        max_coeff_index = np.unravel_index(corr_matrix.argmax(), corr_matrix.shape)


    # remove attributes from the example vectors
    removed_attributes = sorted(removed_attributes, reverse=True)
    for ex in dataset:
        for index in removed_attributes:
            del ex[index]
    
if __name__ == "__main__":
    run(iap.parser_logreg_args().parse_args())
