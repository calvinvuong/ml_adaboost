from mldata import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# input from our other python files for shared functionality
import input_arg_processing as iap
from preprocessing import *
from calculations import sigmoid
import common as cmn

# Define macro for class attribute in the table
CLASS_ATTR = -1

# Train Naive bayes model on whole example set
# Evaluate on whole example set, return accuracy
def train_whole_dataset(dataset, M, printout=True):
    # Information about the dataset:
    schema = dataset.schema
    num_attributes = len(schema)-1
    
    # Define each example weight
    example_weights = [1/len(dataset)] * len(dataset)

    # Learn.
    p_pos, p_neg, parameters = MLE_learn(
        dataset, 
        example_weights,
        init_params(schema, num_attributes), 
        num_attributes, 
        M
    )
    # Evaluate.
    accuracy, precision, recall, pos, neg, confidence_list = evaluate_model(
        dataset,
        p_pos, 
        p_neg, 
        parameters, 
        num_attributes
    )
    
    # Get summary information
    return summarize_performance_measures(
        [accuracy], 
        [precision], 
        [recall], 
        pos,
        neg,
        confidence_list,
        printout = printout
    )

# Perform stratified k-fold cross validation to train and evaluate tree
# Return average accuracy over k-folds
def train_k_fold_dataset(dataset, k, M, printout=True):
    # Information about the dataset:
    schema = dataset.schema
    num_attributes = len(schema)-1
    
    # Generate folds
    folds = cmn.stratified_folds(dataset, k)

    # Persistant accuracy.
    accuracy_list = []
    precision_list = []
    recall_list = []
    confidences = []

    all_pos, all_neg = 0, 0
    
    # Iterate through each fold partitioning
    for i, holdout in enumerate(folds):
        # Construct training set of all except holdout
        training_set = ExampleSet([ex for fold in folds if fold != holdout for ex in fold])
        example_weights = [1/len(training_set)] * len(training_set)

        # Evaluate the training set on the validation set.
        schema = training_set.schema
        num_attributes = len(schema)-1

        #Learning
        p_pos, p_neg, parameters = MLE_learn(
            training_set,
            example_weights,
            init_params(schema, num_attributes), 
            num_attributes, 
            M
        )

        #Classification
        accuracy, precision, recall, pos, neg, confidence_list = evaluate_model(
            holdout, 
            p_pos, 
            p_neg, 
            parameters, 
            num_attributes
        )

        # Remember our accuracy, precision, and recall.
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)

        # Update our confidences for each value, as well as the pos + neg values.
        confidences += confidence_list
        all_pos += pos
        all_neg += neg
        
    return summarize_performance_measures(
                                accuracy_list, 
                                precision_list, 
                                recall_list, 
                                all_pos, 
                                all_neg, 
                                confidences,
                                printout = printout
                            )


# Setup data structure to hold learned parameters
# Assume two class labels 0, 1
def init_params(schema, num_attributes):
    parameters = [[0] * (num_attributes-1), [0] * (num_attributes-1)]
    for i in range(1, num_attributes):
        attribute = schema[i]
        count_dict = {}
        for v in attribute.values:
            count_dict[v] = 0

        # put dict in matrix
        parameters[0][i-1] = count_dict
        parameters[1][i-1] = count_dict.copy()
    return parameters

# Perform MLE learning and store values in parameters matrix
def MLE_learn(examples, ex_weights, parameters, num_attributes, M):
    num_class_pos = 0
    num_class_neg = 0

    for i, ex in enumerate(examples):
        class_label = ex[CLASS_ATTR]
        if class_label:
            num_class_pos += ex_weights[i]
        else:
            num_class_neg += ex_weights[i]

        # loop thru example's attribute values and increment appropriately
        for j in range(1, num_attributes):
            attr_val = ex[j]
            parameters[class_label][j-1][attr_val] += ex_weights[i]

    # postcondition: parameters[k][j] now stores dict: # of examples with class k, attribute j

    # Apply m-estimates and divide frequency counts
    for label in (0, 1):
        num_class = num_class_neg if label == 0 else num_class_pos
        for attr in parameters[label]:
            p = 1 / len(attr)
            m = len(attr) if M < 0 else M # Laplace smoothing
            for v in attr:
                attr[v] = (attr[v] + M*p) / (num_class + M)
                
    p_pos = num_class_pos / (num_class_pos + num_class_neg)
    return p_pos, 1-p_pos, parameters

# Evaluate the Naive bayes model on the training set, with given threshold
# Returns the accuracy, precision, and recall
# Also returns a list of predicted confidence, true label pairs
def evaluate_model(example_set, p_pos, p_neg, parameters, num_attributes, threshold=0.0):
    TP, TN, FP, FN = 0, 0, 0, 0
    confidence_list = []
    for ex in example_set:
        confidence = classify(ex, p_pos, p_neg, parameters,num_attributes, binary = False)
        predicted_label = confidence >= threshold
        true_label = ex[CLASS_ATTR]
        confidence_list.append((confidence, true_label))

        TP, TN, FP, FN = cmn.increment_counts(true_label, predicted_label, TP, TN, FP, FN)
        
    accuracy = (TP + TN) / len(example_set)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    all_pos = TP + FN
    all_neg = TN + FP
    
    return accuracy, precision, recall, all_pos, all_neg, confidence_list

# Generalization function used to classify an example given learned parameters and the vector.
def gen_classify(example, parameters):
    #return classify(example, parameters[0], parameters[1], parameters[2], parameters[3], False) * 2 - 1
    return 1 if classify(example, parameters[0], parameters[1], parameters[2], parameters[3], True) else -1

# Generalization function for learning with nbayes on the dataset.
def gen_learn(data_set, example_weights, num_attributes):
    m = .1
    schema = data_set.schema 
    
    p_pos, p_neg, parameters = MLE_learn(
                    data_set,
                    example_weights,
                    init_params(schema, num_attributes), 
                    num_attributes, 
                    m
                )
    
    return (p_pos, p_neg, parameters, num_attributes)

# Given the parameters, classify example as class + or -
# p_pos and p_neg are the prior class probabilities
# Optionally, takes in a treshold value for + classification
def classify(ex, p_pos, p_neg, parameters,num_attributes, binary=False):
    # calculate log p(ex, class = -), log p(ex, class = +)
    log_p_ex_neg = np.log(p_neg)
    log_p_ex_pos = np.log(p_pos)
    for i in range(1, num_attributes):
        log_p_ex_neg += np.log( parameters[0][i-1][ex[i]] )
        log_p_ex_pos += np.log( parameters[1][i-1][ex[i]] )

    # if binary output desired, return class with greater probability
    if binary:
        return log_p_ex_pos >= log_p_ex_neg
    else:
        return log_p_ex_pos - log_p_ex_neg

# Takes the list of accuracy, precision, recall calculated over all the folds
# Takes a pooled list of confidence value true label pairs
# Prints the average and standard deviation of each value and area under ROC
# Returns the average accuracy, precision, recall, area under ROC
def summarize_performance_measures(
            accuracy_list, 
            precision_list, 
            recall_list, 
            all_pos, 
            all_neg, 
            confidence_list, 
            debug=False, 
            printout=True
        ):
    mean_accuracy = np.mean(accuracy_list)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    auc_val = cmn.auc(all_pos, all_neg, confidence_list, debug=debug)

    if printout:
        print("Accuracy:", mean_accuracy, np.std(accuracy_list))
        print("Precision:", mean_precision, np.std(precision_list))
        print("Recall:", mean_recall, np.std(recall_list))
        print("Area under ROC:", auc_val)
    
    return mean_accuracy, mean_precision, mean_recall, auc_val

# Method used to run the nbayes algorithm from the file.
def run(args, print_flag=True):
    # Seed the random generator:
    RNG_SEED = 12345
    random.seed(RNG_SEED)
    cmn.random.seed(RNG_SEED)

    # Set the number of folds:
    NUM_FOLDS = 5

    # Set values based on input arguments:    
    M = args.m_val # m-estimate val
    
    # Process the input:
    example_set = parse_c45(args.name, args.path)
    numerify_nominal(example_set)
    discretize_examples(example_set, args.num_bins, verbose_labels = False)

    # Train the model based on the full sample or with validation.
    if args.full_sample:
        return train_whole_dataset(example_set, M, printout = print_flag)
    else:
        return train_k_fold_dataset(example_set, NUM_FOLDS, M, printout = print_flag)

if __name__ == "__main__":
    run(iap.parser_naive_args().parse_args())

    

    
