#!/usr/bin/env python3

from mldata import *
import dtree
import nbayes
import logreg
import common as cmn
import preprocessing
import random
import numpy as np
import input_arg_processing as iap

import progressbar

# GLOBAL Parameter Declaration
RNG_SEED = 12345
NUM_FOLDS = 5
NUM_BINS = 25
ITERATIONS = 3

# Research extension:
EXT = False
STD_THRESH = 2
MISCLASSIFICATION_THRESH = 3

# Seed the modeling
random.seed(RNG_SEED)
cmn.random.seed(RNG_SEED)

# Method to run the algorithm of boosting given input arguments.
def run(args):
    # These would be read from input.
    algorithm = args.algorithm
    algorithm_alt = dtree # default value
    ITERATIONS = args.iterations

    # Determine whether to use the research extension
    global EXT
    EXT = args.extension == 1

    # Whether or not to print out fold information
    print_option = args.print_folds == 1
    
    # Process the dataset
    example_set = parse_c45(args.name, args.path)
    
    # Modify data based on algorithm type (generalization)
    if algorithm == "nbayes":
        algorithm_alt = nbayes
        preprocessing.numerify_nominal(example_set)
        preprocessing.discretize_examples(example_set, NUM_BINS, verbose_labels = False)
    elif algorithm == "logreg":
        algorithm_alt = logreg
        preprocessing.numerify_nominal(example_set)
        preprocessing.standardize_input(example_set)

    # Run boosting with each algorithm
    if args.full_sample:
        train_whole(example_set, algorithm_alt, ITERATIONS)
    else:
        k_fold_cross_validation(example_set, NUM_FOLDS, algorithm_alt, ITERATIONS, print_fold=print_option)
        

# K fold cross validation with boosting the given algorithm
def k_fold_cross_validation(dataset, k, algorithm_alt, num_iterations, print_fold=False):
    # reset rng seed explicitly for testing script
    random.seed(RNG_SEED)
    cmn.random.seed(RNG_SEED)

    # Define information about incoming data
    schema = dataset.schema    
    num_attributes = len(schema)-1

    # Generate folds
    folds = cmn.stratified_folds(dataset, k)

    # Define lists to hold fold information.
    accuracy_list = []
    precision_list = []
    recall_list = []
    confidences = []
    all_pos, all_neg = 0, 0

    # Loop over each fold partitioning
    for i, holdout in enumerate(folds):
        # Create the training set.
        training_set = ExampleSet([ex for fold in folds if fold != holdout for ex in fold])

        # Remeber each trained classifier across boosting layers
        classifiers = []
        train_error = -1
        j=0

        # initial example weights
        example_weights = [1/len(training_set)] * len(training_set)

        # Define consecutive missclasifications for research extension
        consecutive_misclassifications = [0] * len(training_set)
    
        # While we have not converged, or hit our iteration limits
        while((train_error != 0 and train_error < .5) and j < num_iterations):
            j+=1

            #Train the given algorithm with the training set.
            parameter = algorithm_alt.gen_learn(
                training_set, 
                example_weights, 
                num_attributes
            )

            # Determine the error and new example wieghts given the training set, and learned parameters
            # Ie update the model
            train_error, example_weights = iterate(
                training_set, 
                example_weights, 
                algorithm_alt, 
                parameter, 
                classifiers, 
                consecutive_misclassifications
            )

            # Final metric calculation
            mean = np.mean(example_weights)
            std = np.std(example_weights)

        #Evaluate the boosted classifier:
        accuracy, precision, recall, pos, neg, confidence_list = evaluate(holdout, algorithm_alt,classifiers)
        
        #Print out each fold for t testing
        if print_fold:
            print("Fold ", i, "Accuracy", accuracy)

        # Remember our accuracy, precision, and recall.
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)

        # Update our confidences for each value, as well as the pos + neg values.
        confidences += confidence_list
        all_pos += pos
        all_neg += neg

    # Output the final metrics for the cv algorithm.
    return cmn.summarize_performance_measures(
        accuracy_list, 
        precision_list, 
        recall_list, 
        all_pos, 
        all_neg, 
        confidences,
        printout = True
    )

# Train the whole dataset on the given algorithm
def train_whole(dataset, algorithm_alt, num_iterations):
    # reset rng seed explicitly for testing script
    random.seed(RNG_SEED)
    cmn.random.seed(RNG_SEED)

    # Define constants for learning
    schema = dataset.schema    
    num_attributes = len(dataset[0]) -1

    # Remember each classifier that is created while boosting
    classifiers = []
    train_error = -1
    i = 0

    # initial example weights
    example_weights = [1/len(dataset)] * len(dataset)

    # Consectutive missclasification counts for research
    consecutive_misclassifications = [0] * len(dataset)

    # While the boosting hasn't converged or hit it's limit
    while((train_error != 0 and train_error < .5) and i < num_iterations):
        i+= 1

        # Learn on the algorithm and dataset.
        parameter = algorithm_alt.gen_learn(
                dataset, 
                example_weights, 
                num_attributes
        )

        # Determine the new updated weights for the next iteration
        train_error, example_weights = iterate(
                dataset, 
                example_weights, 
                algorithm_alt, 
                parameter, 
                classifiers, 
                consecutive_misclassifications
        )

    # Evaluate the final boosted classifier
    accuracy, precision, recall, all_pos, all_neg, confidence_list = evaluate(dataset, algorithm_alt,classifiers)

    # Output final metrics.
    return cmn.summarize_performance_measures(
        [accuracy], 
        [precision], 
        [recall], 
        all_pos,
        all_neg,
        confidence_list,
        printout = True
    )

# Method which takes in the data with current example weights and updates the weights based on the performance and classifications.
def iterate(dataset, example_weights,  model, parameters, classifiers, consecutive_misclassifications):
    # Find the classification of each example
    classifications = classify_examples(dataset, model, parameters, consecutive_misclassifications)
    
    # Determine the training error based on example weights and classifications
    train_error = weighted_training_error(example_weights, classifications)

    # Calculate the weight of the final classifier that was learned
    classi_weight = classifier_weight(train_error)

    #Remember the classifier (parameters) and it's weight for voting.
    classifiers.append((classi_weight, parameters))

    # Determine the error and new weights for the next iteration
    return train_error, update_weights(example_weights, classi_weight, classifications, consecutive_misclassifications)

# Calculate the updated weights for the next iteration.
# Takes as input list of prev example weights, list of classifier weights, and list classifications
# where classifications[i] = 1 if true label and predicted label agree, -1 otherwise
def update_weights(prev_weights, classifier_weight, classifications, consecutive_misclassifications):
    new_weights = []

    # Calculate the new weights for each example based on the example's classification
    for i, label_match in enumerate(classifications):
        new_weights.append(prev_weights[i] * np.exp(-classifier_weight * label_match))

    # Normalization
    weight_sum = sum(new_weights)
    new_weights = [ x / weight_sum for x in new_weights]

    # Zero weights of examples where misclassification is consistent.
    if EXT: # if research extension option set
        mean = np.mean(new_weights)
        std = np.std(new_weights)
        weight_threshold = mean + STD_THRESH * std # threshold for example weights to apply extension
        for i, weight in enumerate(new_weights):
            if weight > weight_threshold and consecutive_misclassifications[i] > MISCLASSIFICATION_THRESH:
                new_weights[i] = 0 
        weight_sum = sum(new_weights)
        new_weights = [ x / weight_sum for x in new_weights] # normalize again
        
    return new_weights

# Given examples, classify using the given algorithm (model) and the learned parameters
def classify_examples(examples, model, parameters, consecutive_misclassifications):
    classifications = []
    # For each example
    for i, example in enumerate(examples):
        #Classify
        classification = model.gen_classify(example, parameters)
        true_label = 1 if example[-1] else -1
        classifications.append(classification * true_label)

        # And count misclassifications (or reset)
        if EXT and true_label != classification:
            consecutive_misclassifications[i] += 1
        elif EXT:
            consecutive_misclassifications[i] = 0
            
    return classifications

# Weighted training error:
# Calculate the sum of the weights for misclassified examples
def weighted_training_error(prev_weights, classifications):
    error = 0
    for classification_idx in range(len(classifications)):
        if(classifications[classification_idx] == - 1):
            error += prev_weights[classification_idx]

    return error

# Weight of the classifier:
# the weight of the trained classifier:
def classifier_weight (error):
    return .5 * np.log((1-error) / error) if error != 0 else 1

# Weighted majority vote for boosting classification
# Look at all of the classifiers, 
def classify(example, model, ensemble):
    result = 0
    total_weight = 0
    for classifier in ensemble:
        result += classifier[0] * model.gen_classify(example, classifier[1])
        total_weight += classifier[0]

    return result / total_weight

# Evaluate the model based on counting metrics
def evaluate(example_set, model, ensemble, threshold=0):
    TP, TN, FP, FN = 0, 0, 0, 0
    confidence_list = []
    # print(ensemble)
    for ex in example_set:
        confidence = classify(ex, model, ensemble)
        predicted_label = confidence >= threshold
        true_label = ex[-1]
        confidence_list.append((confidence, true_label))

        TP, TN, FP, FN = cmn.increment_counts(true_label, predicted_label, TP, TN, FP, FN)
        
    accuracy = (TP + TN) / len(example_set)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    all_pos = TP + FN
    all_neg = TN + FP
    
    return accuracy, precision, recall, all_pos, all_neg, confidence_list
    
    
if __name__=='__main__':
    run(iap.parser_boost_args().parse_args())

