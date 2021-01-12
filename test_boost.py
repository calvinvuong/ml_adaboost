from mldata import *
import boost
import dtree
import nbayes
import logreg
import common as cmn
import preprocessing
import random
import numpy as np
import sys

# GLOBAL Definitions
dtree.MAX_TREE_DEPTH = 1
NUM_BINS = 25
NUM_FOLDS = 5

# Script to allow for bulk running boosting iterations.
def main():
    # Simple input argument processing
    if len(sys.argv) < 5:
        print("Usage: python3 test_boost.py dataset algorithm iterations cross-validation t-test")
        print("dataset: voting, volcanoes, spam")
        print("cross-valiation = 0 if cross validate, 1 if train whole")
        print("t-test = 1 if need results per fold")
        print("iterations can be a range; if you are using a range please be of the form a-b")
        print("iterations can also be comma separated values of the form a,b,c,d,e")
        print("iterations CANNOT be a combined form of both")
        sys.exit(0)
        
    name = sys.argv[1]
    path = "data/" + name
    
    #Determine algorithm to be used
    algorithm = sys.argv[2]
    if algorithm == "dtree":
        algorithm_alt = dtree
        algorithm_alt.MAX_TREE_DEPTH = 1
    elif algorithm == "nbayes":
        algorithm_alt = nbayes
    elif algorithm == "logreg":
        algorithm_alt = logreg

    # Determine what range of iterations should be run (sequential, list, or individual)
    if "-" in sys.argv[3]:
        start_iteration = int(sys.argv[3].split("-")[0])
        end_iteration = int(sys.argv[3].split("-")[1])
        iter_vals = range(start_iteration, end_iteration+1)
    elif "," in sys.argv[3]:
        iter_vals = [int(x) for x in sys.argv[3].split(",")]
    else:
        iter_vals = [int(sys.argv[3])]

    # Whether to cross validate the training
    cross_validate = int(sys.argv[4]) == 0

    # Whether to print out fold information
    t_test = False
    if len(sys.argv) >= 6:
        t_test = int(sys.argv[5]) == 1

    # Parse the dataset
    example_set = parse_c45(name, path)

    # Normalization for algorithms
    if algorithm == "nbayes":
        preprocessing.numerify_nominal(example_set)
        preprocessing.discretize_examples(example_set, NUM_BINS, verbose_labels = False)
    elif algorithm == "logreg":
        preprocessing.numerify_nominal(example_set)
        preprocessing.standardize_input(example_set)

    # Learn based on the input arguments
    test(example_set, algorithm, algorithm_alt, iter_vals, cross_validate, t_test)
    
# Method to perform the training 
def test(example_set, algorithm, algorithm_alt, iter_vals, cross_validate, t_test):
    # Train a new boost for each iteration value
    for num_iterations in iter_vals:
        if cross_validate:
            print(t_test)
            print("%d Iterations:" %(num_iterations))
            boost.k_fold_cross_validation(example_set, NUM_FOLDS, algorithm_alt, num_iterations, print_fold=t_test)
            print()
        else:
            print("%d Iterations:" %(num_iterations))
            boost.train_whole(example_set, algorithm_alt, num_iterations)
            print()

main()
