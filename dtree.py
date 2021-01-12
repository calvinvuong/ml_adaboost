#!/usr/bin/env python3
from mldata import *
from heuristic import *
import numpy as np
import random
import time
import sys
import os

class Node(object):
    """
    Describes a node in the decision tree
    """
    class Type:
        """
        Enumerate types of nodes
        """
        ID         = 'ID'
        BINARY     = 'BINARY'
        NOMINAL    = 'NOMINAL'
        CONTINUOUS = 'CONTINUOUS'
        LEAF       = 'LEAF'
        
    def __init__(self, attribute_id, ntype, parent, depth, boundary=None):
        self.id = attribute_id
        self.type = ntype
        self.boundary = boundary
        self.children = []
        self.parent = parent
        self.depth = depth

    # add branch
    def add_child(self, value, subtree):
        self.children.append((value, subtree))

class Leaf(Node):
    # Defines a leaf on the tree (termination)
    def __init__(self, label, parent, depth, reason="not specified", frac_positive=-1):
        super().__init__(None, Node.Type.LEAF, parent, depth)
        self.label = label
        self.reason = reason
        self.frac_positive = frac_positive

# Define the possition for the class attribute in the list (last index)
CLASS_ATTR = -1

# Wrapper function to perform the ID3 algorithm
def ID3(dataset, ex_weights, attribute_set):
    return _ID3(dataset, ex_weights, attribute_set, -1, 0)

# performs the ID3 algorithm
def _ID3(dataset, ex_weights, attribute_set, parent, depth):
    # Schema Information
    schema = dataset.schema
    num_examples = np.sum(ex_weights)
    
    # check if dataset is pure
    positive_labels = 0
    negative_labels = 0
    for i, ex in enumerate(dataset):
        if ex[CLASS_ATTR]:
            positive_labels += ex_weights[i]
        else:
            negative_labels += ex_weights[i]

    if positive_labels == num_examples:
        # pure node; label +
        return Leaf(True, parent, depth, "pure +", positive_labels / num_examples)
    elif positive_labels == 0:
        # pure node; label -
        return Leaf(False, parent, depth, "pure -", positive_labels / num_examples)

    # All attributes exhausted or max depth reached
    if depth >= MAX_TREE_DEPTH or attribute_set.count(False) == 0: 
        return majority_leaf(parent, depth, "exhausted no attr", positive_labels, num_examples)
    
    # select attribute to split on
    best_attribute, attribute_boundary = choose_attribute(attribute_set, dataset, ex_weights)

    # no attributes yield information gain
    if best_attribute == -1:
        return majority_leaf(parent, depth, "exhausted no attr", positive_labels, num_examples)
    
    # FOR NOMINAL/DISCRETE ATTRIBUTES ONLY
    if schema[best_attribute].type != Feature.Type.CONTINUOUS:
        # create new internal test node for this attribute
        attribute_values = schema[best_attribute].values
        test_node = Node(best_attribute, schema[best_attribute].type, parent, depth)
        # remove attribute from set of future tests
        attribute_set[best_attribute] = True
        
        # split dataset based on possible values of selected attribute
        for v in attribute_values:
            subset = ExampleSet(dataset.schema)
            subset_ex_weights = []
            for i, ex in enumerate(dataset):
                if ex[best_attribute] == v:
                    subset.append(ex)
                    subset_ex_weights.append(ex_weights[i])

            # if no examples in this branch, make leaf labeled with majority class
            if len(subset) == 0:
                subtree = majority_leaf(parent, depth, "exhausted empty subset", positive_labels, num_examples)
            else:
                # recurse the tree
                subtree = _ID3(subset, subset_ex_weights, attribute_set, best_attribute, depth+1)
            test_node.add_child(v, subtree)
        return test_node
    
    # FOR CONTINUOUS ATTRIBUTES
    else:
        # create new internal test node for this attribute
        attribute_values = schema[best_attribute].values
        test_node = Node(best_attribute, schema[best_attribute].type, parent, depth, boundary=attribute_boundary)
        
        # Divide examples into subsets
        leq_subset = ExampleSet(dataset.schema)
        leq_subset_weights = []
        gt_subset = ExampleSet(dataset.schema)
        gt_subset_weights = []
        for i, ex in enumerate(dataset):
            if ex[best_attribute] <= attribute_boundary:
                leq_subset.append(ex)
                leq_subset_weights.append(ex_weights[i])
            else:
                gt_subset.append(ex)
                gt_subset_weights.append(ex_weights[i])

        # if no examples in <= branch, make leaf labeled with majority class
        if len(leq_subset) == 0:
            subtree = majority_leaf(parent, depth, "exhausted empty subset", positive_labels, num_examples)
        else:
            # recurse
            subtree = _ID3(leq_subset, leq_subset_weights, attribute_set, best_attribute, depth+1)
        test_node.add_child("<=", subtree)

        # > subset split
        if len(gt_subset) == 0:
            subtree = majority_leaf(parent, depth, "exhausted empty subset", positive_labels, num_examples)
        else:
            # recurse
            subtree = _ID3(gt_subset, gt_subset_weights, attribute_set, best_attribute, depth+1)
        test_node.add_child(">", subtree)

        return test_node

# Returns a leaf node with label equal to the majority example class
def majority_leaf(parent, depth, message, positive_labels, num_examples):
    if positive_labels >= num_examples / 2:
        # majority node; label +
        return Leaf(True, parent, depth, "+ " + message, positive_labels/num_examples)
    else:
        # majority node; label -
        return Leaf(False, parent, depth, "+ " + message, positive_labels/num_examples)

# Generalization function that allows a dtree to be created on data with weights and attributes.
def gen_learn(data_set, example_weights, num_attributes):
    global MAX_TREE_DEPTH
    global USE_GAIN_RATIO

    MAX_TREE_DEPTH = 1
    USE_GAIN_RATIO = False
    
    attributes_tested = [True] + [False] * (num_attributes-1)

    return ID3(data_set, example_weights, attributes_tested)

# Generalization function to classify an example given the dtree parameters.
def gen_classify(example, parameters):
    classification = classify(example, parameters)
    return 1 if classification else -1

# Takes a feature vector representation of an example and returns the class label given by dtree
def classify(example, dtree):
    # Base case: leaf node reached
    if dtree.type == Node.Type.LEAF:
        return dtree.label

    # Recurse and check down the tree.
    attribute_test = dtree.id
    if dtree.type == Node.Type.CONTINUOUS:
        if example[attribute_test] <= dtree.boundary:
            return classify(example, dtree.children[0][1])
        elif example[attribute_test] > dtree.boundary:
            return classify(example, dtree.children[1][1])
        else:
            print("Classification error.")
            return -1
        
    for branch, child in dtree.children:
        if branch == example[attribute_test]:
            return classify(example, child)
    return -1
    

# Return the index of the attribute that will be used for splitting
def choose_attribute(attribute_set, examples, ex_weights):
    return selected_attribute(attribute_set, examples, ex_weights, USE_GAIN_RATIO)
    
# Generate k stratified folds on the example dataset
# Returns a list of k ExampleSets
def stratified_folds(dataset, k):
    # shuffle dataset
    random.shuffle(dataset)
    
    k_folds = [ExampleSet(schema) for i in range(k)]
    # allocate the labels
    positive_ctr, negative_ctr = 0, 0
    for example in dataset:
        if example[CLASS_ATTR] == True:
            k_folds[positive_ctr % k].append(example)
            positive_ctr += 1
        else:
            k_folds[negative_ctr % k].append(example)
            negative_ctr += 1
    # shuffle individual folds
    for fold in k_folds:
        random.shuffle(fold)

    return k_folds

# Perform stratified k-fold cross validation to train and evaluate tree
# Return average accuracy over k-folds
def k_fold_cross_validation(dataset, k):
    # Generate folds
    folds = stratified_folds(dataset, k)
    
    avg_accuracy, avg_size, avg_depth = 0, 0, 0
    firsts = []
    for i, holdout in enumerate(folds):
        # train on all folds except holdout
        training_set = ExampleSet([ex for fold in folds if fold != holdout for ex in fold])
        example_weights = [1/len(training_set)] * len(training_set)
        
        attributes_tested = [True] + [False] * (num_attributes-1)
        decision_tree = ID3(training_set, example_weights, attributes_tested)

        tree_size, tree_depth = tree_stats(decision_tree)

        # validate on holdout
        accuracy = evaluate_tree(holdout, decision_tree)
        avg_accuracy += accuracy
        avg_size += tree_size
        avg_depth += tree_depth
        first_attr = None if decision_tree.type == Node.Type.LEAF else decision_tree.id
        firsts.append(first_attr)
    
    # Final averages.
    avg_accuracy /= len(folds)
    avg_size /= len(folds)
    avg_depth /= len(folds)
    common_first = max(set(firsts), key=firsts.count)

    if PRINT:
        print("Cross-Validation Average")
        print_output(avg_accuracy, avg_size, avg_depth, common_first)
        
    return avg_accuracy

# Train the ID3 tree on whole example set
# Evaluate tree on whole example set, return accuracy
def train_whole(examples):
    attributes_tested = [True] + [False] * (num_attributes-1)

    example_weights = [1/len(examples)] * len(examples)
    decision_tree = ID3(examples, example_weights, attributes_tested)

    tree_size, tree_depth = tree_stats(decision_tree)
    accuracy = evaluate_tree(examples, decision_tree)
    first_attr = None if decision_tree.type == Node.Type.LEAF else decision_tree.id

    if PRINT:
        print_output(accuracy, tree_size, tree_depth, first_attr)
    return accuracy

def print_output(accuracy, size, depth, first):
    print("Accuracy:", accuracy)
    print("Size:", size)
    print("Maximum Depth:", depth)
    print("First Feature:", schema[first].name)

# Classify all examples in validate_set using dtree and return the accuracy
def evaluate_tree(validate_set, dtree):
    total_match = 0
    for ex in validate_set:
        predicted = classify(ex, dtree)
        total_match += 1 if predicted == ex[CLASS_ATTR] else 0

    accuracy = total_match / len(validate_set)
    return accuracy
        
# Returns the size and depth of the tree
def tree_stats(root):
    if root.type == Node.Type.LEAF:
        return 1, 0 # size 1, depth 0
    # recurse on subtrees
    subtree_stats = [tree_stats(sub[1]) for sub in root.children]
    size = sum([s[0] for s in subtree_stats]) + 1
    depth = max([s[1] for s in subtree_stats]) + 1
    return size, depth
    
# debug purposes only 
def print_tree(root):
    global num_nodes
    if type(root) is Leaf:
        print("Leaf:", root.label, "parent:", root.parent, "depth:", root.depth, "%.3f" %(root.frac_positive), root.reason)
        return
    print(root.id, "parent:", root.parent, "depth:", root.depth)
    print("children", [(t[0], t[1].id) for t in root.children])
    print("------------")
    for s in root.children:
        print_tree(s[1])

# Function to allow for calling within the jupyter notebook.
# Runs the dtree given input parameters.
def main(input_parameters, allow_print=True):
    # Global Parameters
    global schema 
    global num_attributes
    global USE_CROSS_VALIDATION
    global MAX_TREE_DEPTH
    global USE_GAIN_RATIO
    global PRINT
    
    if len(input_parameters) < 5:
        print("Usage: python3 dtree.py data-path cv max-depth gain-ratio")
        sys.exit(0)
        
    # File information
    file_path = str(input_parameters[1])
    file_location, file_name = os.path.split(file_path)
    
    # Decision tree parameters
    USE_CROSS_VALIDATION = int(input_parameters[2]) == 0
    MAX_TREE_DEPTH = float('inf') if int(input_parameters[3]) <= 0 else int(input_parameters[3])
    USE_GAIN_RATIO = int(input_parameters[4]) == 1

    # Define run parameters
    PRINT = allow_print
    NUM_FOLDS = 5
    RNG_SEED = 12345
    random.seed(RNG_SEED)

    #Get the dataset
    example_set = parse_c45(file_name, file_location)
    
    # Necessary Parameters for the DTREE
    num_attributes = len(example_set[0]) - 1
    schema = example_set.schema

    # Run on the appropriate methods for kfold or whole dataset.
    if USE_CROSS_VALIDATION:
        avg_accuracy = k_fold_cross_validation(example_set, 5)
        return avg_accuracy
    else:
        accuracy = train_whole(example_set)
        return accuracy

if __name__ == '__main__':
    main(sys.argv)
    
