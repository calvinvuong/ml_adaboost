from mldata import *

import math
import numpy as np

# Find the indexes for the features which are continuous
# example_set: The set of examples we want to look at.
# return: A list of indexes for which features are continuous
def cont_attr_indices(example_set):
    cont_indices = []

    for feature_index in range(len(example_set.schema)):
        if example_set.schema[feature_index].type == Feature.Type.CONTINUOUS:
            cont_indices.append(feature_index)
    
    return cont_indices

# Find the index to new value mappings for nominal values.
# example_set: The set of examples we want to look at.
# return: A dict of indexes to possible feature value -> index mapping
def nom_attr_labels(example_set):
    nom_labels = {}

    for feature_index in range(len(example_set.schema)):
        if example_set.schema[feature_index].type == Feature.Type.NOMINAL:
            
            value_index = {}
            value_count = 1

            for value in example_set.schema[feature_index].values:
                value_index[value] = value_count
                value_count += 1

            nom_labels[feature_index] = value_index
    
    return nom_labels

# find the minimum / maximum value in the examples for each feature
# example_set: The set of examples
# cont_indices: The list of indices for continous features.
# return: the minimum and maximum value within each 
def cont_feature_min_max(example_set, cont_indices):
    cont_feature_min_max = {}

    for example in example_set:
        for cont_index in cont_indices:
            example_value = example[cont_index]

            min_max = cont_feature_min_max.get(cont_index, [float('inf'), float('-inf')])

            min_max[0] = min(min_max[0], example_value)
            min_max[1] = max(min_max[1], example_value)

            cont_feature_min_max[cont_index] = min_max
    return cont_feature_min_max

# determine feature value labels by "<left bound> to <right bound>" (Verbose):
# cont_feature_min_max: the dict of feature indexes to min/max values.
# num_bins: The number of intended bins to split the data into.
# return: dict of the feature index mapped to the feature values.
def all_feature_bin_labels_verbose(cont_feature_min_max, num_bins):
    bin_labels = {}

    for cont_index, (min_val, max_val) in cont_feature_min_max.items():
        feature_bin_labels = []

        bin_width = (max_val - min_val) / num_bins

        bin_left_bound = min_val
        bin_right_bound = bin_left_bound + bin_width

        for bin_index in range(num_bins):
            feature_bin_labels.append(str(bin_left_bound) + " to " + str(bin_right_bound))

            # Shift bounds
            bin_left_bound = bin_right_bound
            bin_right_bound += bin_width

        bin_labels[cont_index] = feature_bin_labels

    return bin_labels

# determine feature value labels by x in range of [0, num_bins) (Verbose):
# cont_feature_min_max: the dict of feature indexes to min/max values.
# num_bins: The number of intended bins to split the data into.
# return: dict of the feature index mapped to the feature values.
def all_feature_bin_labels(cont_feature_min_max, num_bins):
    bin_labels = {}
    for cont_index, min_max in cont_feature_min_max.items():
        bin_labels[cont_index] = list(range(num_bins))
    
    return bin_labels

# modify the example_set determine which discrete feature value to assign
# example_set: the set of example data
# cont_feature_min_max: the minimum and maximum values for each feature
# bin_labels: dict of the desired labels to be used for each value key (feature index)
def bin_examples(example_set, cont_feature_min_max, all_feature_bin_labels):
    for example in example_set:
        for cont_index, feature_bin_labels in all_feature_bin_labels.items():
            example_value = example[cont_index]

            min_val, max_val = cont_feature_min_max[cont_index]

            num_bins = len(feature_bin_labels)
            bin_width = (max_val - min_val) / num_bins

            value_bin = math.floor((example_value - min_val)/bin_width)
            
            example[cont_index] = feature_bin_labels[value_bin if value_bin != num_bins else num_bins - 1] # To check to make sure the last point (max) is included

# Make sure that the schema for the example set is consistent
# example_set: The modified example data that had been discretized.
# all_feature_bin_labels: All the labels for each feature values for each feature index.
def update_schema(example_set, all_feature_bin_labels):
    updated_schema = []
    for feature_index in range(len(example_set.schema)):
        target_feature = example_set.schema[feature_index]

        if (feature_index in all_feature_bin_labels):
            updated_schema.append(Feature(target_feature.name, Feature.Type.NOMINAL, all_feature_bin_labels[feature_index]))
        else:
            updated_schema.append(target_feature)

    # CALVIN's EDIT
    updated_schema = Schema(updated_schema)
    
    # Used to update all schemas no matter how they are accessed.
    example_set.schema = updated_schema
    for example in example_set:
        example.schema = updated_schema

# discretize all of the examples from the example set
# example_set: the examples of the data
# num_bins: The number of desired bins to discretize continuous features.
# verbose_labels: In case feature values should be verbose.
def discretize_examples(example_set, num_bins, feature_min_max=None, bin_labels=None, verbose_labels = False):

    if feature_min_max == None and bin_labels == None:
        indices = cont_attr_indices(example_set)
        feature_min_max = cont_feature_min_max(example_set, indices)

        if(verbose_labels):
            bin_labels = all_feature_bin_labels_verbose(feature_min_max, num_bins) #4 is the constant for number of bins TODO Handle 0
            print("\n=======================  INDICES FOR CONTINUOUS  =======================")
            print(indices)
            print("\n=======================  MIN_MAX FOR INDEX =======================")
            print(feature_min_max)
            print("\n=======================  Bin Labels =======================")
            print(bin_labels)
        else:
            bin_labels = all_feature_bin_labels(feature_min_max, num_bins)

    #Modify the examples to bin them
    bin_examples(example_set, feature_min_max, bin_labels)

    #Update the schema of the example_set to represent the updated data
    update_schema(example_set, bin_labels)

    return feature_min_max, bin_labels

# Numerifies the nominal attributes
# exmaple_set the set of examples for which nominal attributes should have their values re-mapped.
def numerify_nominal(example_set):
    nom_feature_labels = nom_attr_labels(example_set)

    for example in example_set:
        for nom_index, feature_bin_labels in nom_feature_labels.items():
            example[nom_index] = feature_bin_labels[example[nom_index]]

    updated_schema_labels={}
    for nom_index, feature_bin_labels in nom_feature_labels.items():
        updated_schema_labels[nom_index] = list(feature_bin_labels.values())

    update_schema(example_set, updated_schema_labels)

# Takes all attributes and standardizes so that the are with mean 0 and unit variance.
# IMPORTANT: Only works well for gaussian distributions
# example_set: The set of examples required to be standardized.
# feature_mean: optional input mean
# feature_std: Optional input standard deviation.
def standardize_input(example_set, feature_mean=None, feature_std=None):
    example_set_schema = example_set.schema

    # Initialize standard deviation and mean dictionaries.
    if feature_mean == None and feature_std == None:
        feature_mean, feature_std = {}, {}
        
        for feature_index in range(len(example_set_schema)):
            feature_type = example_set_schema[feature_index].type
            if feature_type == Feature.Type.NOMINAL or feature_type == Feature.Type.CONTINUOUS:
                feature_mean[feature_index] = 0
                feature_std[feature_index] = 0

        # Calculate the mean and standard deviation.
        fill_feature_mean(example_set, feature_mean)
        fill_feature_std(example_set, feature_std, feature_mean)

    # Calculate the new values for the examples.
    for example in example_set:
        for feature_index, mean in feature_mean.items():
            example[feature_index] = (example[feature_index] - feature_mean[feature_index]) / feature_std[feature_index]
    
    new_schema_labels = {}

    # Update the labels for continuous attributes that were modified.
    for feature_index in feature_mean:
        feature = example_set.schema[feature_index]
        if feature.type == Feature.Type.NOMINAL:
            new_labels = []
            for value in feature.values:
                new_labels.append((value - feature_mean[feature_index]) / feature_std[feature_index])
            new_schema_labels[feature_index] = new_labels
    
    update_schema(example_set, new_schema_labels)
    
    return feature_mean, feature_std

# Fills the mean set with values
# example_set: The data to compute the mean from
# mean_set: The empty set to hold the mean for each attribute.
def fill_feature_mean(example_set, mean_set):
    number_examples = len(example_set)
    for example in example_set:
        for feature_index in mean_set:
            mean_set[feature_index] = mean_set[feature_index] + example[feature_index] 

    for feature_index in mean_set:
            mean_set[feature_index] = mean_set[feature_index] / number_examples 
            
# Fills the std set with values for features
# example_set: The data to compute the std of
# feature_std: the standard deviation dict
# The previously computed mean used to compute std
def fill_feature_std(example_set, feature_std, feature_mean):
    number_examples = len(example_set)

    for example in example_set:
        for feature_index in feature_std:
            feature_std[feature_index] = feature_std[feature_index] + math.pow((example[feature_index] - feature_mean[feature_index]),2)

    for feature_index in feature_std:
        feature_std[feature_index] = math.sqrt(feature_std[feature_index] / number_examples)

#  ======================================================================= #
#   Testing possible research extension: Normalization and LOG transform.  #
#  ======================================================================= #
# Get all min / max values
def feature_min_max(example_set):
    feature_min_max = {}
    for example in example_set:
        for cont_index in range(len(example_set.schema)):
            feature_type = example_set.schema[cont_index].type
            if feature_type == Feature.Type.NOMINAL or feature_type == Feature.Type.CONTINUOUS:
                example_value = example[cont_index]

                min_max = feature_min_max.get(cont_index, [float('inf'), float('-inf')])

                min_max[0] = min(min_max[0], example_value)
                min_max[1] = max(min_max[1], example_value)

                feature_min_max[cont_index] = min_max
    return feature_min_max

# Perform normalization on the example set.
def normalize_input(example_set, feature_mean={}):
    example_set_schema = example_set.schema

    if not feature_mean:
        for feature_index in range(len(example_set_schema)):
            feature_type = example_set_schema[feature_index].type
            #if feature_type == Feature.Type.CONTINUOUS:
            if feature_type == Feature.Type.NOMINAL or feature_type == Feature.Type.CONTINUOUS:
                feature_mean[feature_index] = 0

        fill_feature_mean(example_set, feature_mean)

    feat_min_max = feature_min_max(example_set)

    for example in example_set:
        for feature_index, mean in feature_mean.items():
            example[feature_index] = (example[feature_index] - mean) / (feat_min_max[feature_index][1] - feat_min_max[feature_index][0])

    new_schema_labels = {}

    for feature_index in feature_mean:
        feature = example_set.schema[feature_index]
        if feature.type == Feature.Type.NOMINAL:
            new_labels = []
            for value in feature.values:
                new_labels.append((value - feature_mean[feature_index]) / (feat_min_max[feature_index][1] - feat_min_max[feature_index][0]))
            new_schema_labels[feature_index] = new_labels
    
    update_schema(example_set, new_schema_labels)
    return feature_mean

# Log transform the example set:
def log_transform_input(example_set):
    example_set_schema = example_set.schema

    feat_min_max = feature_min_max(example_set)

    for example in example_set:
        for feature_index in range(len(example_set_schema)):
            if example_set_schema[feature_index].type == Feature.Type.CONTINUOUS:
                value = np.absolute(feat_min_max[feature_index][0]) + example[feature_index]
                example[feature_index] = np.log(value) if value != 0 else 0 



