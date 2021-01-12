from mldata import *
import numpy as np
import math
import operator
import random
import sys
import traceback

CLASS_ATTR = -1

# Used for the research extension
RESTRICT = False
TYPE = None
SIGMA = 0

def set_restrict(t_type, sigma):
    global RESTRICT
    global SIGMA
    global TYPE
    RESTRICT = True
    SIGMA = sigma
    TYPE = t_type
    
def remove_restrict():
    global RESTRICT
    RESTRICT = False
    TYPE = None
    
# Calculates one class label's entropy
# numerator: The count of examples
# denominator: The total examples
def entropy_step(numerator, denominator):
    # if total count is 0, return 0.
    if denominator == 0:
        return 0

    p = numerator / denominator
    if p == 0:
        return 0

    return (- p * math.log(p, 2))

# Calculate the entorpy of each attribute value:
# attribute_counts: A list of each attribute values' occurences (positive class, and negative class, total)
# Returns the entropy associated with each class value
def entropy(attribute_counts):

    entropy_values = []
    for count in attribute_counts:
        partition_count = count[2]
        entropy_values.append(
            [entropy_step(count[0], partition_count) + entropy_step(count[1], partition_count), partition_count])

    return entropy_values

# Calculate the entropy of a partition given across attribute labels.
# entropy: a list of each attribute value's positive / negative class label counts.
# total: number of examples
# Returns the information gain of an attribute.
def sum_entropy(entropies, total):
    entropy_sum = 0

    for entropy, count in entropies:
        entropy_sum += (count / total) * entropy

    return entropy_sum

# Calculate an attribute's entropy within the partition.
# class_label_counts: The counts for class labels. Will ONLY use attribute value totals.
# returns: the entropy of the class label, and 1 if entropy is 0.
def attribute_entropy(class_label_counts, example_count):
    entropy_sum = 0

    for count in class_label_counts:
        entropy_sum += entropy_step(count[2], example_count)

    return entropy_sum if entropy_sum != 0 else 1

# Take in the class label counts and determine the overarching partition entropy
# class_label_counts: The counts for class labels in the example set. Important: depends on "hidden" attribute examples.
def partition_entropy(class_label_counts):
    pos_count = 0
    neg_count = 0
    total_count = 0

    # Use the totals for each attribute stored already.
    for counts in class_label_counts:
        pos_count += counts[0]
        neg_count += counts[1]
        total_count += counts[2]

    return entropy_step(pos_count, total_count) + entropy_step(neg_count, total_count)

# Counts the class label occurrences of each attribute value
# attribute_index: the index for the attribute value of interest.
# examples: the example_set.
# Returns a list of each attribute value's count in the class labels
def count_class_labels(attribute_index, examples, ex_weights):

    attribute_values = examples.schema[attribute_index].values
    count = {}

    # Initialize all counts attribute value counts to none.
    for attribute in attribute_values:
        count[attribute] = [0, 0, 0]

    # Iterate through examples and count the attribute.
    for i, example in enumerate(examples):

        attribute_value_count = count[example[attribute_index]]

        if example[CLASS_ATTR]:
            attribute_value_count[0] += ex_weights[i]
        else:
            attribute_value_count[1] += ex_weights[i]

        attribute_value_count[2] += ex_weights[i]

    return count.values()

# Find the map of boundary tests to their gains.
# attribute_index: The index for the attribute whose tests we want to look through
# examples: The exampleset with our data
# use_gain_ratio: Whether we want to split based on gain ratio or not.
def boundary_gains(attribute_index, examples, ex_weights, use_gain_ratio):
    boundary_set = [] # set of boundary values
    boundary_map = {} # map of boundary to IG or GainRatio
    boundary_counts = [] # for calculating attribute entropy

    # pair up examples with their weights
    example_tuples = [(ex, ex_weights[i]) for i, ex in enumerate(examples)]
    sorted_examples = sorted(example_tuples, key=lambda ex_t: (ex_t[0][attribute_index], ex_t[0][CLASS_ATTR]))
    #sorted_examples = sorted(examples, key=lambda ex: (ex[attribute_index], ex[CLASS_ATTR]))
    tot_pos, tot_neg = 0, 0
    
    # Base case, initialize comparison
    prev = sorted_examples[0][0]
    prev_weight = sorted_examples[0][1]
    
    tot_pos, tot_neg = increment_if_true_else(prev[CLASS_ATTR], prev_weight, tot_pos, tot_neg)
    bound_count = prev_weight
    #bound_count = 1

    # LOOP and count total examples, as well as boundary bins
    for index in range(1, len(sorted_examples)):
        current_example = sorted_examples[index][0]
        current_example_weight = sorted_examples[index][1]

        tot_pos, tot_neg = increment_if_true_else(current_example[CLASS_ATTR], current_example_weight, tot_pos, tot_neg)

        # Boundary Detected.
        if(prev[CLASS_ATTR] != current_example[CLASS_ATTR]):
            boundary = (prev[attribute_index] + current_example[attribute_index]) / 2

            if len(boundary_set) == 0 or boundary != boundary_set[-1]:
                boundary_set.append(boundary)
                boundary_counts.append([0, 0, bound_count])

            bound_count = 0

        bound_count += current_example_weight
        
        prev = current_example
        prev_weight = current_example_weight
        
    boundary_counts.append([0, 0, bound_count])
    
    # If no boundaries, end
    if len(boundary_set) == 0:
        return {}

    if use_gain_ratio:
        #hx = attribute_entropy(boundary_counts, len(examples))
        hx = attribute_entropy(boundary_counts, np.sum(ex_weights))
    
    cum_pos, cum_neg = 0, 0
    boundary_ptr = 0

    # LOOP and find entropies for each border.
    for ex, ex_w in sorted_examples:
        current_bound = boundary_set[boundary_ptr]
        if ex[attribute_index] > current_bound:
            # Create necessary datastructure to calculate entropy
            class_label_counts = [0, 0]
            class_label_counts[0] = [cum_pos, cum_neg, cum_pos + cum_neg] # <=
            class_label_counts[1] = [tot_pos-cum_pos, tot_neg-cum_neg, tot_pos + tot_neg - cum_pos - cum_neg] # >

            # calculate IG or GainRatio heuristic
            raw_gain = partition_entropy(class_label_counts) - sum_entropy(entropy(class_label_counts), len(examples))

            # Modify for gain ratio
            if use_gain_ratio:
                raw_gain /= hx
            
            # Map the bound to the gain
            boundary_map[current_bound] = raw_gain

            if boundary_ptr == len(boundary_set) - 1:
                break # from loop thru sorted examples
            
            #Consider next boundary.
            boundary_ptr += 1

        cum_pos, cum_neg = increment_if_true_else(ex[CLASS_ATTR], ex_w, cum_pos, cum_neg)
            
    return boundary_map

# Helper method to increment two counters if a condition is true.
# condition: Boolean condition
# true_increment: counter to increment if true
# false_increment: counter to increment if false
# returns: Tuple of the true and false increments
def increment_if_true_else(condition, incr_val, true_increment, false_increment):
    if condition:
        true_increment += incr_val
    else:
        false_increment += incr_val
    
    return true_increment, false_increment


#Priamry mehtod used to selected the best attribute (and test)
# attributes: An list of booleans determining whether an attribute at that index is still available (false) or finished (true)
# examples: The exampleset for the data
# use_gain_Ratio: Boolean determining whether we want to use the gain ratio calculation.
def selected_attribute(attributes, examples, ex_weights, use_gain_ratio):
    # Constants
    NO_TEST = -1
    NO_GAIN = -1
    
    attributes_ig = []

    # For each possible attribute
    for index in range(len(attributes)):
        gain = (NO_GAIN,NO_TEST)
        # If the attribute is still available
        if not attributes[index]:
            attribute_type = examples.schema.features[index].type
            
            #In the case the attribute is continuous
            if (attribute_type==Feature.Type.CONTINUOUS):
                # Get the gain (IG or IGR) associated with each boundary test.
                boundary_map = boundary_gains(index, examples, ex_weights, use_gain_ratio)

                if len(boundary_map) != 0:
                    attr_test, gain = max(boundary_map.items(), key=operator.itemgetter(1))
                    gain = (gain, attr_test)
            # If the attribute is discrete
            else:
                # Get data counts
                class_label_counts = count_class_labels(index, examples, ex_weights)
                raw_gain = partition_entropy(class_label_counts) - sum_entropy(entropy(class_label_counts), np.sum(ex_weights))

                if use_gain_ratio:
                    raw_gain /= attribute_entropy(class_label_counts, np.sum(ex_weights))

                # Set attribute gain.
                gain = (raw_gain, NO_TEST)

        attributes_ig.append(gain)



    if RESTRICT:
        info_gain = [val[0] for val in attributes_ig if val[0] != -1] # list of info gain for all attributes
        if TYPE == 'std':
            prune_attributes(attributes_ig, attributes, np.mean(info_gain) - np.std(info_gain) * SIGMA)
        elif TYPE == 'quantile':
            prune_attributes(attributes_ig, attributes, five_percent_ig(info_gain))

    return max_ig(attributes_ig)

# Takes the information gain for each attribute, and returns the index for the attribute, as well as the contiuous boundry (if applicable, -1 otherwise)
# attributes_ig: each attribute's information gain matched by their index in the list
# returns: a tuple of the gain index, as well as attribute test if applicable.
def max_ig(attributes_ig):
    max_ig = 0
    max_ig_index = -1

    for attr_tuple_index in range(len(attributes_ig)):
        
        tuple_ig = attributes_ig[attr_tuple_index][0]

        if(tuple_ig > max_ig):
            max_ig = tuple_ig
            max_ig_index = attr_tuple_index

    return (max_ig_index, attributes_ig[max_ig_index][1] if max_ig_index != -1 else -1)


# Research extension: removes attributes from consideration that
# have information gain or gain ratio below mean - std * sigma
def prune_attributes(attributes_ig, attributes, cutoff):
    for attr_tuple_index in range(len(attributes_ig)):
        if (not attributes[attr_tuple_index]):
            if(attributes_ig[attr_tuple_index][0] < cutoff):
                attributes[attr_tuple_index] = True
                
def five_percent_ig(attributes_ig):
    sorted_gain = sorted(attributes_ig)
    index = math.floor(len(sorted_gain)*.05)
    if (index != 0):
        return sorted_gain[index]
    else:
        return 0


