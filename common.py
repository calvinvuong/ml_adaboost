import random
import matplotlib.pyplot as plt
import numpy as np

from mldata import ExampleSet

CLASS_ATTR = -1

# Return the area under the ROC curve given confidences.
# all_pos: The number of true positives.
# all_neg: The number of true negatives.
# confidence_list: The list of confidences (and the example true label)
# return: The AUC for the confidence list.
def auc(all_pos, all_neg, confidence_list, debug=False):
    # Sort by confidences.
    sorted_confidence = sorted(confidence_list, key = lambda example: example[0], reverse=True)
    
    area = 0

    prev_TP = 0
    prev_FP = 0
    prev_TN = all_neg
    prev_FN = all_pos
    
    # Used for graphing ROC curve.
    x = []
    y = []

    # Loop through each confidence and modify our beliefs.
    for confidence, true_label in sorted_confidence:
        next_TP, next_TN, next_FN, next_FP = prev_TP, prev_TN, prev_FN, prev_FP
        if (true_label):
            next_TP = prev_TP + 1
            next_FN = prev_FN - 1
        else:
            next_FP = prev_FP + 1
            next_TN = prev_TN - 1
        
        FP_rate = next_FP / all_neg 
        TP_rate = next_TP / all_pos 
        
        area += (FP_rate - prev_FP / all_neg) * (TP_rate + prev_TP / all_pos) / 2

        if debug:
            x.append(prev_FP / all_neg)
            y.append(prev_TP / all_pos)

    
        prev_TP, prev_TN, prev_FN, prev_FP = next_TP, next_TN, next_FN, next_FP

    # Used to debug plots.
    if debug:
        plt.plot(x, y)
        plt.show(block=False)
        plt.pause(.01)
        plt.close

    return area

# Generate k stratified folds on the example dataset
# Returns a list of k ExampleSets
def stratified_folds(dataset, k):
    schema = dataset.schema

    # shuffle dataset
    random.shuffle(dataset)
    
    k_folds = [ExampleSet(schema) for i in range(k)]

    # allocate the labels
    positive_ctr, negative_ctr = 0, 0
    for example in dataset:
        if example[CLASS_ATTR]:
            k_folds[positive_ctr % k].append(example)
            positive_ctr += 1
        else:
            k_folds[negative_ctr % k].append(example)
            negative_ctr += 1
    # shuffle individual folds
    for fold in k_folds:
        random.shuffle(fold)

    return k_folds

# Increment counts based on the true and predicted labels:
# true_label: The true label of the example
# predicted_label: The predicted label of the example.
# TP, TN, FP, FN: True Pos / Neg and False Pos / Neg counts.
# Return: Updated values.
def increment_counts (true_label, predicted_label, TP, TN, FP, FN):
    if true_label == predicted_label:
            if true_label:
                TP += 1
            else:
                TN += 1
    else:
            if true_label:
                FN += 1
            else:
                FP += 1
    return TP, TN, FP, FN

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
    auc_val = auc(all_pos, all_neg, confidence_list, debug=debug)

    if printout:
        print("Accuracy:", mean_accuracy, np.std(accuracy_list))
        print("Precision:", mean_precision, np.std(precision_list))
        print("Recall:", mean_recall, np.std(recall_list))
        print("Area under ROC:", auc_val)
    
    return mean_accuracy, mean_precision, mean_recall, auc_val