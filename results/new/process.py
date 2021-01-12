import numpy as np

f = open("t_test_dtree_spam.txt", "r")

iter1_vals = [] 
iter2_vals = []
iter_val = -1
for line in f:
    if "Iterations" in line:
        iter_val = int(line.split(" ")[0])
    elif "Fold" in line:
        acc = float(line.split(" ")[-1])
        if iter_val == 1:
            iter1_vals.append(acc)
        elif iter_val == 30:
            iter2_vals.append(acc)
    elif line.find("Accuracy: ") == 0:
        acc = float(line.split(" ")[1])
        if iter_val == 1:
            iter1_vals.append(acc)
        elif iter_val == 30:
            iter2_vals.append(acc)
f.close()
print(np.asarray(iter1_vals) - np.asarray(iter2_vals))
