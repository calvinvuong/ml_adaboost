import numpy as np

f = open("logreg_test.txt", "r")

iter1_vals = [] 
iter2_vals = []
iter_val = -1
for line in f:
    if "modified" in line.lower():
        iter_val = line.strip("\n")
        print(iter_val)
    elif "Fold" in line:
        acc = float(line.split(" ")[-1])
        if iter_val == "Unmodified":
            iter1_vals.append(acc)
        elif iter_val == "Modified":
            iter2_vals.append(acc)
    elif line.find("Accuracy: ") == 0:
        acc = float(line.split(" ")[1])
        if iter_val == "Unmodified":
            iter1_vals.append(acc)
        elif iter_val == "Modified":
            iter2_vals.append(acc)
f.close()
print(np.asarray(iter1_vals) - np.asarray(iter2_vals))
