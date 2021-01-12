import math
import numpy as np
#from scipy import special

# Returns the sigmoid calculation on the dot product of the weights and example
# w: The weights, where the first weight is a bias
# x: The actual example values, where x_0 = 1 for bias.
def input_weight_activation(w, x):
    x_vector = x[:-1]
    x_vector[0] = 1 # id feature not important
    product = np.dot(w, x_vector)
    return sigmoid(product)

# Returns the value of the sigmoid function on input x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
