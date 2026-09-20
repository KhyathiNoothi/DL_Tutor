from neuron import weighted_sum
import numpy as np

inputs = [2, 3]
weights = [0.5, 0.2]
bias = 1

result = weighted_sum(inputs, weights, bias)

print("Weighted sum:", result)