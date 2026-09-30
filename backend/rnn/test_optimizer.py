import numpy as np

from .optimizer import update_parameters


# -----------------------------------------
# Initial parameters
# -----------------------------------------

Wxh = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])

Whh = np.array([
    [0.5, 0.6],
    [0.7, 0.8]
])

bh = np.array([
    0.1,
    0.2
])

Why = np.array([
    0.3,
    0.4
])

by = 0.5


# -----------------------------------------
# Gradients
# -----------------------------------------

dL_dWxh = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

dL_dWhh = np.array([
    [0.05, 0.06],
    [0.07, 0.08]
])

dL_dbh = np.array([
    0.1,
    0.2
])

dL_dWhy = np.array([
    0.3,
    0.4
])

dL_dby = 0.5


# -----------------------------------------
# Learning rate
# -----------------------------------------

learning_rate = 0.1


# -----------------------------------------
# Update
# -----------------------------------------

result = update_parameters(
    Wxh=Wxh,
    Whh=Whh,
    bh=bh,
    Why=Why,
    by=by,
    dL_dWxh=dL_dWxh,
    dL_dWhh=dL_dWhh,
    dL_dbh=dL_dbh,
    dL_dWhy=dL_dWhy,
    dL_dby=dL_dby,
    learning_rate=learning_rate
)


# -----------------------------------------
# Display
# -----------------------------------------

print("Updated Wxh:")
print(result["Wxh"])

print("\nUpdated Whh:")
print(result["Whh"])

print("\nUpdated bh:")
print(result["bh"])

print("\nUpdated Why:")
print(result["Why"])

print("\nUpdated by:")
print(result["by"])