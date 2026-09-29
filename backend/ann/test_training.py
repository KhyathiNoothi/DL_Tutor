from network import NeuralNetwork
from training import train_network
import numpy as np
np.random.seed(42)

print("======================================")
print("       MULTI-LAYER ANN TRAINING")
print("======================================")


# ======================================
# 1. INPUTS
# ======================================

input_size = int(
    input("\nHow many inputs? ")
)

inputs = []

for i in range(input_size):

    value = float(
        input(f"Enter input x{i + 1}: ")
    )

    inputs.append(value)


# ======================================
# 2. TASK TYPE
# ======================================

print("\nChoose task:")
print("1. Binary Classification")
print("2. Regression")

task_choice = input(
    "Enter choice: "
)


if task_choice == "1":

    task = "classification"

elif task_choice == "2":

    task = "regression"

else:

    print("Invalid task!")
    exit()


# ======================================
# 3. NETWORK ARCHITECTURE
# ======================================

network = NeuralNetwork()


# Hidden layer

hidden_neurons = int(
    input(
        "\nHow many neurons in hidden layer? "
    )
)


print("\nChoose hidden-layer activation:")
print("1. ReLU")
print("2. Sigmoid")
print("3. Tanh")

hidden_activation_choice = input(
    "Enter choice: "
)


if hidden_activation_choice == "1":

    hidden_activation = "relu"

elif hidden_activation_choice == "2":

    hidden_activation = "sigmoid"

elif hidden_activation_choice == "3":

    hidden_activation = "tanh"

else:

    print("Invalid activation!")
    exit()


network.add_layer(
    input_size,
    hidden_neurons,
    hidden_activation
)


# ======================================
# 4. OUTPUT LAYER
# ======================================

output_neurons = int(
    input(
        "\nHow many neurons in output layer? "
    )
)


if task == "classification":

    output_activation = "sigmoid"

else:

    output_activation = "linear"


network.add_layer(
    hidden_neurons,
    output_neurons,
    output_activation
)


# ======================================
# 5. LOSS FUNCTION
# ======================================



# ======================================
# 6. TARGET VALUES
# ======================================

actual = []

print("\nEnter target values:")

for i in range(output_neurons):

    value = float(
        input(
            f"Target y{i + 1}: "
        )
    )

    actual.append(value)


# ======================================
# 7. BACKPROPAGATION
# ======================================



# ======================================
# 8. LEARNING RATE
# ======================================

learning_rate = float(
    input("\nEnter learning rate: ")
)


# ======================================
# 9. EPOCHS
# ======================================

epochs = int(
    input("How many epochs? ")
)


# ======================================
# 10. OPTIMIZER
# ======================================

print("\nChoose optimizer:")

print("1. Gradient Descent")
print("2. Momentum")
print("3. Adam")

optimizer_choice = input(
    "Enter choice: "
)


if optimizer_choice == "1":

    optimizer_name = "gradient_descent"

elif optimizer_choice == "2":

    optimizer_name = "momentum"

elif optimizer_choice == "3":

    optimizer_name = "adam"

else:

    print("Invalid optimizer!")
    exit()


# ======================================
# 11. DISPLAY SETTINGS
# ======================================

print("\n======================================")
print("          TRAINING SETTINGS")
print("======================================")

print("Task:",
      task)

print("Inputs:",
      inputs)

print("Hidden neurons:",
      hidden_neurons)

print("Hidden activation:",
      hidden_activation)

print("Output neurons:",
      output_neurons)

print("Output activation:",
      output_activation)

print("Optimizer:",
      optimizer_name)

print("Learning rate:",
      learning_rate)

print("Epochs:",
      epochs)


# ======================================
# 12. TRAIN NETWORK
# ======================================

print("\n======================================")
print("             TRAINING...")
print("======================================")

result = train_network(
    network=network,
    inputs=inputs,
    actual=actual,
    learning_rate=learning_rate,
    epochs=epochs,
    optimizer_name=optimizer_name,
    task=task
)


# ======================================
# 13. TRAINING SUMMARY
# ======================================

print("\n======================================")
print("          TRAINING COMPLETE")
print("======================================")


print(
    "Final prediction:",
    result["final_output"]
)

print(
    "Actual:",
    actual
)

print(
    "Final loss:",
    result["final_loss"]
)

print(
    "Optimizer:",
    result["optimizer"]
)




# ======================================
# 14. EPOCH SUMMARY
# ======================================

print("\n======================================")
print("          TRAINING PROGRESS")
print("======================================")


for record in result["history"]:

    epoch = record["epoch"]

    # Print every epoch for small runs
    # For large runs, print every 10 epochs

    if epochs <= 20 or epoch % 10 == 0 or epoch == 1:

        print(
            f"Epoch {epoch:3d} | "
            f"Prediction: {record['prediction']} | "
            f"Loss: {record['loss']:.6f}"
        )