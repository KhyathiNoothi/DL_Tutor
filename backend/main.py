import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel, Field


# --------------------------------------------------
# Make the ann folder available for imports
# --------------------------------------------------

ANN_PATH = os.path.join(
    os.path.dirname(__file__),
    "ann"
)

if ANN_PATH not in sys.path:
    sys.path.append(ANN_PATH)


from network import NeuralNetwork
from training import train_network


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="DL Tutor API",
    description="Interactive Deep Learning Tutor Backend",
    version="1.0.0"
)


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "DL Tutor API is running 🧠"
    }


# ==================================================
# ANN TRAINING REQUEST
# ==================================================

class ANNTrainRequest(BaseModel):

    inputs: list[float]

    task: str

    hidden_neurons: int = Field(
        gt=0
    )

    hidden_activation: str

    output_neurons: int = Field(
        gt=0
    )

    target: list[float]

    learning_rate: float = Field(
        gt=0
    )

    epochs: int = Field(
        gt=0
    )

    optimizer: str


# ==================================================
# ANN TRAINING ENDPOINT
# ==================================================

@app.post("/ann/train")
def train_ann(request: ANNTrainRequest):

    # ------------------------------------------------
    # Validate task
    # ------------------------------------------------

    if request.task not in [
        "classification",
        "regression"
    ]:

        return {
            "error": (
                "Task must be "
                "'classification' or 'regression'"
            )
        }


    # ------------------------------------------------
    # Validate hidden activation
    # ------------------------------------------------

    if request.hidden_activation not in [
        "relu",
        "sigmoid",
        "tanh"
    ]:

        return {
            "error": (
                "Hidden activation must be "
                "'relu', 'sigmoid', or 'tanh'"
            )
        }


    # ------------------------------------------------
    # Validate optimizer
    # ------------------------------------------------

    if request.optimizer not in [
        "gradient_descent",
        "momentum",
        "adam"
    ]:

        return {
            "error": (
                "Optimizer must be "
                "'gradient_descent', "
                "'momentum', or 'adam'"
            )
        }


    # ------------------------------------------------
    # Validate input size
    # ------------------------------------------------

    if len(request.inputs) == 0:

        return {
            "error": "At least one input is required"
        }


    # ------------------------------------------------
    # Validate target size
    # ------------------------------------------------

    if len(request.target) != request.output_neurons:

        return {
            "error": (
                "Number of target values must match "
                "number of output neurons"
            )
        }


    # ------------------------------------------------
    # Choose output activation automatically
    # ------------------------------------------------

    if request.task == "classification":

        output_activation = "sigmoid"

    else:

        output_activation = "linear"


    # ------------------------------------------------
    # Create neural network
    # ------------------------------------------------

    network = NeuralNetwork()


    # ------------------------------------------------
    # Add hidden layer
    # ------------------------------------------------

    network.add_layer(
        input_size=len(request.inputs),
        neuron_count=request.hidden_neurons,
        activation=request.hidden_activation
    )


    # ------------------------------------------------
    # Add output layer
    # ------------------------------------------------

    network.add_layer(
        input_size=request.hidden_neurons,
        neuron_count=request.output_neurons,
        activation=output_activation
    )


    # ------------------------------------------------
    # Train network
    # ------------------------------------------------

    result = train_network(
        network=network,
        inputs=request.inputs,
        actual=request.target,
        learning_rate=request.learning_rate,
        epochs=request.epochs,
        optimizer_name=request.optimizer
    )


    # ------------------------------------------------
    # Return result
    # ------------------------------------------------

    return {
        "task": request.task,

        "inputs": request.inputs,

        "hidden_neurons":
            request.hidden_neurons,

        "hidden_activation":
            request.hidden_activation,

        "output_neurons":
            request.output_neurons,

        "output_activation":
            output_activation,

        "target":
            request.target,

        "learning_rate":
            request.learning_rate,

        "epochs":
            request.epochs,

        "optimizer":
            request.optimizer,

        "final_prediction":
            result["final_output"],

        "final_loss":
            result["final_loss"],

        "history":
            result["history"]
    }