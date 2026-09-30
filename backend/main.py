from fastapi import FastAPI
from pydantic import BaseModel, Field

from ann.network import NeuralNetwork
from ann.training import train_network

from cnn.network import CNN
from cnn.training import train_cnn as train_cnn_model


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

    if len(request.inputs) == 0:

        return {
            "error": "At least one input is required"
        }

    if len(request.target) != request.output_neurons:

        return {
            "error": (
                "Number of target values must match "
                "number of output neurons"
            )
        }

    if request.task == "classification":

        output_activation = "sigmoid"

    else:

        output_activation = "linear"

    network = NeuralNetwork()

    network.add_layer(
        input_size=len(request.inputs),
        neuron_count=request.hidden_neurons,
        activation=request.hidden_activation
    )

    network.add_layer(
        input_size=request.hidden_neurons,
        neuron_count=request.output_neurons,
        activation=output_activation
    )

    result = train_network(
        network=network,
        inputs=request.inputs,
        actual=request.target,
        learning_rate=request.learning_rate,
        epochs=request.epochs,
        optimizer_name=request.optimizer,
        task=request.task
    )

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


# ==================================================
# CNN TRAINING REQUEST
# ==================================================

class CNNTrainRequest(BaseModel):

    image: list[list[float]]

    kernels: list[list[list[float]]]

    stride: int = Field(
        default=1,
        gt=0
    )

    padding: int = Field(
        default=0,
        ge=0
    )

    pool_size: int = Field(
        default=2,
        gt=0
    )

    pool_stride: int = Field(
        default=2,
        gt=0
    )

    target: float

    learning_rate: float = Field(
        gt=0
    )

    epochs: int = Field(
        gt=0
    )


# ==================================================
# CNN TRAINING ENDPOINT
# ==================================================

@app.post("/cnn/train")
def train_cnn_endpoint(request: CNNTrainRequest):

    if request.target not in [0, 1]:

        return {
            "error": "Target must be 0 or 1"
        }

    if len(request.image) == 0:

        return {
            "error": "Image cannot be empty"
        }

    if len(request.kernels) == 0:

        return {
            "error": "At least one kernel is required"
        }

    cnn = CNN(
        kernels=request.kernels,
        pool_size=request.pool_size,
        pool_stride=request.pool_stride,
        dense_output_size=1,
        dense_activation="linear"
    )

    result = train_cnn_model(
        cnn=cnn,
        image=request.image,
        target=request.target,
        learning_rate=request.learning_rate,
        epochs=request.epochs,
        stride=request.stride,
        padding=request.padding
    )

    return {

        "task": "binary_classification",

        "image":
            request.image,

        "kernels":
            request.kernels,

        "stride":
            request.stride,

        "padding":
            request.padding,

        "pool_size":
            request.pool_size,

        "pool_stride":
            request.pool_stride,

        "target":
            request.target,

        "learning_rate":
            request.learning_rate,

        "epochs":
            request.epochs,

        "final_probability":
            result["final_probability"],

        "final_loss":
            result["final_loss"],

        "history":
            result["history"]
    }