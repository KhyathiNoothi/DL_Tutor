from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ann.network import NeuralNetwork
from ann.training import train_network

from cnn.network import CNN
from cnn.training import train_cnn as train_cnn_model
from rnn.text_training_model import train_text_dataset
from rnn.inference import predict_text
import numpy as np


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="DL Tutor API",
    description="Interactive Deep Learning Tutor Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trained_ann = None
trained_cnn = None
trained_rnn = None
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
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(value) for value in obj]
    if isinstance(obj, tuple):
        return [convert_numpy(value) for value in obj]
    return obj

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

    global trained_ann

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
    trained_ann = network
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
# ANN PREDICTION REQUEST
# ==================================================

class ANNPredictRequest(BaseModel):

    inputs: list[float]


# ==================================================
# ANN PREDICTION ENDPOINT
# ==================================================

@app.post("/ann/predict")
def predict_ann(request: ANNPredictRequest):

    if trained_ann is None:

        return {
            "error": (
                "ANN has not been trained yet. "
                "Train the ANN first."
            )
        }

    if len(request.inputs) == 0:

        return {
            "error": "At least one input is required"
        }

    result = trained_ann.forward(request.inputs)

    return {
        "inputs": request.inputs,
        "prediction": result["final_output"],
        "forward": result
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

    global trained_cnn

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
    trained_cnn = cnn
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

    # ==================================================
# CNN PREDICTION REQUEST
# ==================================================

class CNNPredictRequest(BaseModel):

    image: list[list[float]]

    stride: int = Field(
        default=1,
        gt=0
    )

    padding: int = Field(
        default=0,
        ge=0
    )


# ==================================================
# CNN PREDICTION ENDPOINT
# ==================================================

@app.post("/cnn/predict")
def predict_cnn(request: CNNPredictRequest):

    if trained_cnn is None:
        return {
            "error": "CNN has not been trained yet. Train the CNN first."
        }

    if len(request.image) == 0:
        return {
            "error": "Image cannot be empty"
        }

    result = trained_cnn.forward(
        image=request.image,
        stride=request.stride,
        padding=request.padding
    )

    return {
        "image": request.image,
        "stride": request.stride,
        "padding": request.padding,
        "prediction": convert_numpy(result)
    }
# ==================================================
# RNN TRAINING REQUEST
# ==================================================

class RNNTrainRequest(BaseModel):

    texts: list[str]

    targets: list[float]

    learning_rate: float = Field(
        gt=0
    )

    epochs: int = Field(
        gt=0
    )

    hidden_size: int = Field(
        gt=0
    )

    embedding_size: int = Field(
        gt=0
    )


# ==================================================
# RNN PREDICTION REQUEST
# ==================================================

class RNNPredictRequest(BaseModel):

    text: str
# ==================================================
# RNN TRAINING ENDPOINT
# ==================================================

@app.post("/rnn/train")
def train_rnn_endpoint(request: RNNTrainRequest):

    global trained_rnn

    if len(request.texts) == 0:
        return {
            "error": "At least one training text is required"
        }

    if len(request.texts) != len(request.targets):
        return {
            "error": (
                "Number of texts must match "
                "number of targets"
            )
        }

    for target in request.targets:

        if target not in [0, 1]:
            return {
                "error": "Targets must be 0 or 1"
            }

    result = train_text_dataset(
        texts=request.texts,
        targets=request.targets,
        learning_rate=request.learning_rate,
        epochs=request.epochs,
        hidden_size=request.hidden_size,
        embedding_size=request.embedding_size
    )

    trained_rnn = result

    return {
        "task": "text_binary_classification",

        "texts": request.texts,

        "targets": request.targets,

        "learning_rate": request.learning_rate,

        "epochs": request.epochs,

        "hidden_size": request.hidden_size,

        "embedding_size": request.embedding_size,

        "vocabulary": result["vocabulary"].word_to_id,

        "final_loss": result["final_loss"],

        "history": result["history"]
    }
# ==================================================
# RNN PREDICTION ENDPOINT
# ==================================================

@app.post("/rnn/predict")
def predict_rnn_endpoint(request: RNNPredictRequest):

    if trained_rnn is None:

        return {
            "error": (
                "RNN has not been trained yet. "
                "Train the RNN first."
            )
        }

    result = predict_text(
        text=request.text,
        vocabulary=trained_rnn["vocabulary"],
        embedding=trained_rnn["embedding"],
        model=trained_rnn["model"]
    )

    return {
    "text": result["text"],
    "tokens": result["tokens"],
    "token_ids": result["token_ids"],
    "embeddings": result["embeddings"],
    "forward_history": result["forward_history"],
    "final_hidden": result["final_hidden"],
    "logit": result["logit"],
    "probability": result["probability"],
    "prediction": result["prediction"],
    "label": result["label"]
}