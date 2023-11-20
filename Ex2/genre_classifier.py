################### Imports ###################

from abc import abstractmethod
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass
import json
import torchaudio
# import matplotlib.pyplot as plt
import torchaudio.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import librosa
################### Constants ###################

MODEL_PATH = "./model_files/model_weights"
SUFFIX = ".pth"
SR = 22050
# MODEL_PATH = "./model_files/model_weights.pth"

################### CLASSES ###################
class Genre(Enum):
    """
    This enum class is optional and defined for your convinience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genras in your predictions.
    """
    CLASSICAL: int = 0
    HEAVY_ROCK: int = 1
    REGGAE: int = 2


@dataclass
class TrainingParameters:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    batch_size: int = 256
    num_epochs: int = 100
    train_json_path: str = "jsons/train.json"  # you should use this file path to load your train data
    test_json_path: str = "jsons/train.json"  # you should use this file path to load your test data
    # other training hyper parameters


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.0001
    num_of_classes = 3
    # mfcc hyper-parameters
    mfcc_features = 100
    frames = 1664


class MusicClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, opt_params: OptimizationParameters, **kwargs):
        """
        This defines the classifier object.
        - You should defiend your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization and you are welcome to experiment
        """
        self.opt_params = opt_params
        self.weights_shape = [opt_params.num_of_classes, (opt_params.mfcc_features * opt_params.frames)]
        self.weights = (-5e-4) + (1e-3) * torch.randn(self.weights_shape)

        self.bias = torch.zeros((opt_params.num_of_classes))

    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        mfcc_transform = transforms.MFCC(
            sample_rate=SR,
            n_mfcc=100,  # Number of MFCC coefficients to compute
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 100}
        )

        # Apply MFCC transform to the waveform
        mfcc = mfcc_transform(wavs.squeeze())
        return mfcc

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """

        z = (self.weights @ feats.T).T + self.bias
        output_scores = torch.softmax(z, dim=1)
        return output_scores

    def backward(self, feats: torch.Tensor, output_scores: torch.Tensor, labels: torch.Tensor):
        """
        this function should perform a backward pass through the model.
        - calculate loss
        - calculate gradients
        - update gradients using SGD

        Note: in practice - the optimization process is usually external to the model.
        We thought it may result in less coding needed if you are to apply it here, hence 
        OptimizationParameters are passed to the initialization function
        """
        batch_size = labels.shape[0]

        # calc gradients
        dw = ((1 / batch_size) * torch.matmul(feats.T, (output_scores - labels))).T # calcs loss
        db = (1 / batch_size) * torch.sum(output_scores - labels, dim=0)

        # update weights and bias
        self.weights -= self.opt_params.learning_rate * dw
        self.bias -= self.opt_params.learning_rate * db

    def get_weights_and_biases(self):
        """
        This function returns the weights and biases associated with this model object, 
        should return a tuple: (weights, biases)
        """
        return self.weights, self.bias

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should recieve a torch.Tensor of shape [batch, channels, time] (float tensor) 
        and a output batch of corresponding labels [B, 1] (integer tensor)
        """
        feats = self.exctract_feats(wavs)
        feats = feats.reshape(len(wavs), -1)
        y_pred = self.forward(feats)
        return torch.argmax(y_pred, dim=1).unsqueeze(dim=1)

    ################### Added functions ###################
    def accuracy(self, output_scores: torch.Tensor, labels: torch.Tensor, is_train=True) -> float:
        """
        Calculates the accuracy of the predicted labels.
        output_scores: predicted labels tensor of shape
        labels: ground truth labels tensor of shape
        Returns total_correct
        """
        if is_train:
            equals = torch.all(output_scores == labels, dim=1)
        else:
            output_scores = output_scores.squeeze()
            equals = (output_scores == labels)
        total_correct = torch.sum(equals)
        return total_correct.item()

    def set_weights_and_biases(self, weights, bias):
        """
        This function returns the weights and biases associated with this model object,
        should return a tuple: (weights, biases)
        """
        self.weights = weights
        self.bias = bias


class ClassifierHandler:

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        train_set = ClassifierHandler.load_data(training_parameters.train_json_path)

        optimizationParameters = OptimizationParameters()
        musicClassifier = MusicClassifier(optimizationParameters)

        train_loader = DataLoader(train_set, batch_size=training_parameters.batch_size, shuffle=True)
        train_accuracy, test_accuracy = [], []
        for epoch in range(1, training_parameters.num_epochs + 1):
            total_correct = 0
            for data in train_loader:
                wavs, labels = data[0], data[1]
                feats = musicClassifier.exctract_feats(wavs)
                feats = feats.reshape(len(wavs), -1) # flattern each wav
                output_scores = musicClassifier.forward(feats)
                musicClassifier.backward(feats, output_scores, labels)

                total_correct += musicClassifier.accuracy(output_scores, labels, is_train=True)

            train_accuracy.append((total_correct / len(train_set)) * 100)
            print(f"Epoch {epoch}/{training_parameters.num_epochs}, Train accuracy: {train_accuracy[epoch - 1]:.4f}%")
            cur_test_accuracy = ClassifierHandler.evaluate_model(training_parameters, musicClassifier)
            test_accuracy.append(cur_test_accuracy)


        # save model and plot train and test accuracies
        ClassifierHandler.save_model(musicClassifier, MODEL_PATH + SUFFIX)
        # ClassifierHandler.plot_accuracies(train_accuracy, test_accuracy, training_parameters.num_epochs)

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights /
        hyperparameters and return the loaded model
        """
        optimizationParameters = OptimizationParameters()
        musicClassifier = MusicClassifier(optimizationParameters)
        ClassifierHandler.load_model(musicClassifier, "./model_files/model_weights.pth")
        return musicClassifier

    ################### Added functions ###################

    @staticmethod
    def load_data(file_name, is_train=True):
        """
        Load wav files and their labeling
        """
        # Load JSON file
        with open(file_name, 'r') as f:
            data = json.load(f)
        # Load audio files
        data_set = []
        for audio_dic in data:
            audio_path, label = audio_dic['path'], audio_dic['label']
            audio = librosa.load(audio_path)[0]  # Load audio file using librosa
            # audio = torchaudio.load(audio_path)[0]  # Load audio file using librosa
            data_set.append([audio, ClassifierHandler.get_genre_representation(label, is_train)])
        return data_set



    @staticmethod
    def save_model(musicClassifier, filepath: str):
        """
        Saves the model weights and biases to a file.
        filepath: Path to the file where the weights will be saved.
        """
        weights, bias = musicClassifier.get_weights_and_biases()
        model_state = {
            'weights': weights,
            'bias': bias
        }
        torch.save(model_state, filepath)

    @staticmethod
    def load_model(musicClassifier, filepath: str):
        """
        Loads the model weights and biases from a file.
        filepath: Path to the file containing the weights.
        """
        model_state = torch.load(filepath)
        weights = model_state['weights']
        bias = model_state['bias']

        musicClassifier.set_weights_and_biases(weights, bias)

    @staticmethod
    def get_genre_representation(genre_name, is_train=True):
        """
        get genre representation - a vector for training and a number for testing
        """
        if genre_name == "classical":
            result = [torch.tensor([1, 0, 0]), 0]
        elif genre_name == "heavy-rock":
            result = [torch.tensor([0, 1, 0]), 1]
        elif genre_name == "reggae":
            result = [torch.tensor([0, 0, 1]), 2]

        return result[0] if is_train else result[1]

    @staticmethod
    def evaluate_model(training_parameters, musicClassifier: MusicClassifier):
        """
        evaluate model performence over the test set.
        returns model accuracy
        """
        test_set = ClassifierHandler.load_data(training_parameters.test_json_path, is_train=False)
        test_loader = DataLoader(test_set, batch_size=training_parameters.batch_size, shuffle=True)
        total_correct = 0
        for data in test_loader:
            wavs, labels = data[0], data[1]
            output_scores = musicClassifier.classify(wavs)
            total_correct += musicClassifier.accuracy(output_scores, labels, is_train=False)
        accuracy = (total_correct / len(test_set)) * 100
        print(f"Test Accuracy: {accuracy:.4f}%")
        return accuracy

    # @staticmethod
    # def plot_accuracies(train_accuracy, test_accuracy, num_epochs):
    #     """
    #     plot train and test accuracys
    #     """
    #     epochs = range(1, num_epochs + 1)
    #     plt.plot(epochs, train_accuracy, label='Train Accuracy')
    #     plt.plot(epochs, test_accuracy, label='Test Accuracy')
    #
    #     plt.title('Training and Testing Accuracies')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.legend()
    #     plt.show()

# This is the main we used to trained and test our classifier
if __name__ == "__main__":
    """
    inits ClassifierHandler and TrainingParameters
    loads a pretrained classifier
    evaluate the pretrain model over the test set
    """
    handler = ClassifierHandler()
    training_parameters = TrainingParameters(batch_size=256, num_epochs=100)
    ClassifierHandler.train_new_model(training_parameters)
    musicClassifier = handler.get_pretrained_model()
    handler.evaluate_model(training_parameters, musicClassifier)
