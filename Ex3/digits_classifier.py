import os
from abc import abstractmethod
import torch
import typing as tp
from dataclasses import dataclass
import librosa
import numpy as np

MODEL_PATH = "pre_trained_model.pth"


@dataclass
class ClassifierArgs:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    # we will use this to give an absolute path to the data, make sure you read the data using this argument. 
    # you may assume the train data is the same
    path_to_training_data_dir: str = "./train_files"
    path_to_test_data_dir: str = "./test_files/"
    digits_dict = {"/one": 1, "/two": 2, "/three": 3, "/four": 4, "/five": 5}

    # you may add other args here
    batch_size = 10
    sr = 22050


class DigitClassifier():
    """
    You should Implement your classifier object here
    """

    def __init__(self, args: ClassifierArgs):
        # init variables
        self.path_to_training_data = args.path_to_training_data_dir
        self.args = args
        self.train_audio_files = []
        self.audio_files = []

    def euclidean_distance(self, train_audio, test_audio):
        """
        calculate euclidean distance
        """
        return torch.sqrt(torch.sum((test_audio - train_audio) ** 2))

    def dtw_distance(self, train_audio, test_audio):
        """
        calculate dtw distance
        """
        m, n = train_audio.shape[0], test_audio.shape[0]
        dtw_matrix = torch.full(size=[train_audio.shape[0], test_audio.shape[0]], fill_value=torch.inf)
        # Initialize the first cell with the Euclidean distance
        dtw_matrix[0, 0] = self.euclidean_distance(train_audio[0], test_audio[0])
        # Initialize the first column with cumulative distances
        for i in range(1, m):
            dtw_matrix[i, 0] = dtw_matrix[i - 1, 0] + self.euclidean_distance(train_audio[i], test_audio[0])

        # Initialize the first row with cumulative distances
        for j in range(1, n):
            dtw_matrix[0, j] = dtw_matrix[0, j - 1] + self.euclidean_distance(train_audio[0], test_audio[j])

        # Compute matrix using dynamic programing
        for i in range(1, m):
            for j in range(1, n):
                cur_distance = self.euclidean_distance(train_audio[i], test_audio[j])

                dtw_matrix[i, j] = cur_distance + min(dtw_matrix[i - 1, j - 1], dtw_matrix[i, j - 1],
                                                      dtw_matrix[i - 1, j])
        return dtw_matrix[m - 1, n - 1]

    @abstractmethod
    def classify_using_eucledian_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or  a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        euclidean_predictions = []
        for audio_file in audio_files:
            curr_test_mfcc = self.load_audio_and_extract_features(audio_file)

            min_dist = torch.inf
            nearest_neighbor_index = -1

            for i, train in enumerate(self.train_audio_files):
                curr_dist = self.euclidean_distance(train[0], curr_test_mfcc)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    nearest_neighbor_index = i

            euclidean_predictions.append(
                self.train_audio_files[nearest_neighbor_index][1])  # get nearest neighbor label
        return euclidean_predictions

    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a  batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        dtw_predictions = []
        j = 0
        for audio_file in audio_files:

            curr_test_mfcc = self.load_audio_and_extract_features(audio_file)

            min_dist = torch.inf
            nearest_neighbor_index = -1
            print("test example number " + str(j))

            for i, train in enumerate(self.train_audio_files):
                curr_dist = self.dtw_distance(train[0], curr_test_mfcc)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    nearest_neighbor_index = i
            j += 1
            dtw_predictions.append(
                self.train_audio_files[nearest_neighbor_index][1])  # get nearest neighbor label
        return dtw_predictions

    @abstractmethod
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance} - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """
        euclidean_predictions = self.classify_using_eucledian_distance(audio_files)
        dtw_predictions = self.classify_using_DTW_distance(audio_files)

        # save output predictions
        classified_audios = []
        f = open("output.txt", "a")
        for i, filename in enumerate(audio_files):
            filename = filename.replace(self.args.path_to_test_data_dir, "", 1)
            output = '{} - {} - {}\n'.format(filename, euclidean_predictions[i], dtw_predictions[i])
            f.write(output)
            classified_audios.append(output)

        f.close()
        return classified_audios

    def load_data(self):
        """
        load train set and apply mffc
        extract test set filenames
        """
        # load train set and apply mfcc to extract features
        for digit_path in self.args.digits_dict.keys():  # run over each digit directory
            directory_path = self.path_to_training_data + digit_path
            digit_label = self.args.digits_dict[digit_path]
            for filename in os.listdir(directory_path):  # run over each file in digit directory
                if filename.endswith('.wav'):
                    file_path = os.path.join(directory_path, filename)

                    self.train_audio_files.append(
                        [self.load_audio_and_extract_features(file_path),
                         digit_label])

        # extract filenames of test files
        for filename in os.listdir(self.args.path_to_test_data_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(self.args.path_to_test_data_dir, filename)
                self.audio_files.append(file_path)

    def load_audio_and_extract_features(self, file_path):
        audio, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, lifter=26)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        concatenated_mfccs = torch.from_numpy(np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0))
        return concatenated_mfccs


class ClassifierHandler:

    @staticmethod
    def get_pretrained_model() -> DigitClassifier:
        """
        This function should load a pretrained / tuned 'DigitClassifier' object.
        We will use this object to evaluate your classifications
        """
        model_state = torch.load(MODEL_PATH)
        train_audio_files = model_state['train_audio_files']
        args = model_state['args']
        audio_files = model_state['audio_files']
        digitClassifier = DigitClassifier(args)
        digitClassifier.train_audio_files = train_audio_files
        digitClassifier.audio_files = audio_files
        return digitClassifier

    @staticmethod
    def save_model(digitClassifier):
        """
        Saves the train_audio_files and args
        filepath: Path to the file where the weights will be saved.
        """
        train_audio_files = digitClassifier.train_audio_files
        audio_files = digitClassifier.audio_files
        args = digitClassifier.args
        model_state = {
            'train_audio_files': train_audio_files,
            'args': args,
            'audio_files' : audio_files
        }
        torch.save(model_state, MODEL_PATH)


if __name__ == '__main__':
    # classifierArgs = ClassifierArgs()
    # digitClassifier = DigitClassifier(classifierArgs)
    # digitClassifier.load_data()
    # digitClassifier.classify(digitClassifier.audio_files)
    # ClassifierHandler.save_model(digitClassifier)
    digitClassifier = ClassifierHandler.get_pretrained_model()
    classified_audios = digitClassifier.classify(digitClassifier.audio_files)
