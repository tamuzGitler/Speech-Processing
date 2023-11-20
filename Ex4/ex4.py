################### Imports ###################

import numpy as np
import sys


################### Functions ###################
def add_underscore(z):
    """
    adds '_' char at the begining and at the end of the given string (z), and also between every characters
    :param string: z
    :return: z_with_blanks
    """
    result = '_' + '_'.join(z)
    return result + '_'


def create_index_dictionary(tokens):
    """
    Creating a dictionary to hold the indices of each alphabet token for fast retrieval
    :param tokens: alphabet provided
    :return: dictionary {sign : index}
    """
    index_dictionary = {'_': 0}
    for i, char in enumerate(tokens):
        index_dictionary[char] = i + 1
    return index_dictionary


def print_p(p: float):
    """
    Formatted print as required
    :param p: result probability
    :return: None (print)
    """
    print("%.3f" % p)


################### Main ###################

if __name__ == '__main__':
    # check if the given input is legal
    assert (len(sys.argv) == 4)

    # unpack arguments and load the prob matrix from the given path
    matrix_path, z, tokens, = sys.argv[1], sys.argv[2], sys.argv[3]
    y = np.load(matrix_path).T  # transpose to get matrix shape as we saw in lecture - Phonemes*Time
    rows, cols = y.shape

    index_dictionary = create_index_dictionary(tokens)

    # (1) add blanks to labeling
    z_with_blanks = add_underscore(z)

    # (2) init prob_matrix with Base conditions - adds epsilon between all the phonemes
    prob_matrix = np.zeros((len(z_with_blanks), cols))
    prob_matrix[0, 0], prob_matrix[1, 0] = y[0, 0], y[1, 0]

    # (3) fill prob_matrix
    # run over the columns - over time
    for t in range(1, prob_matrix.shape[1]):
        # run over the rows - over phonemes
        for s in range(prob_matrix.shape[0]):
            # Definition:
            # alpha_1 = Alpha(s−1,t−1),alpha_2 = Alpha(s,t−1), alpha_3 = Alpha(s−2,t−1)

            Zs = z_with_blanks[s]
            Ys_t = y[index_dictionary[Zs], t]  # get probability

            # check if Z(s-2) exists:
            if s >= 2:
                Zprev_s = z_with_blanks[s - 2]  # Z(s-2)
                alpha_1 = prob_matrix[s - 1, t - 1]
                alpha_2 = prob_matrix[s, t - 1]
                alphas = alpha_1 + alpha_2

                # checks if we are in case 2 where  (Zs != ϵ) and (Zs != Zs−2)
                if not (Zs == '_' or Zs == Zprev_s):
                    alpha_3 = prob_matrix[s - 2, t - 1]
                    alphas += alpha_3

            else:  # s == 0 or s == 1 ('_' or Zs is the first phoneme), in this case alpha_3 doesn't exists.
                alpha_2 = prob_matrix[s, t - 1]  # relevant for s == 0 and s == 1
                alphas = alpha_2

                if s == 1:
                    alpha_1 = prob_matrix[s - 1, t - 1]
                    alphas += alpha_1

            # update prob_matrix for all cases
            prob_matrix[s, t] = alphas * Ys_t

    # extract probability of z
    row_length, col_length = prob_matrix.shape
    z_prob = prob_matrix[row_length - 1, col_length - 1] + prob_matrix[row_length - 2, col_length - 1]
    print_p(z_prob)
