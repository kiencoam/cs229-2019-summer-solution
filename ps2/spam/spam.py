import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = message.split(' ')
    for word in words:
        word = word.lower()
    return words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    dictionary = {}
    counter = {}
    for message in messages:
        words = tuple(get_words(message))
        for word in words:
            if word not in counter:
                counter[word] = 1
            else:
                counter[word] += 1
    for word, count in counter.items():
        if count >= 5:
            dictionary[word] = len(dictionary)
    return dictionary
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    matrix = np.zeros((len(messages), len(word_dictionary)))
    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                matrix[i, word_dictionary[word]] += 1
    return matrix
    # *** END CODE HERE ***

class NaiveBayesModel:
    def __init__(self, phi0, phi1, phi_y):
        self.phi0 = phi0
        self.phi1 = phi1
        self.phi_y = phi_y

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    # phi0[i] Be estimated by proportion of word i in the number of non-spam massage's words
    phi0 = np.ones(matrix.shape[1])

    # phi1[i] Be estimated by proportion of word i in the number of spam massage's words
    phi1 = np.ones(matrix.shape[1])

    positive_count = 0
    positive_word_count = 0
    negative_word_count = 0
    for i in range(matrix.shape[0]):
        if labels[i] == 1:
            positive_count += 1
            phi1 += matrix[i]
            positive_word_count += np.sum(matrix[i])
        else:
            phi0 += matrix[i]
            negative_word_count += np.sum(matrix[i])

    phi_y = positive_count / labels.shape[0]
    phi1 /= matrix.shape[1] + positive_word_count
    phi0 /= matrix.shape[1] + negative_word_count

    return NaiveBayesModel(phi0, phi1, phi_y)
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array contain the predictions from the model
    """
    # *** START CODE HERE ***
    pred_labels = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        threshold = np.log(model.phi_y) - np.log(1 - model.phi_y)
        for j in range(matrix.shape[1]):
            threshold += matrix[i, j] * (np.log(model.phi1[j]) - np.log(model.phi0[j]))
        if threshold >= 0:
            pred_labels[i] = 1
    return pred_labels
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    top_five_words = []
    indices = list(range(len(dictionary)))
    indices.sort(key=lambda x: np.log(model.phi1[x]) - np.log(model.phi0[x]), reverse=True)
    for i in range(5):
        for key, value in dictionary.items():
            if value == indices[i]:
                top_five_words.append(key)
                break
    return top_five_words
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    highest_accuracy = 0
    best_radius = 0
    for radius in radius_to_consider:
        pred_labels = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius=radius)
        accuracy = np.mean(pred_labels == val_labels)
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_radius = radius
    return best_radius
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
