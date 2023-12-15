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
    #message= re.sub(r'[^\w\s]','', message)
    words = message.split()
    for i in range(len(words)):
        words[i] = words[i].lower()


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
    dictionary = dict()
    Dictionary = dict()

    

    for i in range(len(messages)):
        words = get_words(messages[i])
        Sets = list(set(words))
        for word in Sets:
            if word in dictionary.keys():
                dictionary[word] = dictionary.get(word,0) + 1
            else:
                dictionary[word] = 1
    
   

    for key in dictionary.keys():
        if (dictionary[key] >= 5):
            Dictionary[key] = dictionary[key]
    return Dictionary
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
    Row = len(messages)
    Col = len(word_dictionary)
    shape = (Row, Col)
    word_list = list(word_dictionary.keys())
    Map = np.zeros(shape)
    for i in range(Row):
        words = get_words(messages[i])
        for word in words:
            if word in word_list:
                j = word_list.index(word)
                Map[i][j] = Map[i][j]+1
        
    return Map
    # *** END CODE HERE ***


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
    V = np.shape(matrix)[1]
    D = np.shape(matrix)[0]

    phi_y = np.mean(labels == 1)

    k_0 = matrix[labels == 0]
    k_1 = matrix[labels == 1]


    N_0 = np.sum(k_0, axis = 0)
    N_1 = np.sum(k_1, axis = 0)
    D_0 = np.sum(k_0)*D
    D_1 = np.sum(k_1)*D

    phi_k_0 = (N_0 + 1) / (D_0 + V)
    phi_k_1 = (N_1 + 1) / (D_1 + V)

    trained_model = list()

    trained_model.append(phi_y)
    trained_model.append(phi_k_0)
    trained_model.append(phi_k_1)

    return trained_model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    p_0 = matrix * np.log(model[1])
    p_1 = matrix * np.log(model[2])
    prob_0 = np.log(1 - model[0]) + np.sum(p_0, axis = 1)
    prob_1 = np.log(model[0]) + np.sum(p_1, axis = 1) 
   
    prediction = (prob_1 > prob_0)
    return prediction
  

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
    #Metric: log(p(x|y = 1)) - log(p(x|y=0))
    M = np.log(model[2]) - np.log(model[1])
    key_list = list(dictionary.keys())

    top_list = list()
    for i in range(len(key_list)):
        top_list.append((key_list[i], M[i]))
    
    top_list.sort(key=lambda x: -x[1])
    
    top_five = list()
    for word, val in top_list[:5]:
        top_five.append(word)
    
    
    return top_five
    
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
    radius_list = dict()

    for i in range(len(radius_to_consider)):
        prediction = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius_to_consider[i])
        svm_accuracy = np.mean(prediction == val_labels)
        r = str(radius_to_consider[i])
        radius_list[r] = svm_accuracy
    best_radius = max(radius_list, key=radius_list.get)

    return float(best_radius)
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')
    #print(train_messages)
    #print(train_labels.shape)
    dictionary = create_dictionary(train_messages)
    #print(dictionary)
    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)
    train_mat = train_matrix[:100,:]
    #print(train_matrix.shape)
    np.savetxt('spam_sample_train_matrix', train_mat)

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
