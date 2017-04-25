import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import math
import json

import tensorflow as tf

from lstm_common import *
from sklearn import metrics
from sklearn.base import BaseEstimator

NUM_LAYERS = 2
LSTM_HIDDEN_DIM = 60
LEMMA_DIM = 50
POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1

EMPTY_PATH = ((0, 0, 0, 0),)
MAX_PATH_LEN = 6
BATCH_SIZE = 10
UNK_INDEX = 0

LSTM_OUTPUT_DIM = LSTM_HIDDEN_DIM
LSTM_INPUT_DIM = LEMMA_DIM + POS_DIM + DEP_DIM + DIR_DIM


class PathLSTMClassifier(BaseEstimator):

    def __init__(self, num_lemmas, num_pos, num_dep, num_directions=5, n_epochs=10, num_relations=2,
                 lemma_embeddings=None, dropout=0.0, num_hidden_layers=0):
        """'
        Initialize the LSTM
        :param num_lemmas Number of distinct lemmas in the paths + words in the (x, y) pairs
        :param num_pos Number of distinct part of speech tags
        :param num_dep Number of distinct depenedency labels
        :param num_directions Number of distinct path directions (e.g. >,<)
        :param n_epochs Number of training epochs
        :param num_relations Number of classes (e.g. binary = 2)
        :param lemma_embeddings Pre-trained word embedding vectors for the path-based component
        :param dropout Dropout rate
        :param num_hidden_layers The number of hidden layers for the term-pair classification network
        """
        self.n_epochs = n_epochs
        self.num_lemmas = num_lemmas
        self.num_pos = num_pos
        self.num_dep = num_dep
        self.num_directions = num_directions
        self.num_relations = num_relations
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers

        self.lemma_vectors = None
        if lemma_embeddings is not None:
            self.lemma_vectors = lemma_embeddings

        # Create the network
        self.model_parameters = create_computation_graph(self.num_lemmas, self.num_pos, self.num_dep, self.num_directions,
                                                         self.num_relations, self.lemma_vectors, self.num_hidden_layers)

        self.session = tf.Session()

    @classmethod
    def load_model(cls, model_file_prefix):
        """
        Load the trained model from a file
        :param model_file_prefix the path + file name (no extension) where the model files are saved
        """

        # Load the parameters from the json file
        with open(model_file_prefix + '.params') as f_in:
            params = json.load(f_in)

        classifier = PathLSTMClassifier(params['num_lemmas'], params['num_pos'], params['num_dep'],
                                        params['num_directions'], num_relations=params['num_relations'],
                                        num_hidden_layers=params['num_hidden_layers'])

        # Initialize the session and start training
        classifier.session.run(tf.global_variables_initializer())

        # Load the model
        tf.train.Saver().restore(classifier.session, model_file_prefix)

        # Get the variables
        variable_names = ['W1', 'b1', 'lemma_lookup', 'pos_lookup', 'dep_lookup', 'dir_lookup']

        if classifier.num_hidden_layers == 1:
            variable_names += ['W2', 'b2']

        classifier.model_parameters.update({ v.name : v for v in tf.global_variables() })

        # Load the dictionaries from the json file
        with open(model_file_prefix + '.dict') as f_in:
            dictionaries = json.load(f_in)

        word_index, pos_index, dep_index, dir_index = dictionaries

        return classifier, word_index, pos_index, dep_index, dir_index

    def save_model(self, output_prefix, dictionaries):
        """
        Save the trained model to a file
        :param output_prefix Where to save the model
        :param dictionaries hyper-parameters to save
        """
        tf.train.Saver().save(self.session, output_prefix)

        # Save the model hyper-parameters
        params = { 'num_relations' : self.num_relations, 'num_hidden_layers' : self.num_hidden_layers,
                   'num_lemmas' : self.num_lemmas, 'num_pos' : self.num_pos, 'num_directions' : self.num_directions,
                   'num_dep' : self.num_dep }

        with open(output_prefix + '.params', 'w') as f_out:
            json.dump(params, f_out, indent=2)

        # Save the dictionaries
        with open(output_prefix + '.dict', 'w') as f_out:
            json.dump(dictionaries, f_out, indent=2)

    def close(self):
        """
        Close the session
        """
        self.session.close()
        tf.reset_default_graph()

    def fit(self, X_train, y_train, x_y_vectors=None):
        """
        Train the model
        :param X_train the train instances (paths)
        :param y_train the train labels
        :param x_y_vectors the train (x, y) vector indices
        """
        print 'Training the model...'
        train(self.session, self.model_parameters, X_train, y_train, self.n_epochs, self.num_relations, self.num_lemmas,
              self.num_pos, self.num_dep, self.num_directions, x_y_vectors, self.dropout)
        print 'Done!'

    def predict(self, X_test, x_y_vectors=None):
        """
        Predict the classification of the test set
        """
        predictions, scores = zip(*self.predict_with_score(X_test, x_y_vectors))
        return np.array(predictions)

    def predict_with_score(self, X_test, x_y_vectors=None):
        """
        Predict the classification of the test set
        :param X_test the test instances (paths)
        :param x_y_vectors the test (x, y) vector indices
        """
        model_parameters = self.model_parameters

        # Define the neural network model (predict every 100 instances together)
        batch_paths = model_parameters['batch_paths']
        seq_lengths = model_parameters['seq_lengths']
        num_batch_paths = model_parameters['num_batch_paths']
        path_lists = model_parameters['path_lists']
        path_counts = model_parameters['path_counts']
        x_vector_inputs = model_parameters['x_vector_inputs']
        y_vector_inputs = model_parameters['y_vector_inputs']
        predictions = model_parameters['predictions']

        # Sort the pairs by number of paths, and add the empty path to pairs with no paths
        num_paths = np.array([len(instance) for instance in X_test])
        sorted_indices = np.argsort(num_paths)
        x_y_vectors = [x_y_vectors[i] for i in sorted_indices]
        X_test = [X_test[i] if len(X_test[i]) > 0 else { EMPTY_PATH : 1 } for i in sorted_indices]

        pad = lambda lst : lst if len(lst) == BATCH_SIZE else lst + [0] * (BATCH_SIZE - len(lst))
        test_pred = [0] * (len(sorted_indices))

        for chunk in xrange(0, len(X_test), BATCH_SIZE):

            # Initialize the variables with the current batch data
            batch_indices = list(range(chunk, min(chunk + BATCH_SIZE, len(X_test))))
            actual_batch_size = len(batch_indices)
            batch_indices = pad(batch_indices)

            curr_batch_paths, curr_path_lists, curr_path_counts, curr_labels, x_vectors, y_vectors,\
            curr_seq_lengths = prepare_batch(x_y_vectors, X_test, batch_indices, self.num_relations)

            curr_predictions = self.session.run(predictions, feed_dict={ batch_paths : curr_batch_paths,
                                                          num_batch_paths : curr_batch_paths.shape[0],
                                                          seq_lengths : curr_seq_lengths,
                                                          path_lists : curr_path_lists,
                                                          path_counts : curr_path_counts,
                                                          x_vector_inputs : x_vectors,
                                                          y_vector_inputs : y_vectors })

            for index_in_batch, index_in_dataset in enumerate(batch_indices[:actual_batch_size]):
                vec = curr_predictions[index_in_batch]
                test_pred[sorted_indices[index_in_dataset]] = (np.argmax(vec), vec[np.argmax(vec)])

        return test_pred


def mlp_model(model_parameters, path_embeddings, num_relations, num_hidden_layers=0):
    """
    Defines the MLP operations
    :param model_parameters: the network parameters
    :param path_embeddings: the matrix of paths variable computed by the LSTM
    :param num_relations: the number of classes in the output layer
    :param num_hidden_layers: the number of hidden layers (supports 0 and 1)
    :return: the prediction object to be computed in a Session
    """

    lemma_lookup = model_parameters['lemma_lookup']

    W1 = model_parameters['W1']
    b1 = model_parameters['b1']
    W2 = None
    b2 = None

    if num_hidden_layers == 1:
        W2 = model_parameters['W2']
        b2 = model_parameters['b2']

    # Define the place holders
    path_lists = tf.placeholder(tf.int32, (BATCH_SIZE, None)) # list of paths for each item in the batch
    path_counts = tf.placeholder(tf.int32, (BATCH_SIZE, None)) # list of path counts for each item in the batch
    x_vector_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    y_vector_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, num_relations])

    # Define the operations
    num_paths = tf.reduce_sum(tf.cast(path_counts, tf.float32), 1) # number of paths for each pair [BATCH_SIZE, 1]
    curr_path_embeddings = [tf.squeeze(tf.gather(path_embeddings, path_list))
                            for path_list in tf.split(path_lists, BATCH_SIZE, axis=0)] # a list of [MAX_PATHS, 60]

    path_counts_lst = tf.split(path_counts, BATCH_SIZE, axis=0) # a list of [MAX_PATHS, 1]
    path_counts_tiled = [tf.transpose(tf.tile(tf.cast(path_count, tf.float32), tf.stack([LSTM_HIDDEN_DIM, 1])))
                         for path_count in path_counts_lst] # a list of [MAX_PATHS, 60]

    weighted = [tf.multiply(curr_path_embedding, path_count) for (curr_path_embedding, path_count)
                in zip(curr_path_embeddings, path_counts_tiled)]

    weighted_sum = [tf.reduce_sum(weighted[i], 0) for i in range(BATCH_SIZE)]

    pair_path_embeddings = tf.stack([tf.div(weighted_sum_item, num_paths_item)
                                     for weighted_sum_item, num_paths_item in zip(weighted_sum, tf.unstack(num_paths))])

    # Concatenate the path embedding to the word embeddings and feed it to the MLP
    x_vectors = tf.nn.embedding_lookup(lemma_lookup, x_vector_inputs)
    y_vectors = tf.nn.embedding_lookup(lemma_lookup, y_vector_inputs)
    network_input = tf.concat([x_vectors, pair_path_embeddings, y_vectors], 1)
    h = tf.add(tf.matmul(network_input, W1), b1)
    output = h

    if num_hidden_layers == 1:
        output = tf.add(tf.matmul(tf.nn.tanh(h), W2), b2)

    predictions = tf.nn.softmax(output)
    return path_lists, path_counts, x_vector_inputs, y_vector_inputs, predictions, output, labels


def lstm_model(model_parameters):
    """
    Defines the LSTM operations
    :param model_parameters: the network parameters
    :return: a matrix of path embeddings
    """

    lemma_lookup = model_parameters['lemma_lookup']
    pos_lookup = model_parameters['pos_lookup']
    dep_lookup = model_parameters['dep_lookup']
    dir_lookup = model_parameters['dir_lookup']

    # Define the place holders
    batch_paths = tf.placeholder(tf.int32, shape=[None, MAX_PATH_LEN, 4]) # the paths to compute in this batch
    seq_lengths = tf.placeholder(tf.int32, shape=[None]) # the length of each path
    num_batch_paths = tf.placeholder(tf.int32)

    lookup_tables = [lemma_lookup, pos_lookup, dep_lookup, dir_lookup]

    edges = tf.split(batch_paths, MAX_PATH_LEN, axis=1)
    edge_components = [tf.split(edge, 4, axis=2) for edge in edges]

    path_matrix = [tf.concat([tf.nn.embedding_lookup(lookup_table, component)
                    for lookup_table, component in zip(lookup_tables, edge)], -1)
                   for edge in edge_components]

    path_matrix = [tf.concat(lst, -1) for lst in path_matrix]
    path_matrix = tf.squeeze(tf.stack(path_matrix, 0))
    path_matrix = tf.reshape(path_matrix, tf.stack([num_batch_paths, MAX_PATH_LEN, LSTM_INPUT_DIM]))

    # Define the operations
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN_DIM)
    initial_state = lstm_cell.zero_state(num_batch_paths, tf.float32)
    lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_cell, path_matrix, initial_state=initial_state, sequence_length=seq_lengths)

    # Get the last output from each item in the batch
    path_embeddings = extract_last_relevant(lstm_outputs, num_batch_paths, seq_lengths)

    return batch_paths, seq_lengths, path_embeddings, num_batch_paths


def extract_last_relevant(data, dim1, length):
    """
    From: https://danijar.com/variable-sequence-lengths-in-tensorflow/
    Get specified elements along the second axis of a tensor
    :param data: tensor to be subsetted
    :param dim1: the size of dimension 1
    :param ind: indices to take (one for each element along axis 1 of data)
    :return: Subsetted tensor
    """
    out_size = int(data.get_shape()[2])
    index = tf.range(0, dim1) * MAX_PATH_LEN + (length - 1)
    flat = tf.reshape(data, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


def train(session, model_parameters, X_train, y_train, nepochs, num_relations, num_lemmas, num_pos, num_dep, num_dir,
          x_y_vectors=None, dropout=0.0):
    """
    Train the LSTM
    :param model_parameters: the model parameters
    :param X_train: the train instances
    :param y_train: the train labels
    :param nepochs: number of epochs
    :param num_relations: the number of possible output classes
    :param num_lemmas Number of distinct lemmas in the paths + words in the (x, y) pairs
    :param num_pos Number of distinct part of speech tags
    :param num_dep Number of distinct depenedency labels
    :param num_directions Number of distinct path directions (e.g. >,<)
    :param x_y_vectors: the word vectors of x and y
    :param dropout The word dropout rate
    """

    # Define the batches
    n_batches = int(math.ceil(len(y_train) / BATCH_SIZE))

    # Define the neural network model
    batch_paths = model_parameters['batch_paths']
    seq_lengths = model_parameters['seq_lengths']
    num_batch_paths = model_parameters['num_batch_paths']
    path_lists = model_parameters['path_lists']
    path_counts = model_parameters['path_counts']
    x_vector_inputs = model_parameters['x_vector_inputs']
    y_vector_inputs = model_parameters['y_vector_inputs']
    predictions = model_parameters['predictions']
    labels = model_parameters['labels']

    # Define the loss function and the optimization algorithm
    loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss_fn)

    # Initialize the session and start training
    session.run(tf.global_variables_initializer())

    # Apply dropout on every component of every path
    print 'Applying dropout...'
    dropouts = []
    for num in [num_lemmas, num_pos, num_dep, num_dir]:
        mask = np.random.binomial(1, dropout, num)
        dropouts.append(set([i for i, value in enumerate(mask) if value == 1]))

    X_train = [instance if len(instance) > 0 else { EMPTY_PATH : 1 } for instance in X_train]
    X_train = [{ tuple([tuple([component if component not in dropouts[comp_num] else UNK_INDEX
                               for comp_num, component in enumerate(edge)]) for edge in path]) : count
                for path, count in instance.iteritems() } for instance in X_train]

    print 'Training...'

    # Sort the pairs by number of paths, and add the empty path to pairs with no paths
    num_paths = np.array([len(instance) for instance in X_train])
    sorted_indices = np.argsort(num_paths)
    X_train = [X_train[i] for i in sorted_indices]
    y_train = [y_train[i] for i in sorted_indices]
    x_y_vectors = [x_y_vectors[i] for i in sorted_indices]

    for epoch in range(nepochs):

        epoch_loss = 0.0
        epoch_indices = list(range(len(y_train)))
        y_pred = np.zeros(len(y_train))

        for minibatch in range(n_batches):

            batch_indices = epoch_indices[minibatch * BATCH_SIZE:(minibatch + 1) * BATCH_SIZE]

            # Compute each path in the batch once, create a matrix of path embeddings, and average for each word-pair
            curr_batch_paths, curr_path_lists, curr_path_counts, curr_labels, x_vectors, y_vectors, curr_seq_lengths \
                = prepare_batch(x_y_vectors, X_train, batch_indices, num_relations, labels=y_train)

            _, curr_loss, curr_predictions = session.run([optimizer, loss_fn, predictions],
                                                      feed_dict={ batch_paths : curr_batch_paths, # distinct paths in the batch
                                                             num_batch_paths : curr_batch_paths.shape[0],
                                                             seq_lengths : curr_seq_lengths, # the length of each path
                                                             path_lists : curr_path_lists, # paths for each pair
                                                             path_counts : curr_path_counts, # count for each path
                                                             labels : curr_labels,
                                                             x_vector_inputs : x_vectors,
                                                             y_vector_inputs : y_vectors })

            epoch_loss += curr_loss
            curr_predictions = np.argmax(curr_predictions, 1)
            for i in range(len(batch_indices)):
                y_pred[batch_indices[i]] = curr_predictions[i]

        epoch_loss /= len(y_train)
        precision, recall, f1, support = metrics.precision_recall_fscore_support(y_train, y_pred, average='weighted')
        print 'Epoch: %d/%d, Loss: %f, Precision: %.3f, Recall: %.3f, F1: %.3f' % \
              (epoch + 1, nepochs, epoch_loss, precision, recall, f1)

    return session


def prepare_batch(x_y_vectors, instances, batch_indices, num_relations, labels=None):
    """
    Populate the variables for the current batch
    :param x_y_vectors: the word vectors of x and y
    :param instances: the train instances
    :param batch_indices: the indices from the train set to use in the current batch
    :param num_relations: the number of possible output classes
    :param labels: the train labels
    :return:
    """
    batch_size = len(batch_indices)

    # Get all the distinct paths in the batch
    batch = [instances[batch_indices[i]] for i in range(batch_size)]
    index_to_path = list(set([path for instance in batch for path in instance]))
    path_to_index = { path : i for i, path in enumerate(index_to_path) }

    batch_paths = np.stack([np.vstack(pad_path(path)) for path in index_to_path])
    seq_lengths = np.array([len(path) for path in path_to_index])

    # Get the paths for each instance
    max_path_per_ins = max([len(instances[batch_indices[i]]) for i in range(batch_size)])
    pad = lambda lst : lst[:max_path_per_ins] if len(lst) >= max_path_per_ins \
        else lst + [0] * (max_path_per_ins - len(lst))
    curr_path_lists = np.vstack([pad([path_to_index[path] for path in instances[batch_indices[i]]])
                                 for i in range(batch_size)])
    curr_path_counts = np.vstack([pad(instances[batch_indices[i]].values()) for i in range(batch_size)])

    curr_labels = np.zeros((batch_size, num_relations))
    if labels is not None:
        curr_labels_temp = np.array([labels[batch_indices[i]] for i in range(batch_size)])
        curr_labels = np.eye(batch_size, num_relations)[curr_labels_temp]

    x_vectors = np.array([x_y_vectors[batch_indices[i]][0] for i in range(batch_size)])
    y_vectors = np.array([x_y_vectors[batch_indices[i]][1] for i in range(batch_size)])

    return batch_paths, curr_path_lists, curr_path_counts, curr_labels, x_vectors, y_vectors, seq_lengths


def pad_path(path):
    """
    Pad the path with empty edges to make it MAX_PATH_LEN long
    :param path: the original path
    :return: the padded path
    """
    path = list(path)

    if len(path) < MAX_PATH_LEN:
        path += [(0, 0, 0, 0)] * (MAX_PATH_LEN - len(path))

    return [np.array(list(edge)) for edge in path]


def create_computation_graph(num_lemmas, num_pos, num_dep, num_directions, num_relations, wv=None, num_hidden_layers=0):
    """
    Initialize the model
    :param num_lemmas Number of distinct lemmas
    :param num_pos Number of distinct part of speech tags
    :param num_dep Number of distinct depenedency labels
    :param num_directions Number of distinct path directions (e.g. >,<)
    :param num_relations Number of classes (e.g. binary = 2)
    :param wv Pre-trained word embeddings file
    :param num_hidden_layers The number of hidden layers for the term-pair classification network
    :return: the model parameters: LSTM, parameters and lookup tables
    """
    model_parameters = {}
    initializer = tf.contrib.layers.xavier_initializer()

    # Define the MLP
    network_input = LSTM_OUTPUT_DIM + 2 * LEMMA_DIM

    #  'the optimal size of the hidden layer is usually between the size of the input and size of the output layers'
    hidden_dim = int((network_input + num_relations) / 2)

    if num_hidden_layers == 0:

        model_parameters['W1'] = tf.get_variable('W1', shape=[network_input, num_relations], initializer=initializer)
        model_parameters['b1'] = tf.get_variable('b1', shape=[num_relations], initializer=initializer)

    elif num_hidden_layers == 1:

        model_parameters['W1'] = tf.get_variable('W1', shape=[network_input, hidden_dim], initializer=initializer)
        model_parameters['b1'] = tf.get_variable('b1', shape=[hidden_dim], initializer=initializer)

        model_parameters['W2'] = tf.get_variable('W2', shape=[hidden_dim, num_relations], initializer=initializer)
        model_parameters['b2'] = tf.get_variable('b2', shape=[num_relations], initializer=initializer)

    else:
        raise ValueError('Only 0 or 1 hidden layers are supported')

    # Create the embeddings lookup
    if wv != None:
        model_parameters['lemma_lookup'] = tf.Variable(wv, name='lemma_lookup', dtype=tf.float32)
    else:
        model_parameters['lemma_lookup'] = tf.get_variable('lemma_lookup', shape=[num_lemmas, LEMMA_DIM],
                                                           initializer=initializer)

    model_parameters['pos_lookup'] = tf.get_variable('pos_lookup', shape=[num_pos, POS_DIM], initializer=initializer)
    model_parameters['dep_lookup'] = tf.get_variable('dep_lookup', shape=[num_dep, DEP_DIM], initializer=initializer)
    model_parameters['dir_lookup'] = tf.get_variable('dir_lookup', shape=[num_directions, DIR_DIM], initializer=initializer)

    # Define the neural network model
    batch_paths, seq_lengths, path_embeddings, num_batch_paths = lstm_model(model_parameters)
    path_lists, path_counts, x_vector_inputs, y_vector_inputs, predictions, output, labels = \
        mlp_model(model_parameters, path_embeddings, num_relations, num_hidden_layers)
    model_parameters.update({ 'batch_paths' : batch_paths, 'seq_lengths' : seq_lengths,
                              'path_embeddings' : path_embeddings, 'num_batch_paths' : num_batch_paths,
                              'path_lists' : path_lists, 'path_counts' : path_counts,
                              'x_vector_inputs' : x_vector_inputs, 'y_vector_inputs' : y_vector_inputs,
                              'predictions' : predictions, 'output' : output, 'labels' : labels})

    return model_parameters
