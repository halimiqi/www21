from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
# seed = 121   # last random seed is 141           0.703
# #random.seed(seed)
# np.random.seed(seed)
# tf.set_random_seed(seed)

from gcn.utils import *
from gcn.models import GCN, MLP

# # Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
#
# seed = 121   # last random seed is 141           0.703
# #random.seed(seed)
# np.random.seed(seed)
# tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# #flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
# #flags.DEFINE_string('gcn_model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
# flags.DEFINE_float('gcn_learning_rate', 0.01, 'Initial learning rate.')
# #flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
# flags.DEFINE_integer('gcn_hidden1', 16, 'Number of units in hidden layer 1.')
# #flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
# flags.DEFINE_float('gcn_weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
# #flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
def run(dataset,adj,features,y_train,y_val, y_test, train_mask, val_mask, test_mask,  name = "original", model_str = "gcn", epochs = 200, dropout = 0.5, early_stopping = 5,seed = 142):
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        seed = seed
        np.random.seed(seed)
        tf.set_random_seed(seed)
        with tf.variable_scope(name) as scope:
            #_, _, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
            #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)

            # Some preprocessing
            features = preprocess_features(features)
            if model_str == 'gcn':
                support = [preprocess_adj(adj)]
                num_supports = 1
                model_func = GCN
            elif model_str == 'gcn_cheby':
                support = chebyshev_polynomials(adj, FLAGS.max_degree)
                num_supports = 1 + FLAGS.max_degree
                model_func = GCN
            elif model_str == 'dense':
                support = [preprocess_adj(adj)]  # Not used
                num_supports = 1
                model_func = MLP
            else:
                raise ValueError('Invalid argument for model: ' + str(model_str))

            # Define placeholders
            placeholders = {
                'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
                'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
                'labels_mask': tf.placeholder(tf.int32),
                'dropout': tf.placeholder_with_default(0., shape=()),
                'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
            }

            # Create model
            model = model_func(placeholders, input_dim=features[2][1], logging=True, name = name)

            # Initialize session
            sess = tf.Session()


            # Define model evaluation function
            def evaluate(features, support, labels, mask, placeholders):
                t_test = time.time()
                feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
                outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
                return outs_val[0], outs_val[1], (time.time() - t_test)


            # Init variables
            sess.run(tf.global_variables_initializer())

            cost_val = []
            acc_val = 0
            # Train model
            for epoch in range(epochs):

                t = time.time()
                # Construct feed dictionary
                feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
                feed_dict.update({placeholders['dropout']: dropout})

                # Training step
                outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

                # Validation
                cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
                cost_val.append(cost)
                acc_val = acc
                # Print results
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                      "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                      "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

                if epoch > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping+1):-1]):
                    print("Early stopping...")
                    break

            print("Optimization Finished!")

            # Testing
            test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
            print("Test set results:", "cost=", "{:.5f}".format(test_cost),
                  "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    return test_acc, acc_val

if __name__ == "__main__":
    seed = 132
    np.random.seed(seed)
    tf.set_random_seed(seed)
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("citeseer")
    test_acc, _ = run("citeseer", adj)
    test_acc_2, _ = run("citeseer", adj)
    print(test_acc, test_acc_2)
