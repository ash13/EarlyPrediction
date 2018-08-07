import numpy as np
import tf_network as tfnet
from tf_network import Network, Cost, Normalization, Optimizer
import tensorflow as tf

WHEELSPIN = 0
STOPOUT  = 1

def run_sample():
    predictive_metric = WHEELSPIN # 0 WHEELSPIN, 1 STOPOUT
    # we will evaluate with a 5 fold cross validation
    n_folds = 5

    # once saved, we can ignore everything up
    seq = dict()
    seq['x'] = np.load('seq_x_sample.npy')
    seq['y'] = np.load('seq_y_sample.npy')
    seq['key'] = np.load('seq_k_sample.npy')

    seq['y'] = tfnet.extract_from_multi_label(seq['y'], predictive_metric) 

    # we can get the number of input nodes by looking at our formatted data
    n_cov = len(seq['x'][0][0])

    # now for model training - we use a for loop for the cross validation
    for i in range(n_folds):
        tf.reset_default_graph()
        tf.set_random_seed(1)
        np.random.seed(1)

        """
        BEGIN MODEL BUILDING
        """
        # split the data into a training and test set using the already-folded key values
        training = np.argwhere(np.array(seq['key'][:, 0], dtype=np.int32) != i).ravel()
        test_set = np.argwhere(np.array(seq['key'][:, 0], dtype=np.int32) == i).ravel()
        
        # then we build the network
        net = Network().add_input_layer(n_cov, normalization=Normalization.Z_SCORE)
        net.add_lstm_layer(64, activation=tf.nn.leaky_relu)
        net.add_lstm_layer(32, activation=tf.nn.leaky_relu)
        net.add_lstm_layer(16, activation=tf.nn.leaky_relu)
        net.add_lstm_layer(4, activation=tf.nn.leaky_relu)

        # layers between the begin/end multi output functions are created at the same level
        # and correspond with each of the label sets we've defined
        net.begin_multi_output(cost_methods=[tf.nn.softmax_cross_entropy_with_logits_v2])
        net.add_dropout_layer(1, keep=0.5, activation=tf.nn.sigmoid)
        net.end_multi_output()

        """
        END MODEL BUILDING
        """

        # default cost method is used if not defined in the begin_multi_output
        net.set_default_cost_method(Cost.RMSE)

        # the optimizer defines how training updates are applied
        net.set_optimizer(Optimizer.ADAM)

        """
        BEGIN MODEL TRAINING
        """
        net.train(x=seq['x'][training],
                  y=seq['y'][training],
                  step=0.001,
                  use_validation=True,
                  max_epochs=4, threshold=0.0, batch=4)
        """
        END MODEL TRAINING
        """
        """ 
        BEGIN MODEL TESTING
        """

        train=seq['x'][training]
        split_train = np.array_split(train, 100)
        pred_train = np.array([])
        for sample in split_train:
            r = net.predict(x=sample, layer_index=4)
            if pred_train.size == 0:
                pred_train = np.array(r)
            else:
                pred_train = np.concatenate((pred_train, np.array(r)), axis=1)
        pred_train = np.array(pred_train)
        pred = net.predict(x=seq['x'][test_set], layer_index=4)

        if predictive_metric == WHEELSPIN:
            prefix = "w"
        else if predictive_metric == STOPOUT:
            prefix = "s"
        name_test = prefix + "_test_set" + str(i) +".npy"
        name_training = prefix + "_training" + str(i) +".npy"
        name_pred_train = prefix + "_pred_train" + str(i) +".npy"
        name_pred = prefix + "_pred" + str(i) + ".npy"

        np.save(name_test, test_set)
        np.save(name_training, training)
        np.save(name_pred_train, pred_train)
        np.save(name_pred, pred)


if __name__ == "__main__":
    run_sample()



