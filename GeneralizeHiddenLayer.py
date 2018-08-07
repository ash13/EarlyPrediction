import numpy as np
import tf_network as tfnet
import evaluationutility as eu
import tensorflow as tf

def generalizeDense(train, train_label, test, test_label):
    x_train = tfnet.flatten_sequence(train)
    print(x_train.shape)
    y_train = train_label
    x_test = tfnet.flatten_sequence(test)
    y_test = test_label

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64,  activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    print(model.evaluate(x_test, y_test))

    pred = model.predict(x=x_test)
    return pred

def generalizeDecisionTree(train, train_label, test, test_label):    
    from sklearn import tree
    dtc = tree.DecisionTreeClassifier(max_depth=4)
    dtc = dtc.fit(x_train, y_train)
    pred = dtc.predict(x_test)
    score = dtc.score(x_test, y_test)
    
    return pred


def run_sample():
    kappa_w = []
    aucs_w = []
    aucs_s = []
    kappa_s = []

    # we will evaluate with a 5 fold cross validation
    n_folds = 5

    # once saved, we can ignore everything up
    seq = dict()
    seq['x'] = np.load('seq_x_sample.npy')
    seq['y'] = np.load('seq_y_sample.npy')
    seq['key'] = np.load('seq_k_sample.npy')
    stopout = tfnet.extract_from_multi_label(seq['y'], 1)
    wheelspin = tfnet.extract_from_multi_label(seq['y'], 0)

    seq['y'] = tfnet.extract_from_multi_label(seq['y'], 0)

    # we can get the number of input nodes by looking at our formatted data
    desc = tfnet.describe_multi_label(seq['y'], True)

    # now for model training - we use a for loop for the cross validation
    for i in range(n_folds):
        name_test = "w_test_set" + str(i) + ".npy"
        name_training = "w_training" + str(i) + ".npy"
        name_pred_train = "w_pred_train" + str(i) + ".npy"
        name_pred = "w_pred" + str(i) + ".npy"

        test_set = np.load(name_test)
        training = np.load(name_training)
        pred_train = np.load(name_pred_train)
        pred_test = np.load(name_pred)

        # loop through each label set...
        for p in range(len(pred_test)):

            predicted_train = pred_train[0]
            actual_train_w = tfnet.flatten_sequence(tfnet.extract_from_multi_label(wheelspin[training], 0)).ravel()
            actual_train_s = tfnet.flatten_sequence(tfnet.extract_from_multi_label(stopout[training], 0)).ravel()

            predicted_test = pred_test[0]
            actual_w = tfnet.flatten_sequence(tfnet.extract_from_multi_label(wheelspin[test_set], p)).ravel()
            actual_s = tfnet.flatten_sequence(tfnet.extract_from_multi_label(stopout[test_set], p)).ravel()

            predicted_w = generalizeDense(predicted_train, actual_train_w, predicted_test, actual_w) # can change generalization functions here 
            predicted_s = generalizeDense(predicted_train, actual_train_s, predicted_test, actual_s) # can change generalization functions here 


            print("--------Wheelspin--------")

            auc_w = eu.auc(actual_w, predicted_w, average_over_labels=True)
            print("Fold AUC (Label Set {}): {:<.3f}".format(p, auc_w))

            if desc['n_labels'][p] > 1:
                kpa_w = eu.cohen_kappa_multiclass(actual_w, predicted_w)
            else:
                kpa_w = eu.cohen_kappa(actual_w, predicted_w, average_over_labels=True)

            print("Fold KAPPA (Label Set {}): {:<.3f}".format(p, kpa_w))

            print("--------Stopout--------")
            auc_s = eu.auc(actual_s, predicted_s, average_over_labels=True)
            print("Fold AUC (Label Set {}): {:<.3f}".format(p, auc_s))

            if desc['n_labels'][p] > 1:
                kpa_s = eu.cohen_kappa_multiclass(actual_s, predicted_s)
            else:
                kpa_s = eu.cohen_kappa(actual_s, predicted_s, average_over_labels=True)

            print("Fold KAPPA (Label Set {}): {:<.3f}".format(p, kpa_s))

            aucs_w.append(auc_w)
            kappa_w.append(kpa_w)
            aucs_s.append(auc_s)
            kappa_s.append(kpa_s)

            print("")
    print("Wheelspin")
    print("AUC: ", aucs_w)
    print("KPA: ", kappa_w)
    print("Stopout")
    print("AUC: ", aucs_s)
    print("KPA: ", kappa_s)


if __name__ == "__main__":
    run_sample()

