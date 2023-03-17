'''
run the training model

'''
import time
import os
import numpy as np
import pandas as pd
from tensorflow import keras, nn
from keras import regularizers
from neo_tracklet_classifier.datapipeline import DataPipeline
from neo_tracklet_classifier import directory
import beepy
import argparse

TOTAL_SAMPLES = 5997886

# Create an argument parser
parser = argparse.ArgumentParser()


# Add arguments for the variables
parser.add_argument('-f','--file', type=str, required=True, help='Filename suffix for model, history and results data to be saved under e.g. model_<filename>, history_<filename>, results_<filename>')
parser.add_argument('-l1','--layer1', type=int, default=512, help='Number of neurons to be used in first layer. default 64')
parser.add_argument('-l2','--layer2', type=int, default=1024, help='Number of neurons to be used in second layer. default 32')
parser.add_argument('-r','--L2reg', type=float, default=0.01, help='L2 regularization strength. default 0.0')
parser.add_argument('-s','--sample_size', type=int, default=100000, choices=range(50000, TOTAL_SAMPLES), help='Number of samples to be used. Samples should be in the range of 50k - 5997886. This data set is unblanaced so be sure to choose a sample size that includes enough NEOs. default 100000')
parser.add_argument('-t','--training_size', type=float, default=0.9, help='a float to determine the percentage of data to be used for training. default 0.9')
parser.add_argument('-d','--deviation', type=float, default=0.05, help='float that represents the maximum deviation ratio of NEOs in training, validation and test sets. We want validation and test sets to have the same number of NEOs as the training set. default <0.5%')
parser.add_argument('-b','--batch_size', type=int, default=1024, help='Batch size for training. default 1024')
parser.add_argument('-p','--patience', type=int, default=15, help='Number of epochs without improvement to stop training. default 15')
parser.add_argument('-e','--epochs', type=int, default=1000, help='Number of epochs to train. default 1000')
parser.add_argument('-D','--dropout', type=float, default=0.1, help='dropout rate for the model. default 0.0')

# Parse the command line arguments
args = parser.parse_args()


sub_sample_size = args.sample_size
training_size = sub_sample_size * args.training_size
deviation = args.deviation
batch_size = args.batch_size
patience = args.patience
epochs = args.epochs
nn_1layer = args.layer1
nn_2layer = args.layer2
dropout = args.dropout
l2_reg = args.L2reg

def main():
    """
    The main function loads data, trains a model and saves the results of the
    fitting process for a multi-layer perceptron
    """
    
    start = time.perf_counter()
    filename = args.file

    mlp_weights_path = os.path.join(directory.data_dir, 'weights')
    model_path = os.path.join(directory.data_dir, 'model_'+filename)
    history_path = os.path.join(directory.data_dir, 'history_'+filename)
    results_path = os.path.join(directory.data_dir, 'results_'+filename)

    # load data
    data = DataPipeline(
        TOTAL_SAMPLES, sub_sample_size, training_size, batch_size, deviation, ragged=False)

    # find the initial bias and weights for the imblanaced dataset create a new
    # model and load the weights. Fit the data and save the progress.
    initial_bias, class_weights = weights_bias(data.train_labels)

    initial_weights_from_bias(mlp_weights_path, data.train_ds, initial_bias)

    model = define_model()
    model.load_weights(mlp_weights_path)

    history = model.fit(data.train_ds,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=data.val_ds,
                        callbacks=callbacks(),
                        verbose=1,
                        class_weight=class_weights)

    model.save(model_path)
    np.save(history_path, history.history)
    
    # use the new trained model on test data to see how it did.
    train_predictions_logits = model.predict(data.train_df, batch_size=batch_size, verbose=1)
    test_predictions_logits = model.predict(data.test_df, batch_size=batch_size, verbose=1)
    
    train_predictions_bool = np.where(train_predictions_logits >= 0.5, 1, 0)
    test_predictions_bool = np.where(test_predictions_logits >= 0.5, 1, 0)

    p = pd.DataFrame()
    p['class_label'] = data.test_labels.astype(int)
    p['predictions'] = test_predictions_bool
    
    # add 4 columns to denote if the prediction is one of four possible outcomes
    # if class label is 1 and prediction is 0 it's a false positive; if 1 > 0 -> fp
    p['fp'] = (p['class_label'].to_numpy() > p['predictions'].to_numpy()).astype(int)
    
    # if class label is 0 and predition is 1 then it's a false negative; if 0 < 1 -> fn
    p['fn'] = (p['class_label'].to_numpy() < p['predictions'].to_numpy()).astype(int)
    
    # if class label is 1 and predition is 1 then it's true positive; 1 AND 1 -> tp
    p['tp'] = p['class_label'].to_numpy() & p['predictions'].to_numpy()
    
    # if class label is 0 and predition is 0 then it's a true negative; 0 OR 0 XOR 1 -> tn
    #             OR                       XOR
    # class label | prediction | Q1     Q1 | 1 | Q2
    # ------------------------------    -------------
    #     0              0       0       0   1   1    <----true negative
    #     0              1       1       1   1   0
    #     1              0       1       1   1   0
    #     1              1       1       1   1   0
    p['tn'] = (p['class_label'].to_numpy() | p['predictions'].to_numpy()) ^ 1

    #create a dictionary with our results and save it to the disk for viewing in results.ipynb
    test = data.test
    predictions = pd.merge(p, test, left_index=True, right_index=True)

    results = {'train_predictions_bool': train_predictions_bool,
               'train_predictions_logits': train_predictions_logits,
               'test_predictions_bool': test_predictions_bool,
               'test_predictions_logits': test_predictions_logits,
               'train_labels': data.train_labels,
               'test_labels': data.test_labels,
               'predictions': predictions}

    np.save(results_path, results)

    # display the total counts, time elapsed and ding when done.
    counts(data.train_labels, data.test_labels)
    end = time.perf_counter()
    i, d = divmod((end-start)/60, 1)
    print(f"\nElapsed time {i:.0f} min {d*60:.4f} secs\n\n")

def callbacks():
    """
    Define callbacks

    Returns:
        list : list of callbacks
    """
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_prc',
                                                   verbose=10,
                                                   patience=patience,
                                                   mode='max',
                                                   restore_best_weights=True)

    return [early_stopping]


def weights_bias(data):
    """
    Find the initial bias and weights to use for an unbalanced dataset
    
    initial bias = log(count of positive class / count of negative class)
    
    weight for positive class = (1 / count of positive class)  * (total count / 2)
    
    weight for negative class = (1 / count of negative class)  * (total count / 2)

    Args:
        data Dataframe: training data Dataframe

    Returns:
        float, dict: the initial bias, the weights for the positive and negative class
    """
    _, counts = np.unique(data, return_counts=True)
    neg, pos = counts[0], counts[1]
    total = neg+pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    initial_bias = np.log([pos/neg])
    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}\n\n'.format(weight_for_1))

    return initial_bias, class_weight


def define_model(path=None, output_bias=None):
    """
    define the model including the bias, metrics, learning rate schedule
    layers, optimizer and cost function

    Args:
        path (string, optional): path to load weights. Defaults to None.
        output_bias (float, optional): the initial bias found from weights_bias. Defaults to None.

    Returns:
        tf.keras.Model: the keras model
    """
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    metrics = [keras.metrics.TruePositives(name='tp'),
               keras.metrics.FalsePositives(name='fp'),
               keras.metrics.TrueNegatives(name='tn'),
               keras.metrics.FalseNegatives(name='fn'),
               keras.metrics.BinaryAccuracy(name='accuracy'),
               keras.metrics.Precision(name='precision'),
               keras.metrics.Recall(name='recall'),
               keras.metrics.AUC(name='auc'),
               keras.metrics.AUC(name='prc', curve='PR')]  # precision-recall curve
    

    # 76 dimensions plus 1 labels column is the default from tracklet 
    # standardizer.  Datapipeline strips away unneeded categorical
    # columns and sanitizes data to produce a default output for this 
    # model specifically.

    model = keras.Sequential([keras.layers.Dense(nn_1layer, activation=nn.relu, kernel_regularizer=regularizers.l2(l2_reg), input_shape=(77,)),
                              keras.layers.Dropout(dropout),
                              keras.layers.Dense(nn_2layer, activation=nn.relu, kernel_regularizer=regularizers.l2(l2_reg)),
                              keras.layers.Dropout(dropout),
                              keras.layers.Dense(1, activation=nn.sigmoid, bias_initializer=output_bias)])

    # why did you choose adam and bce or what they do
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=metrics)

    if path is not None:
        model.load_weights(path).expect_partial()

    model.summary()
    
    return model


def initial_weights_from_bias(path, train_data, bias):
    """
    Takes in the initial bias from weights_bias and loads creates a model 
    with the initial bias.  Runs a prediction and evaluate to find 
    initial weights.  Saves weights to be used later in training.

    Args:
        path (string): path to save weights
        train_data (Dataframe): the training data
        bias (float): intial bias from weights_bias
    """
    model = define_model(output_bias=bias)

    model.predict(train_data, batch_size=batch_size, verbose=1)
    results = model.evaluate(
        train_data, batch_size=batch_size, verbose=1)
    model.save_weights(path)

    print("\nLoss: {:0.4f}\n".format(results[0]))


def counts(train, test):
    """
    Outputs the total counts

    Args:
        train (Dataframe): the training data
        test (Dataframe): the test data
    """
    print('\n\nTraining')
    vals, counts = np.unique(train, return_counts=True)
    neg, pos = counts[0], counts[1]
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    print('Test')
    val_vals, val_counts = np.unique(test, return_counts=True)
    val_neg, val_pos = val_counts[0], val_counts[1]
    val_total = val_neg + val_pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        val_total, val_pos, 100 * val_pos / val_total))


if __name__ == "__main__":
    main()
    # Beep when done
    beepy.beep(sound=6)
