'''
run the training model

'''
import time
import os
from neo_tracklet_classifier import directory, header
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow import data
import beepy


TOTAL_SAMPLES = 5997886
SUB_SAMPLE_SIZE = 5997886
TRAINING_SIZE = SUB_SAMPLE_SIZE * 0.9
DEVIATION = 0.05
PATIENCE = 25
BATCH_SIZE = pow(2, 9)
EPOCHS = 10000
NN_1LAYER = 64
NN_2LAYER = 32


def main():
    start = time.perf_counter()

    '''
    Train a baseline model on the raw training data
    '''
    
    # file paths for saving data and loading the loading 
    # training, validation and test data
    filename = 'balanced_64_32'
    model_path = os.path.join(directory.data_dir, 'model_'+filename)
    history_path = os.path.join(directory.data_dir, 'history_'+filename)
    results_path = os.path.join(directory.data_dir, 'results_'+filename)
    npy_train = os.path.join(directory.data_dir, 'train.npy')
    npy_val = os.path.join(directory.data_dir, 'val.npy')
    npy_test = os.path.join(directory.data_dir, 'test.npy')


    # Load the training validation and test data.  Only the first two 
    # opbservations are used and the features are X, Y, Z, M, 
    # mu_ra, mu_dec and mu_sq. The training data is balanced 
    # [0.5, 0.5] between NEO and Non-NEO samples for the training and
    # validation sets.  The test data is is [0.95, 0.5] Non-NEO and NEO
    # respectively, so that it resembles real data.
    train = pd.DataFrame(np.load(npy_train), columns=header.baseline_header2)
    val = pd.DataFrame(np.load(npy_val), columns=header.baseline_header2)
    test = pd.DataFrame(np.load(npy_test), columns=header.baseline_header2)

    # separate the class labels so we can use them for generating
    # tf.data.Datasets and use them later for predicting
    train_labels = train.pop('NEO')
    val_labels = val.pop('NEO')
    test_labels = test.pop('NEO')

    # create tf.data.Dataset for training and validation.  These 
    # Datasets are batched, shuffled and prefetched to 
    # optimize training time.
    train_ds = df_to_dataset(train, BATCH_SIZE, True, train_labels)
    val_ds = df_to_dataset(val, BATCH_SIZE, True, val_labels)
    
    
    # define the model and pass in Datasets for training
    model = define_model()

    history = model.fit(train_ds,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data = val_ds,
                        callbacks=callbacks(),
                        verbose=1)

    # save the model and history
    model.save(model_path)
    np.save(history_path, history.history)
    
    # run predictions using both the training and test sets.  Return logits and bools
    train_predictions_logits = model.predict(train, batch_size=BATCH_SIZE, verbose=1)
    test_predictions_logits = model.predict(test, batch_size=BATCH_SIZE, verbose=1)
    train_predictions_bool = np.where(train_predictions_logits >= 0.5, 1, 0)
    test_predictions_bool = np.where(test_predictions_logits >= 0.5, 1, 0)


    # Create a pd.Dataframe and store the predictions and also create columns for
    # true positives, false positives, true negative and false negatives.
    p = pd.DataFrame()
    p['class_label'] = test_labels.astype(int)
    p['predictions'] = pd.DataFrame(test_predictions_bool, columns=['predictions'])
    print(p)
    
    p['fp'] = (p['class_label'].to_numpy() >
               p['predictions'].to_numpy()).astype(int)
    p['fn'] = (p['class_label'].to_numpy() <
               p['predictions'].to_numpy()).astype(int)
    p['tp'] = p['class_label'].to_numpy() & p['predictions'].to_numpy()
    p['tn'] = (p['class_label'].to_numpy() | p['predictions'].to_numpy()) ^ 1

    test_copy = test
    test_copy.replace(np.nan, 0, inplace=True)
    test_copy.reset_index(inplace=True)

    predictions = pd.merge(p, test_copy, left_index=True, right_index=True)

    # create a dictionary with all the data and save it off for view results later
    results = {'train_predictions_bool': train_predictions_bool,
               'train_predictions_logits': train_predictions_logits,
               'test_predictions_bool': test_predictions_bool,
               'test_predictions_logits': test_predictions_logits,
               'train_labels': train_labels,
               'test_labels': test_labels,
               'predictions': predictions}

    np.save(results_path, results)

    counts(train_labels, test_labels)

    end = time.perf_counter()
    i, d = divmod((end-start)/60, 1)
    print(f"\nElapsed time {i:.0f} min {d*60:.4f} secs\n\n")


'''
Functions used in this file are below

'''
def callbacks():
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_prc',
                                                   verbose=10,
                                                   patience=PATIENCE,
                                                   mode='max',
                                                   restore_best_weights=True)

    return [early_stopping]

def define_model():
    '''
    define the tensorflow model'''

    metrics = [keras.metrics.TruePositives(name='tp'),
               keras.metrics.FalsePositives(name='fp'),
               keras.metrics.TrueNegatives(name='tn'),
               keras.metrics.FalseNegatives(name='fn'),
               keras.metrics.BinaryAccuracy(name='accuracy'),
               keras.metrics.Precision(name='precision'),
               keras.metrics.Recall(name='recall'),
               keras.metrics.AUC(name='auc'),
               keras.metrics.AUC(name='prc', curve='PR')]  # precision-recall curve
    
    model = keras.Sequential([keras.layers.Dense(NN_1LAYER, activation='relu', input_shape=(18,)),
                              keras.layers.Dense(NN_2LAYER, activation='relu'),
                              keras.layers.Dense(1, activation='sigmoid')])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=metrics)

    return model

def counts(train, test):
    print('\n\nTraining')
    _, counts = np.unique(train, return_counts=True)
    neg, pos = counts[0], counts[1]
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    print('Validation')
    _, test_counts = np.unique(test, return_counts=True)
    test_neg, test_pos = test_counts[0], test_counts[1]
    test_total = test_neg + test_pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        test_total, test_pos, 100 * test_pos / test_total))

def df_to_dataset(df, batch_size, shuffle, labels):
    '''
    creates a tensorflow dataset and optimizes it for training
    '''
    # df = {key: np.array(value)[:,tf.newaxis] for key, value in df.items()}
    ds = data.Dataset.from_tensor_slices((df, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=df.shape[0])
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)

    return ds

if __name__ == "__main__":
    main()
    beepy.beep(sound=6)
        