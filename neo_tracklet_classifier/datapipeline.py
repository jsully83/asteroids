'''
defines functions to be used in teh data pipelining processes
for training a neural network to find NEOs
'''

import os
from neo_tracklet_classifier import directory, header
import numpy as np
import pandas as pd

from tensorflow import data
from tensorflow.ragged import constant
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import clear_output


class DataPipeline():
    def __init__(self, total_samples, subsample_size, training_size, batch_size, deviation, ragged=False, create_ds=True) -> None:
        percent_of_data = subsample_size/total_samples
        split = training_size/subsample_size

        # load data
        npy_path = os.path.join(directory.data_dir, 'allneo.npy')

        self.df = pd.DataFrame(np.load(npy_path, allow_pickle=True),
                               columns=header.header_numeric_features)

        # subsample the data and divide into training, valiation and test sets
        self.sample_df = self.sample_dataframe(self.df, percent_of_data, deviation)
        
        
        # maybe this could be it's own function.
        bad_index = []

        for i in range (0,2):
            col = ['JDUTC_', 'X_', 'Y_', 'Z_', 'M_', 'mu_ra_', 'mu_dec_', 'mu_sq_']
            
            for _, name in enumerate(col):
                if name == 'mu_ra_' and i == 0:
                    bad_index.append(self.sample_df[self.sample_df[name+str(i)].isin([0.0])].index.values)
                
                elif name == 'mu_dec_' and i == 0:
                    bad_index.append(self.sample_df[self.sample_df[name+str(i)].isin([0.0])].index.values)
                
                elif name == 'mu_sq_' and i == 0:
                    bad_index.append(self.sample_df[self.sample_df[name+str(i)].isin([0.0])].index.values)
                    
                elif name == 'M_':
                    bad_index.append(self.sample_df[self.sample_df[name+str(i)].isin([99.0])].index.values)

                else:
                    bad_index.append(self.sample_df[self.sample_df[name+str(i)].isin([0.0])].index.values)

        # Maybe needs a comment
        bad_index = np.concatenate([bad_index[n] for n in range(0, len(bad_index))])
        
        
        bad_index = np.unique(bad_index)
        self.sample_df = self.sample_df.drop(bad_index, axis=0)
        
        self.train, self.val, self.test = self.split_data(self.sample_df, split, deviation)
        self.train_df, self.val_df, self.test_df, self.train_labels, self.val_labels, self.test_labels = self.create_features_labels(
           self.train, self.val, self.test)

        if ragged is not False:
            self.train_rag = self.create_ragged_tensor(self.train_df)
            self.val_rag = self.create_ragged_tensor(self.val_df)
            self.test_rag = self.create_ragged_tensor(self.test_df)
            # print('create train rag')
            # self.train_ragds = self.df_to_dataset(self.train_rag, batch_size, True, self.train_labels)
            # print('create train rag')
            # self.val_ragds = self.df_to_dataset(self.val_rag, batch_size, False, self.val_labels)
            # print('create train rag')
            # self.test_ragds = self.df_to_dataset(self.val_rag, batch_size, False, self.test_labels)

        elif create_ds is not False:
            
            train_label_np = np.array(self.train_labels)
            val_label_np = np.array(self.val_labels)
            test_label_np =  np.array(self.test_labels)
            
            print('create train ds')
            self.train_ds = self.df_to_dataset(
                self.train_df, batch_size, True, train_label_np)
            print('create val ds')
            self.val_ds = self.df_to_dataset(
                self.val_df, batch_size, False, val_label_np)
            print('create test ds')
            self.test_ds = self.df_to_dataset(
                self.test_df, batch_size, False, test_label_np)
            print('done')


    def shift_values_left(self, df):
        '''
        Finds zeros in a dataframe and sorts each row so that data is moved to the
        left and zeros are turned into NaNs and moved to the end of each row.  We
        choose to slice the dataframe at column 11 because each tracklet must have
        at least two observations.  We create a bitmask for elements equal to zero
        and mergesort the Trues and Falses while keeping their relative order. i 
        is an array of indicies foreach elements new position.  Values are 
        shifted according to the array i and copied back to the original dataframe.
        https://stackoverflow.com/questions/66733075/efficiently-remove-all-zeroes-from-pandas-dataframe

        '''
        s = df.iloc[:, 11:]
        i = np.argsort(s.eq(0.0), axis=1, kind='mergesort')
        df.iloc[:, 11:] = np.take_along_axis(
            s[s.ne(0.0)].values, i.values, axis=1)

        return df

    def create_ragged_tensor(self, df):
        '''
        Creates a ragged tensor of tracklets and their observations.  We take a
        padded array, remove the zeros and shift all values left.  Changed the
        array to a nested list and iterate through each list while filtering out
        the NaNs.  Returns a ragged tensor with each row containing a tracklet
        of differing lengths.
        https://stackoverflow.com/questions/70982172/remove-all-nan-from-nested-list
        '''
        print('creating ragged')
        x = self.shift_values_left(df)
        y = (x.to_numpy()).tolist()
        uneven_list = [list(filter(lambda x: x == x, inner_list))
                       for inner_list in y]
        ragged_tensor = constant(uneven_list)

        return ragged_tensor

    def find_imbalance(self, df_original, df_sample, deviation):
        '''
        Checks that the ratio of NEOs and Non-NEOs between the two datasets, 
        original and sample, are within the specified deviation.
        '''
        pct_dev = deviation * 100.0

        imbalance = df_original.value_counts('NEO', normalize=True)

        sample_imbalance = df_sample.value_counts('NEO', normalize=True)

        neo_high = imbalance[1] + (imbalance[1] * deviation)
        neo_low = imbalance[1] - (imbalance[1] * deviation)
        sample_neo = sample_imbalance[1]

        nonneo_high = imbalance[0] + (imbalance[0] * deviation)
        nonneo_low = imbalance[0] - (imbalance[0] * deviation)
        sample_nonneo = sample_imbalance[0]

        if (neo_low <= sample_neo <= neo_high) and (nonneo_low <= sample_nonneo <= nonneo_high):
            # print('Relative Frequency\n', end="")
            # print('NEOs:           ', end="")
            # print('%.2f' % (imbalance[1]*100), '%')
            # print('Non-NEOs:       ', end="")
            # print('%.2f' % (imbalance[0]*100), '%\n')

            # print('Sample data should be within %.2f' % pct_dev, '%')

            # print('neo_low:       %.4f' % (neo_low * 100), '%', end='        ')
            # print('nonneo_low:    %.4f' % (nonneo_low * 100), '%')
            # print('sample_neo     %.4f' %
            #       (sample_imbalance[1] * 100), '%', end='        ')
            # print('sample_nonneo: %.4f' % (sample_imbalance[0] * 100), '%')
            # print('neo_high:      %.4f' %
            #       (neo_high * 100), '%', end='        ')
            # print('nonneo_high:   %.4f' % (nonneo_high * 100), '%\n')

            # clear_output()
            return False  # if balance is good don't run again

        else:
            print(f'Not within +/- {deviation * 100} %')
            return True  # if balance is bad run again

    def sample_dataframe(self, df, pct, deviation):
        '''
        Samples the data set by pct and verifies that the sample has within
        0.05% of NEO and Non-NEO rows.
        '''
        b = True
        while b:
            self.df_sample = df.sample(frac=pct)
            b = self.find_imbalance(df, self.df_sample, deviation)

        return self.df_sample

    def split_data(self, df_sample, train_val_size, deviation):
        '''
        Splits the data into training, validation and test sets.  sklearn's
        train_test_split function randomly samples the given array based on
        the fraction given data given.It also does not split into three
        subarrays so we choose a train/val size and split with test then 
        use the train/val data to split into training and validation. The
        validation set should be the same length as the test set.  We 
        then check for overlapping indicies and deviation. 
        '''
        b = np.full(6, False)
        while not b.all():
            train_val_test_split = train_test_split(
                df_sample, train_size=train_val_size)
            train_val_split = train_test_split(
                train_val_test_split[1], test_size=0.5)
            df_train = train_val_test_split[0]
            df_val = train_val_split[1]
            df_test = train_val_split[0]

            b[0] = bool(list(set(df_train.index.values)
                        and set(df_test.index.values)))
            b[1] = bool(list(set(df_test.index.values)
                        and set(df_val.index.values)))
            b[2] = bool(list(set(df_val.index.values)
                        and set(df_train.index.values)))

            clear_output()

            b[3] = not self.find_imbalance(df_sample, df_train, deviation)
            b[4] = not self.find_imbalance(df_sample, df_val, deviation)
            b[5] = not self.find_imbalance(df_sample, df_test, deviation)

        print(len(df_train), 'df_training examples')
        print(len(df_val), 'validation examples')
        print(len(df_test), 'df_test examples\n')

        df_train = pd.DataFrame(df_train).sort_index()
        df_val = pd.DataFrame(df_val).sort_index()
        df_test = pd.DataFrame(df_test).sort_index()
    
        return df_train, df_val, df_test

    def create_features_labels(self, train, val, test):
        '''
        takes the data from our three sets and creates a normalized feature array.
        returns the features and the labels
        '''
        # train_labels = np.array(self.train.pop('NEO'))
        # val_labels = np.array(self.val.pop('NEO'))
        # test_labels = np.array(self.test.pop('NEO'))
        
        train_labels = self.train.pop('NEO')
        val_labels = self.val.pop('NEO')
        test_labels = self.test.pop('NEO')
        
        
        # bool_train_labels = train_labels != 0

        train.replace(0, np.nan, inplace=True)
        val.replace(0, np.nan, inplace=True)
        test.replace(0, np.nan, inplace=True)

        train_features = np.array(train)
        val_features = np.array(val)
        test_features = np.array(test)

        scaler = StandardScaler()
        train_features = pd.DataFrame(scaler.fit_transform(train_features),
                                      columns=header.header_features)
        val_features = pd.DataFrame(scaler.transform(
            val_features), columns=header.header_features)
        test_features = pd.DataFrame(scaler.transform(
            test_features), columns=header.header_features)

        train_features.replace(np.nan, 0, inplace=True)
        val_features.replace(np.nan, 0, inplace=True)
        test_features.replace(np.nan, 0, inplace=True)

        return train_features, val_features, test_features, train_labels, val_labels, test_labels

    def df_to_dataset(self, df, batch_size, shuffle, labels):
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
