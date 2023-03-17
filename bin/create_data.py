"""
Creates the NEO data needed to train a model
"""

import pandas as pd
import numpy as np

# The *tracklet* module we'll use as the source of data ...
from pathlib import Path

# neo classifier 
from neo_tracklet_classifier import directory
from neo_tracklet_classifier import tracklet_standardizer as standardizer
from neo_tracklet_classifier import header
from neo_tracklet_classifier import fetch_mpc_data as fetch


def load_trk(path):
    """
    Loads tracklet data from the tracklet standardizer

    Args:
        path (str): path to the tracklet file

    Returns:
        Dataframe: a dataframe of the tracklet file
    """
    print("Loading tracklet files...")
    df_data = pd.read_csv(path)
    df_data.columns = df_data.columns.str.replace(' ', '')
    return df_data

def process_tracklet(df_neo, df_nonneo):
    """
    Adds a NEO column to the tracklet file to denote weather the tracklet is a 
    1 = NEO or 0 = Non-NEO. Reduce the two dataframes to only the features we 
    want then concatenate

    Args:
        df_neo (Dataframe): dataframe with NEOs
        df_nonneo (Dataframe): dataframe with Non-NEOS

    Returns:
        Dataframe: a dataframe with NEOs and Non-NEOs in a single file
    """
    print('Processing tracklet files...')
    df_neo.insert(0, 'NEO', 1)
    df_nonneo.insert(0, 'NEO', 0)

    df_all = pd.concat([df_neo[header.header_numeric_features],
                        df_nonneo[header.header_numeric_features]],
                        ignore_index=True)

    return df_all

def save_tracklet_features():
    """
    First we load data from tracklet standardizer.  The data is saved 
    seperately in NEO and NON-NEO files. We add a column to 
    differentiate between the two datasets, concatenate them and save
    them as a numpy npy file for faster loading. We usually don't 
    have to do this unless there is new data from the MPC.
    """
    # define filepaths
    neo_path = Path.joinpath(directory.data_dir, "NEO.trk")
    nonneo_path = Path.joinpath(directory.data_dir, "NONNEO.trk")
    npy_path = Path.joinpath(directory.data_dir, 'allneo.npy')
    csv_path = Path.joinpath(directory.data_dir, 'allneo.csv')

    # load data
    df_neo = load_trk(neo_path)
    df_nonneo = load_trk(nonneo_path)

    # process
    df_all = process_tracklet(df_neo, df_nonneo)

    # save as npy
    print('Saving tracklets as an .npy file...')
    np.save(npy_path, df_all)

    # save as csv to use in Top Cat
    print('Saving tracklets as an .csv file...')
    df_all.to_csv(csv_path, index=False)

    print('Completed.')
    
    
if __name__ == "__main__":
    # create a new directory for our data
    
    if not Path.is_dir(directory.data_dir): Path.mkdir(directory.data_dir)

    # Define some filepaths for output files...
    out_filepath_NEO = Path.joinpath(directory.data_dir, "NEO.trk")
    out_filepath_NONNEO = Path.joinpath(directory.data_dir, "NONNEO.trk")
    
    # fetch data from the MPC using the tracklet package
    SU = fetch.mpc_data['UnnObs']
    SN = fetch.mpc_data['NumObs']
    
    # SU.fetch()
    print(SN.get_filepath())
    
    # Use the *create_standardized_file* function from the *tracklet_standardizer* module
    # to create a file of tracklets in a standardized form 
    
    # NEOS
    # with open(directory.data_dir, 'r') as filestream:
    #     standardizer.create_standardized_file(  filestream , 
    #                                             out_filepath=out_filepath_NEO, 
    #                                             orbit_type='NEO' )

    # # Non-NEOS
    # with open(directory.data_dir, 'r') as filestream:
    #     standardizer.create_standardized_file(  filestream , 
    #                                             out_filepath=out_filepath_NONNEO, 
    #                                             orbit_type='NON-NEO' )
    
    # save_tracklet_features()
    