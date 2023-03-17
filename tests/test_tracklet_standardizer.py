# standard imports
import numpy as np
import os, sys

# local imports
import tracklet_standardizer as standardizer
from tracklet import fetch_mpc_data as fetch

def test_get_basic_data_fields():
    ''' We test that the *get_basic_data_fields* function returns a dictionary
        with certain key: value pairs.
    '''
    d = standardizer.get_basic_data_fields()
    assert isinstance(d, dict)
    
    for key, val in {   'primary_obs': 10,
                        'supplementary_obs' : 3,
                        'derived_obs_pair' : 3,
                        'supplementary_tracklet':2}.items():
        assert key in d
        assert len(d[key]) == val
        for s in d[key]:
            assert isinstance(s, str)

def test_get_basic_data_sizes():
    ''' We test that the *get_basic_data_sizes* function returns the
        expected number of variables, and that the variables have the expected
        relationship / values'''
    _ = standardizer.get_basic_data_sizes()
    
    assert len(_) == 6
    
    N_obs, N_ppo, N_spo, N_dpt, N_spt, N_tot = _
    assert N_tot == N_obs * N_ppo + N_obs * N_spo  + N_dpt*(N_obs - 1)  + N_spt


def test_get_header():
    ''' We test that the *get_header* function returns a string that is of the
        same length as the total number of variables returned by the
        *get_basic_data_sizes* function
    '''
    # Run the header function we want to test
    h = standardizer.get_header()
    assert isinstance(h, str)

    # Run the size-func to get the expected length of the header
    N_obs, N_ppo, N_spo, N_dpt, N_spt, N_tot = standardizer.get_basic_data_sizes()
    assert len(h.split(',')) == N_tot


def test_get_template_array():
    ''' We test that the *get_template_array* returns an array of the
        expected length
    '''
    # Run the get-array function we want to test
    a = standardizer.get_template_array()
    assert isinstance(a, np.ndarray)

    # Run the size-func to get the expected length of the header
    N_obs, N_ppo, N_spo, N_dpt, N_spt, N_tot = standardizer.get_basic_data_sizes()
    assert len(a) == N_tot

def test_get_uv():
    ''' We test that the *get_uv* function returns a list (unit-vector)
        with the expected component values
    '''
    # Define som 'obs' to act as inputs
    obs = [1234567.89 , 0.0 , 0.0 , 24.0 ]
    
    # Call the uv function (the one we are testing)
    uv = standardizer.get_uv(obs)
    assert np.allclose(uv , [1.0, 0.0, 0.0])

# def test_annotate_and_reformat_tracklet_A


def test_create_standardized_file():
    ''' We test that the *create_standardized_file* function creates a
        file of tracklets in a standardized form
    '''
    
    # Input file
    SU = fetch.mpc_data['UnnObs']

    # Output file
    this_directory = os.getcwd() # <<-- tracklet_standardizer directory
    data_directory = os.path.join(os.path.dirname(this_directory), "data")
    out_filepath_NONNEO = os.path.join(data_directory, "TEST_UNNOBS_NONNEO.trk")
    
    # Get rid of any pre-existing files ...
    if os.path.isfile(out_filepath_NONNEO): os.remove(out_filepath_NONNEO)
    assert not os.path.isfile(out_filepath_NONNEO)
    if os.path.isfile(out_filepath_NONNEO+'map'): os.remove(out_filepath_NONNEO+'map')
    assert not os.path.isfile(out_filepath_NONNEO+'map')

    # Run the *create_standardized_file* we want to test ...
    with open(SU.get_filepath(), 'r') as filestream:
        standardizer.create_standardized_file(  filestream ,
                                                out_filepath=out_filepath_NONNEO,
                                                orbit_type='NON-NEO',
                                                out_option = 'w',
                                                i_break = 100)
    # Test that a file of tracklets got made
    assert os.path.isfile(out_filepath_NONNEO)
    
    # Test that the file of tracklets contains rows of the expected size
    N_obs, N_ppo, N_spo, N_dpt, N_spt, N_tot = standardizer.get_basic_data_sizes()
    with open(out_filepath_NONNEO,'r') as fh:
        data = fh.readlines()
    for line in data:
        assert len( line.strip('\n').split(',') ) == N_tot + 2 # <<-- Add 2 for objectID & trkID
    
    # Test that an associated "mapping" file got made
    # - This maps the original designation to an anonymous label
    assert os.path.isfile(out_filepath_NONNEO+'map')

                                                
                                                
        
if __name__ == '__main__':
    test_create_standardized_file()
