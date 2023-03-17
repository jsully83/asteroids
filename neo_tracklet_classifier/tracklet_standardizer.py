"""
Code to convert tracklet data from MPC obs80 format
into a standard format that is more useful for Machine Learning

Requires the *tracklet* package

MJP 2022-07-XX
"""
# standard imports
from collections import namedtuple
import numpy as np
import sys
#  imports ...
from neo_tracklet_classifier import split_mpc as split
from neo_tracklet_classifier import fetch_mpc_data as fetch



# ---------- TOP-LEVEL FUNCTION(S) TO GENERATE FILE OF STANDARDIZED-FORMAT TRACKLETS -----
# ----------------------------------------------------------------------------------------

def create_standardized_file( filestream , out_filepath, orbit_type='ALL', out_option = 'w', i_break = None):
    """ Convenience function to convert obs80 observations into a set of
        tracklets, where the data for each tracklet is in a fixed-width
        format.
        
        The original observational data is auugmented by the addition
        of various "derived" and "supplementary" pieces of data
        
        The derived data is saved to the file specified by "out_filepath"
    """
    # Might be useful to record the desig:number mapping
    desig_number_dict = {}
    
    # For subsequent output ...
    list_of_standardized_tracklets = []
    n_chunk = 100
    with open(out_filepath, out_option ) as out_fh:  # <<-- out_option defaults to 'w' (see above)
    
        # Write header to file
        out_fh.write( 'objectID,' + 'trkID,' + get_header() + '\n')
    
        # Use the generator, *gen_minimal_anonymized()*, to create  anonymous tracklets
        # NB: Yields a dictionary per *object*, containing multiple tracklets per object
        for i, _ in enumerate( gen_minimal_anonymized( filestream, orbit_type=orbit_type) ):
        
            try:
        
                # Extract the two dictionaries returned from *gen_minimal_anonymized* ...
                object_output_dict, desig_number_map = _
                desig_number_dict.update(desig_number_map)
                anon_Object = list(desig_number_map.values())[0]
                    
                # Calculate various additional quantities for each tracklet in the object,
                # and create a standardized format for each tracklet
                for trkID, tracklet in object_output_dict.items():
                
                    # Get standard output
                    standardized_tracklet = annotate_and_reformat_tracklet_A( tracklet )
                    # Convert to string (along with objectID & trkID)
                    str_form              = anon_Object +  "," + trkID  +  ","  +  ",".join( [ f"{_}" for _ in standardized_tracklet] ) + "\n"
                    # Save into a list
                    list_of_standardized_tracklets.append( str_form )
                        
            except:
                pass

            # Save to file (writing in chunks...)
            if i > 0 and i % n_chunk == 0 :
                list_of_standardized_tracklets = save_chunk(list_of_standardized_tracklets , out_fh, i)

                   
            # Option to break (useful during development / to make small output-files)
            if i_break != None and i > i_break :
                break
            
        # Save to file (if a partial chunk remains...)
        list_of_standardized_tracklets = save_chunk(list_of_standardized_tracklets , out_fh, '[final]')

    print('Writing mapping...',out_filepath+'map')
    with open(out_filepath+'map', out_option ) as map_fh:  # <<-- out_option defaults to 'w' (see above)
        for k,v in desig_number_dict.items():
            map_fh.write(f"{k}:{v}\n")
            
    print('Writing complete')

def save_chunk(list_of_standardized_tracklets , out_fh, i):
    ''' Write a "chunk" of data to file ...'''
    print('Writing chunk...',i)
    for item in list_of_standardized_tracklets:
        out_fh.write( item  )
    return []


# ---------- LOWER-LEVEL FUNCTIONS FOR ITERATING THROUGH, & REFORMATING, TRACKLETS------------
# --------------------------------------------------------------------------------------------

def gen_minimal_anonymized( filestream , orbit_type):
    """ Generator to create a set of anonymized "minimal" tracklets
        - "Minimal" in the sense that most/all unnecessary info is
          removed from the observations within each tracklet

        Generates the data using the functionality from the
        tracklet module/package ...

        Use the 'orbit_type' option to select the type of objects desired
        
    """
    
    # check the input orbit_type is valid
    allowed_orbit_type = ['ALL','NEO','NON-NEO']
    assert orbit_type in allowed_orbit_type, \
        f"orbit_type = {orbit_type} : this is not in the allowed list: {allowed_orbit_type}"
    
    
    # use the generator function from "tracklet" module to iterate through the file
    # - Note that we use the 'orbit_type' option to select the type of object desired ...
    for m, _ in enumerate( split.gen_object_tracklets(filestream,
                                                    min_obj_obs= 50,
                                                    t_crit_hrs=8.0,
                                                    min_trk_obs=2,
                                                    min_trklts=5,
                                                    NEA_type = orbit_type,
                                                    NEA_type_file = fetch.mpc_data['NEAOrbits'].get_filepath()
                                                    ) ) :
                                                                
                                                                
        # Blank dict to hold reformated results for an object
        # This will be yielded for each object ...
        object_output = {}
        
        # split output from generator: list of observation-objects and list of strings.
        # there are lists-of-lists due to there being one-per-tracklet...
        object_list_of_lists, str_list_of_lists  = _

        # loop through each tracklet ...
        for n, tracklet_obj_list in enumerate(object_list_of_lists):

            # Generate an anonymous trkID for each tracklet
            # Contains object-type, object-number and tracklet-number
            anon_trkID = new_trkID(orbit_type , m, n )
            anon_Object= anon_trkID[:anon_trkID.rfind('_')]
            
            # Select a minimal amoouunt of information for each observation in the tracklet
            object_output[anon_trkID] = [ [o.jdutc, o.ra, o.dec, o.mag] for o in tracklet_obj_list ]
            
        # yield (a) all the data for a single object,
        # and   (b) object-number : desig mapping
        yield object_output, {object_list_of_lists[0][0].desig : anon_Object }


def new_trkID(orbit_type , object_m, tracklet_n ):
    """ Generate a new anonymous name/label for a tracklet """
    return f'{orbit_type[:3]}_{object_m:07}_{tracklet_n:05}'



def annotate_and_reformat_tracklet_A( tracklet ):
    ''' Version (A) of a function to reformat MPC tracklet data into
        a standardized format approopriate for machine learning.
        
        Contains information from the original constituent observations,
        along with with "derived" and "supplementary" data
         - "derived" quantities (e.g. angular-velocities, ...)
         - "supplementary" quantities (e.g. solar-elongation, digest2-score, ...)
         
        Reformats the data into a single list/array for the entire tracklet
        
        NB(1): *** It is entirely possible that the chosen format is bad  ***
               *** for machine learning: if so, make another function     ***
               *** that produces a different format !!!                   ***

        NB(2): *** This function is COMPLETELY UNFINISHED!!!!             ***
               *** Lots of the sub-functions are "stub-functions" that    ***
               *** just return zero / some random integer instead of      ***
               *** performing a useful task. TO BE COMPLETED!!!           ***

    '''
    # Get basic defined sizes for data components ...
    # - This is where we are making a lot of the design choices / specifications
    N_obs, N_ppo, N_spo, N_dpt, N_spt, N_tot = get_basic_data_sizes()

    # Get array of zeroes (standardized length)
    # - This is where we set the fixed size of the output array for each tracklet
    tracklet_data_array = get_template_array()
    
    # If the tracklet has > N_obs, then just take the *LAST* N_obs [could take first N_obs instead?]
    tracklet_subset = tracklet[-N_obs:]

    # Iterate through the observations in each tracklet, and ...
    # (1) Convert obs (RA,Dec) to Unit-Vectors
    # (2) Calculate "supplementary" quantities (e.g. solar-elongation) for each observation
    # (3) Populate the appropriate portion of the array
    for i, obs in enumerate( tracklet_subset ): # <<-- iterate over last 10 obs in tracklet

        # Convert RA,Dec -> Unit Vector
        uv = get_uv(obs)
        
        # Decide what to do if magnitude is blank
        mag = 99 if obs[3] in [None, '', ' '] else float(obs[3])
        
        # Calculate supplementary quantities
        so_list = supplement_obs(obs, uv)
        
        # Put the data into the appropriate section of the 'tracklet_data_array'
        offset = i * (N_ppo + N_spo)                            # offset <=> start of data assoc. w/ each obs
        tracklet_data_array[ offset ]         = float(obs[0])   # 1x time
        tracklet_data_array[ offset + 1 : offset + 4]  = uv     # 3x UV components
        tracklet_data_array[ offset + 4]      = mag             # 1x magnitude
        #...                                                    # 5x uncertainty components NOT TOUCHED IN THIS VERSION s
        tracklet_data_array[ offset + N_ppo : offset + N_ppo + N_spo] = so_list       # 3x supplementary components


    # Calculate "derived" quantities (e.g. angular-velocities) for each obs-pair
    for j in range(len(tracklet_subset)-1):
    
        # calculated derived quantities ...
        d_list = derive_obs_pair(tracklet_subset[j], tracklet_subset[j+1])
        
        # populate the appropriate section of the 'tracklet_data_array'
        offset = (N_obs * N_ppo + N_obs * N_spo) + j * N_dpt      # offset <=> start of data assoc. w/ derived info
        tracklet_data_array[ offset: offset + N_dpt] = d_list     # 3x derived components
        #print(len(tracklet_subset), j, offset, offset + N_dpt, d_list )
        
    # Calculate "supplementary" quantities (e.g. digest2-core) for each tracklet
    st_list = supplement_tracklet(tracklet_subset)
    assert len(st_list) <= N_spt
    offset = (N_obs * N_ppo + N_obs * N_spo) + N_dpt * (N_obs - 1)   # offset <=> start of data assoc. w/ supp. data
    tracklet_data_array[ offset : offset + N_spt] = st_list          # 3x derived components

    # Return the populated tracklet_data_array
    # NB(1) During development, some fields may not be populated
    # NB(2) If the number of supplied obs < 10, then some fields will not be populated
    return tracklet_data_array
    
    
    

# ---------- FUNCTIONS REGARDING THE SPECIFICATION AND FORMATTING OF THE DATA ----------------
# --------------------------------------------------------------------------------------------

def get_basic_data_fields():
    '''
        ****************************************************************************
        *** THIS IS WHERE DECISIONS NEED TO BE MADE ABOUT THE FORMAT OF THE DATA ***
        ***
        *** FOR NOW I AM JUST USING A NUMPY ARRAY:
        ***  - IT MAY BE BETTER / CLEARER TO USE A CLASS/NAMEDTUPLE/PANDAS-TABLE...
        ***
        *** To make progress I am going to assume the following ...
        ***  (i) Fixed max length will correspond to a 10-observation tracklet
        *** (ii) Number of quantities (i.e. length of list/array) will be ...
        ***      N_tot
        ***        = N_obs * N_ppo + N_obs * N_spo  + N_dpt*(N_obs - 1)  + N_spt
        ***        = 10*10 + 10*3 + 27 + 2
        ***        = 159
        *** (iv) Pattern of data will be
        ***                      obs-data                 +       obs-derived       + tracklet-level
        ***   => [N_ppo + N_spo]_0 ... [N_ppo + N_spo]_9  + [N_dpt]_0 ... [N_dpt]_8 + N_spt
        ***
        ****************************************************************************
    '''
    return {
             'primary_obs'              : ['JDUTC','X','Y','Z','M','sigmaJDUTC','sigmaX','sigmaY','sigmaZ','sigmaM'],
             'supplementary_obs'        : ['solar_ang','lunar_ang','jovian_ang'],
             'derived_obs_pair'         : ['mu_ra', 'mu_dec', 'mu_sq'],
             'supplementary_tracklet'   : ['d2_NEO_ALL', 'd2_NEO_NEW']
             }

def get_basic_data_sizes():
    '''
    # Define the number / length of the various quantities
    # - See Google-Doc, "Background: Observations and Tracklets"
    '''
    N_obs = 10              # => Fixed maximum number of observations per tracklet:
    
    # Get basic field names
    basic_data_fields = get_basic_data_fields()
    
    #Number of primary data points per observation
    N_ppo = len(basic_data_fields['primary_obs'])
    
    #Number of supplementary data points per observation
    N_spo = len(basic_data_fields['supplementary_obs'])
    
    #Number of derived data points per obs-pair
    N_dpt = len(basic_data_fields['derived_obs_pair'])
    
    #Number of supplementary data points per tracklet
    N_spt = len(basic_data_fields['supplementary_tracklet'])

    # Total number of data-points per tracklet
    N_tot = N_obs * N_ppo + N_obs * N_spo  + N_dpt*(N_obs - 1)  + N_spt

    return N_obs, N_ppo, N_spo, N_dpt, N_spt, N_tot
    
def get_header():
    ''' Make a header string of length == N_tot
        It contains all of the field names
        The header is populated in the same order as the data
    '''
    # Get basic field names
    basic_data_fields = get_basic_data_fields()
    
    # Get basic data sizes ...
    N_obs, N_ppo, N_spo, N_dpt, N_spt, N_tot = get_basic_data_sizes()

    # Construct header out of the field names ...
    header = ''
    for i in range(N_obs):
    
        # Headers for basic observational quantities
        for f in basic_data_fields['primary_obs']:
            header = header + f'{f}_{i}, '
        
        # Header supplementary quantities per observation
        for f in basic_data_fields['supplementary_obs']:
            header = header + f'{f}_{i}, '


    for i in range(N_obs-1):

        # Headers for derived quantities (e.g. Ang-Vel)
        for f in basic_data_fields['derived_obs_pair']:
            header = header + f'{f}_{i}, '

    # Header(s) for supplementary string at the tracklet level
    for f in basic_data_fields['supplementary_tracklet']:
        header = header + f'{f}, '

    header = header[:-2]
    return header



def get_template_array( HEADER_ONLY = False ):
    '''
        Create a standard template array of standard length to hold the data for a tracklet
        The returned array consists only of **zeroes**
    '''

    # Get basic data sizes ...
    N_obs, N_ppo, N_spo, N_dpt, N_spt, N_tot = get_basic_data_sizes()
    # Return the array-of-zeroes (of length N_tot)
    return np.zeros(N_tot)


# ---------- FUNCTIONS TO PERFORM CALCULATIONS REQUIRED FOR DERIVED/SUPPLEMENTARY DATA -------
# --------------------------------------------------------------------------------------------

        
def get_uv(obs):
    '''
    # Convert RA,Dec -> Unit Vector
    '''
    assert len(obs) in [4,8]
    theta_rad = np.radians(90. - obs[2]) # theta_deg = 90 - Dec_deg
    RA_rad    = np.radians(obs[1])
    uv = [np.sin(theta_rad)*np.cos(RA_rad), np.sin(theta_rad)*np.sin(RA_rad), np.cos(theta_rad)]
    return uv
    
    
def supplement_obs(obs, uv):
    '''
    Calculate supplementary quantities for each observation
    E.g. ['solar_ang','lunar_ang','jovian_ang'],
    *** NOT YET IMPLEMENTED ***
    '''
    supp = [4.0, 5.0, 6.0] ##<<-- This is a placeholder!!!
    assert len(supp) == len( get_basic_data_fields()['supplementary_obs'] )
    return supp

def derive_obs_pair(obs_a , obs_b ):
    '''
    Calculate derived quantities for a given pair of obs
    This is primarily going to be the angular velocities
     - https://en.wikipedia.org/wiki/Proper_motion
    '''
    dt     = obs_b[0]  - obs_a[0]
    mu_ra  = (obs_b[1] - obs_a[1])/dt
    mu_dec = (obs_b[2] - obs_a[2])/dt
    mu_sq  = mu_ra**2 + mu_dec**2 * np.cos(obs_a[2])
    d = [mu_ra, mu_dec, mu_sq]

    assert len(d) == len( get_basic_data_fields()['derived_obs_pair'] )
    return d

def supplement_tracklet(tracklet):
    '''
    # Calculate "supplementary" quantities (e.g. digest2-core) for each tracklet
    *** NOT YET IMPLEMENTED ***
    '''
    supp = [7.0, 8.0] ##<<-- This is a placeholder!!!
    assert len(supp) == len( get_basic_data_fields()['supplementary_tracklet'] )
    return supp
             
