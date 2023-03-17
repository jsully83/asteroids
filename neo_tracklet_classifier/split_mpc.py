'''
Code to parse & split mpc flat-files into TRACKLETS or OBJECTS

Splitting into tracklets will be done according to adjustable criteria:
 - E.g. 8hr data gaps, etc

'''

# import ...
import sys, os
import time
from collections import defaultdict
from functools import lru_cache

# import obs80 package
# - obviously this won't work for anyone else, just doing while developing

# sys.path.append()

# os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'obs80')

from obs80 import obs80


# -------------- READ NEA DESIGNATIONS -----------------

@lru_cache(maxsize = 2)
def read_nea_desigs( filepath ):
    '''
    Reads a file of designations and extracts the designations from the first 'column' of each row
    Returns as a dictionary
    '''
    # read the orbit data
    with open(filepath, 'r') as fp:
        data = fp.readlines()

    # extract the designations
    # - These are a mixture of perm & prov IDs
    # - All are *packed*
    desigs = {_.split()[0].strip():True for _ in data}
    return desigs

# -------------- READ ORBITAL ELEMENTS -----------------

@lru_cache(maxsize = 2)
def read_orbital_els( filepath ):
    '''
    Reads a file of orbits asssumed to be in some format like MPCORB ...
    Des'n     H     G   Epoch     M        Peri.      Node       Incl.       e            n           a        Reference #Obs #Opp    Arc    rms  Perts   Computer

    and extracts ...
     - the designations from the first 'column' of each row
     - the elements a, e, i
     
    Returns as a dictionary-of-lists
    '''
    # read the orbit data
    with open(filepath, 'r') as fp:
        data = fp.readlines()
        hdr  = [ i for i,l in enumerate(data[:100]) if "------------" in l][0]
        data = [ _ for _ in data[hdr + 1:] if len(_) > 10 ]

    # extract the reqd data
    data_dict = {}
    for _ in data:
        desig  = _.split()[0].strip()
        a      = float(_[93:103].strip())
        e      = float(_[69:79].strip())
        i      = float(_[59:68].strip())
        data_dict[desig] = {'a':a, 'e':e, 'i':i }

    return data_dict



# -------------- "SPLITTING" GENERATORS ----------------

def gen_objects(    s ,
                    min_obj_obs     =  2,
                    central_jdutc   =None,
                    delta_jdutc     =None,
                    NEA_type        =None,
                    NEA_type_file   =None,
                    a_min           =None,
                    a_max           =None,
                    e_min           =None,
                    e_max           =None,
                    orbit_els_file  =None,
                    ):
    ''' *generator* : parses a text stream, 's', of 80 column observations.

    uses Sonia's obs80 code

    yields 2-lists: (a) obs80 observation-objects, (b) obs80 test-strings
     - one OBJECT per list
     - I.e. you get all of the observations associated with an object
     
    options:
    --------
    min_obj_obs : int
     - minimum number of observations required per oject
     - defaults to only returning objects with at least 'min_obj_obs' = 2 observations
     
    central_jdutc   : float
    delta_jdutc     : float
     - Used to specify acceptable jdutc ... central_jdutc +- delta_jdutc
     
    NEA_type : str
    NEA_type_file : str
     - Used to specify a type of orbbit to be selected ( NEO / NON-NEO / ALL )
     
    a_min : float
    a_max : float
    e_min : float
    e_max : float
    orbit_els_file : str
     - Used to specify a subset of orbital parameters to draw from


    '''
    
    closed = False


    # Check that any optional inputs are correctly specified ...
    assert  NEA_type is None or \
            NEA_type.upper() in ['ALL','NEO','NON-NEO'] and isinstance(NEA_type_file, str), f'NEA_type={NEA_type}'
            
    orbital_dict = read_orbital_els( orbit_els_file ) if orbit_els_file is not None else {}
            

    # Used to identify & record similarity ...
    SameObject = SameAsPrevious()
    obj_list = []
    str_list = []

    # iterate over obs80 strings
    for _ in obs80.parse80verbose(s):
    
        # The return from parse80 has been hacked to return in the form
        # obs80_object , tuple-of-strings [len=1 or 2]
        try:

            # ---------------------------------------------------------
            # --- (1) --- Exclude problematic data --------------------
            # ---------------------------------------------------------
            try: # If this doesn't have the necessary attributes, skip...
                o, str_ = _               # <<-- will fail if components don't exist
                assert hasattr(o,'cod')   # <<-- will fail if no cod param
                assert hasattr(o,'desig') # <<-- will fail if no desig param
                assert hasattr(o,'num')   # <<-- will fail if no num param
                designation = o.desig.strip() if o.desig not in ['',' '] else o.num.strip()
            except:
                continue
                

                
                
            # ---------------------------------------------------------
            # --- (2) --- Exclude data outside desired range/class ----
            # ---------------------------------------------------------
            # If this is *NOT* within the desired date range, then skip
            if    central_jdutc is not None and \
                    delta_jdutc   is not None and \
                    ( o.jdutc < (central_jdutc - delta_jdutc)  or \
                      o.jdutc > (central_jdutc + delta_jdutc)       ):
                pass
                
            # If this is an NEA and we don't want them, then skip
            elif NEA_type is not None and \
                 NEA_type_file is not None and \
                 NEA_type.upper() == 'NON-NEO' and \
                 designation in read_nea_desigs( NEA_type_file ):
                pass
                
            # If this is *not* an NEA and we only want NEA's, then skip
            elif NEA_type is not None and \
                 NEA_type_file is not None and \
                 NEA_type.upper() == 'NEO' and \
                 designation not in read_nea_desigs( NEA_type_file ):
                pass
                
            # If the orbital semi-major axis is outside the desired range, then skip
            elif orbit_els_file is not None and \
                 a_min is not None and \
                 a_max is not None and \
                (   designation not in orbital_dict or \
                    orbital_dict[designation]['a'] < (a_min)  or \
                    orbital_dict[designation]['a'] > (a_max)  ):
                pass
            
            # If the orbital eccentricity is outside the desired range, then skip
            elif orbit_els_file is not None and \
                 e_min is not None and \
                 e_max is not None and \
                (   designation not in orbital_dict or \
                    orbital_dict[designation]['e'] < (e_min)  or \
                    orbital_dict[designation]['e'] > (e_max)  ):
                pass

                
            # ---------------------------------------------------------
            # --- (3) --- Organize the data that will be returned -----
            # ---------------------------------------------------------

            # If this is the same as previous, append to prev list
            # otherwise, start a new list & yield the prev list
            elif SameObject(o, "Objects"):
                obj_list.append( o )
                str_list.append( str_ )
                        
            # If this is different ...
            else:
                return_obj_list = obj_list[:]
                return_str_list = str_list[:]
                
                obj_list = [ o    ]
                str_list = [ str_ ]
                
                # only return data for objects with >=2 observations !!
                if len(return_obj_list) >= min_obj_obs:
                    yield return_obj_list, return_str_list
                 
        # https://stackoverflow.com/questions/27755989/generatorexit-in-while-loop
        except GeneratorExit :
            return
            
 
def get_tracklet_from_object_list(  obj_list,
                                    str_list,
                                    t_crit_hrs  = 8.0,
                                    min_trk_obs = 2,
                                    ):
    '''FUNCTION, parses a list of obs80 objects (and strings).
    
    Splits the object into lists-of-lists, one list per TRACKLET
    '''
    # Make a dictionary to keep them in sync ...
    # Sort dictionary keys to ensure tracklets are not interleaved (we sort based on the obs-code & jdutc)
    d        = {o:s for o,s in zip(obj_list, str_list) }
    keys     = list(d.keys())
    keys.sort(key=lambda x: (x.cod, x.jdutc))

    # Used to identify & record similarity ...
    SameTracklet = SameAsPrevious()
    obj_list_of_lists = []
    str_list_of_lists = []
    index = -1

    # iterate over observations in object ...
    for o in keys:

        # if the observation is the same tracklet ...
        if SameTracklet(o, "Tracklets", t_crit_hrs ):
            obj_list_of_lists[index].append( o )
            str_list_of_lists[index].append( d[o] )
            
        # if we have a new tracklet ...
        else:
            # increment index in list ...
            # Do initial population of lists in that location ...
            index += 1
            obj_list_of_lists.append( [   o  ] )
            str_list_of_lists.append( [ d[o] ] )
            
    # remove tracklets of length < 2 (or whatever value is passed in via min_trk_obs )
    obj_list_of_lists = [o_list for o_list in obj_list_of_lists if len(o_list) >= min_trk_obs ]
    str_list_of_lists = [s_list for s_list in str_list_of_lists if len(s_list) >= min_trk_obs ]

    assert len(obj_list_of_lists)==len(str_list_of_lists)

    return obj_list_of_lists , str_list_of_lists





                
                
def gen_object_tracklets(   s ,
                            # ------ object options -----
                            min_obj_obs= 2,
                            central_jdutc=None,
                            delta_jdutc=None,
                            NEA_type=None,
                            NEA_type_file=None,
                            a_min=None,
                            a_max=None,
                            e_min=None,
                            e_max=None,
                            orbit_els_file=None,
                            # ------ tracklet options -----
                            t_crit_hrs=8.0,
                            min_trk_obs=2,
                            min_trklts=5,
                            discovery_only=False,
                                    ):
    ''' *generator* : parses a text stream, 's', of 80 column observations.
    
    yields 2-lists-of-lists: (a) obs80 observation-objects, (b) obs80 test-strings
     - the return lists-of-lists contain all TRACKLETS for a GIVEN OBJECT
     - within each TRACKLET you get all the constituent OBSERVATIONS
     - I.e. you get all of the observations associated with an object,
      with the observations being split into tracklets
     
    defaults to only returning objects with at least 'min_obj_obs' = 2 observations
    
    uses Sonia's obs80 code
    '''
    
    
    # Use the gen_objects function/generator to iterate through the file
    for m, obj in enumerate( gen_objects( s ,
                                        min_obj_obs       =min_obj_obs,
                                        central_jdutc     =central_jdutc,
                                        delta_jdutc       =delta_jdutc,
                                        NEA_type          =NEA_type,
                                        NEA_type_file     =NEA_type_file,
                                        a_min             =a_min,
                                        a_max             =a_max,
                                        e_min             =e_min,
                                        e_max             =e_max,
                                        orbit_els_file    =orbit_els_file   ,
                                            ) ) :

        # Extract the lists (gen_objects returns two lists...)
        object_list, str_list  = obj

        # Use the *get_tracklet_from_object_list* function to split all the observations for an
        # object into individual tracklets
        # NB: lol = List-of-Lists ...
        o_lol, s_lol = get_tracklet_from_object_list(   object_list,
                                                        str_list,
                                                        t_crit_hrs  = t_crit_hrs,
                                                        min_trk_obs = min_trk_obs, )

        # If the returned object has sufficient tracklets, then we use the data for this object ...
        if len(o_lol) >= min_trklts:
            yield o_lol, s_lol
            



class SameAsPrevious:
    ''' Class/Functionality to facilitate "chunking" by remembering previous values
    
    Allows comoparison of "labels" (nuum/desig) by default
    Allows additional comparison of obsCode & times (for tracklets)
    
    '''
    def __init__(self):
    
        # Quantities we can use to assess "similarity"
        self.num   = None
        self.desig = None
        self.jdutc = None
        self.cod   = None

    def __call__(self, this, option,  t_crit_hrs = 8.0 ):
    
        # Check the supplied option
        assert option in ["Objects", "Tracklets"]

        # Compare "this" (a new observation) to previous observation
        if   option == "Objects":
            same = True if (self.num == this.num != '') or ( self.desig == this.desig != '') else False
            
        elif option == "Tracklets":
            same_cod    = True if self.cod == this.cod else False
            close_jdutc = True if isinstance(self.jdutc, float) and abs( this.jdutc - self.jdutc ) < t_crit_hrs/24. else False
            same = same_cod and close_jdutc
        else:
            print("Fuck")
            
        # Reset variables
        self.num , self.desig, self.jdutc, self.cod = \
        this.num, this.desig, this.jdutc, this.cod
        
        # Return comparison
        return same


