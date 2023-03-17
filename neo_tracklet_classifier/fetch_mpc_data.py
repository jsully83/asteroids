'''
Some of the MPC flat files are big: too big to be handled by git/git-lfs (>2Gb)

So I am writing code to automate their retreival
'''

# standard imports 

import os
import urllib.request
from neo_tracklet_classifier import directory
from dataclasses import dataclass
from pathlib import Path

# convenience function to get all MPC data
def get_all_MPC_data():
    for label, Source in mpc_data.items():
        Source.doit()
    
# set up a data-class to deal with obs80 downloads
@dataclass
class MPCSource:
    """Class for retreiving MPC data files."""
    name:         str
    url:          str
    gz_filename:  str
    filename:     str

    data_dir:     str = directory.data_dir

    def get_filepath(self) -> str:
        return Path.joinpath( directory.data_dir , self.filename )
    def get_gz_filepath(self) -> str:
        return Path.joinpath( self.data_dir , self.gz_filename )

    def fetch(self) -> None:
        '''  Fetch the file from the MPC and store in local data directory '''
        print("Fetching data from the MPC...", self.url)
        urllib.request.urlretrieve( self.url, self.get_gz_filepath() )

    def unzip(self) -> None:
        ''' Unzip the zipped file '''
        print("Unzipping data ...", self.gz_filename )
        if Path.is_file( self.get_gz_filepath() ):
            os.system( f"gunzip {self.get_gz_filepath()}" )
        
    def doit(self) -> None:
        ''' Convenience function to do it all ... '''
        print("Downloading data: this may take a long time ...")
        self.fetch()
        self.unzip()


# define specific instances of MPCSource dataclass
mpc_data = {
    'UnnObs' : MPCSource( "UnnObs",
                            "https://www.minorplanetcenter.net/iau/ECS/MPCAT-OBS/UnnObs.txt.gz",
                            "UnnObs.txt.gz",
                            "UnnObs.txt"),

    'NumObs' : MPCSource( "NumObs",
                            "https://www.minorplanetcenter.net/iau/ECS/MPCAT-OBS/NumObs.txt.gz",
                            "NumObs.txt.gz",
                            "NumObs.txt"),
                            
    'ITFObs' : MPCSource( "ITFObs",
                            "https://www.minorplanetcenter.net/iau/ITF/itf.txt.gz",
                            "ITFObs.txt.gz",
                            "ITFObs.txt"),
                            
    'NEAOrbits' : MPCSource( "NEAOrbits",
                            "https://minorplanetcenter.net/Extended_Files/nea_extended.dat.gz",
                            "NEAOrbits.dat.gz",
                            "NEAOrbits.dat"),
                            
    'MPCOrbits' : MPCSource( "MPCOrbits",
                            "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz",
                            "MPCOrbits.dat.gz",
                            "MPCOrbits.dat"),
}

