"""
Create some file paths
"""

"""
Create some file paths
"""

from pathlib import Path

# if it's not there create it.
parent_dir = Path.home().joinpath('repos','neo-tracklet-classifier')
data_dir = Path.joinpath(parent_dir, 'data')
