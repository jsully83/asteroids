# neo-tracklet-classifier

The NEO Tracklet Classifier is a machine learning algorithm for classifying tracklets as Near Earth Objects.  You can train a model with new data, search for optimal hyper-parameters for training and see metrics and plots from the training.  Once you have a trained model you can also run new tracklet data using the model.

Installation
This installation will guide you through installing conda and creating a new conda environment(you may use our own favorite virtual environment as well) and installing the required packages for using NEO Tracklet Classifier. 

You can find instructions for installing conda here
https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Once conda is installed create a new virtual environemnt. Open up a new terminal and type

    conda create -n neo-classifier python=3.10

When the environment is created activate the environment by typing

    conda activate neo-classifier

To use the NEO Tracklet Classifier you must install the MPC program Tracklet first.  Create a new directory:

    mkdir ~/neo-classifier
    cd ~/neo-classifier

Clone the Tracklet repository from github into the new directory and install the package:

    git clone https://github.com/Smithsonian/tracklet.git
    pip install ./tracklet 

Clone the obs80 repository from github into a new directory and install the package:

    git clone https://github.com/Smithsonian/obs80.git
    pip install ./obs80

Now clone and install the NEO Tracklet Classifier from github:

    git clone https://github.com/Smithsonian/neo-tracklet-classifier.git
    pip install ./neo-tracklet-classifier
    
To create the data that we need to run a model we need to run create_data.py

    python ~/neo-classifier/neo-tracklet-classifier/bin/create_data.py

After the data is created you can run a basic machine learning model

    python ~/neo-classifier/neo-tracklet-classifier/bin/run_model.py -f neo_tracklet_classifier
    
Now you can review the results of this run in a jupyter notebook.  

    jupyter-notebook ~/neo-classifier/neo-tracklet-classifier/jupyter_notebooks/results.ipynb
    
    
    
    





        
