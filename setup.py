"""
neo-tracklet-classifier
"""
from setuptools import setup, find_packages

# Setting up
setup(
    name="neo_tracklet_classifier",
    version='0.1',
    author="Jonathan Sullivan",
    author_email="<jesully83@gmail.com>",
    description='Machine Learning NEO Classifier',
    packages=find_packages(),
    install_requires=['pandas',
                      'tensorflow-macos==2.10; platform_machine == "arm64"',
                      'tensorflow; platform_machine != "arm64"',
                      'tensorflow-datasets',
                      'tensorboard',
                      'shap',
                      'jupyter',
                      'matplotlib',
                      'numpy',
                      'plotly',
                      'scikit-learn',
                      'keras',
                      'keras-tuner',
                      'beepy',
                      'seaborn']
)
