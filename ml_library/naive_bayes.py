import numpy as np
from numpy.typing import ArrayLike
from matrix_operations import *
from feature_extration.text import IndexVectorizer
"""
Module Name: naive_bayes

Description
-----------

    This module does implement some naive_bayes algorithms, focusing mainly on
        text classification


Usage
-----

    Used to classify text related data into two labels (classification algorithms),
        based on given features

Attributes
----------

    - List any important constants or variables defined in the module.

Classes
-------

    - MultinomialNB: Text classification event model, that assumes features follow
        multinomial distribution, (features represent counts/frequencies)

    - GaussianNB: Text classification model that assumes features to follow gaussian 
        distribution (Highly effective for continuous features)

    - ComplementNB: Text classification model , that uses the complement of the
        data distribution, to compensates for imbalanced data sets 

Functions
---------

    - fit()
    - predict()

"""


# Defining the GaussianNB Class
class GaussianNB:
    def __init__(self) -> None: #initializer
        ...

    def fit(self,x_train)

# Defining the MultivariateBernoulliNB Class
class MultivariateBernoulliNB:
    
    def __init__(self) -> None: #Initializer
        ...

    # Defining the fit method
    def fit(self, x_train: ) -> None:
        ...
    
    # Defining the predict method
    def predict(self, x_test ) ->None:
        ...


class MultinomialNB (IndexVectorizer):

    def __init__(self) -> None:

        self.probabilities_y0= None
        self.probabilities_y1= None
    
    def fit(self, x_train: MatrixLike | ArrayLike) -> None:

        self.probabilities_y0= []
        self.probabilities_y1= []
        for k in super.dictionary()
        
    
    
    def predict(self, x_test ) ->None:
        ...