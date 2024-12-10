import numpy as np

"""
Module Name: naive_bayes

Description
-----------

    This module does implement some naive_bayes algorithms, focusing mainly on
        text classification


Usage:
    Instructions on how to use the module or examples of typical use cases.

Attributes:
    - List any important constants or variables defined in the module.

Classes
-------

    - MultinomialNB: Text classification event model, that assumes features follow
        multinomial distribution, (features represent counts/frequencies)

    - GaussianNB: Text classification model that assumes features to follow gaussian 
        distribution (Highly effective for continuous features)
        
    - ComplementNB: Text classification model , that uses the complement of the
        data distribution, to compensates for imbalanced data sets 

Functions:
    - List and describe the main functions provided by the module.

Example:
    Provide a simple example of how to use the module.

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

class MultinomialNB:

    def __init__(self, ) -> None:
        ...

    # Defining the fit method
    def fit(self, x_train: ) -> None:
        ...
    
    # Defining the predict method
    def predict(self, x_test ) ->None:
        ...