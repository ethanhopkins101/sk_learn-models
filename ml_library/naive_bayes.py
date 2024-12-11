import numpy as np
from numpy.typing import ArrayLike
from matrix_operations import *
from feature_extration.text import IndexVectorizer
from typing import Optional
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

        self.probabilities_given_y0: Optional[List[float]]= None
        self.probabilities_given_y1: Optional[List[float]]= None
        self.probabilities_y0: Optional[float]= None
        self.probabilities_y1: Optional[float]= None

    def _indicator(x_train: MatrixLike | ArrayLike, key: str) -> tuple[int,int]:

        count: int= 0
        vectors_lengths: int= 0
        for i in range(len(x_train)):
            for j in range(x_train.shape[1]):
                if x_train[i,j] == super.dictionary[key]:
                    count+= 1
            vectors_lengths+= len(x_train[i]) #maybe cache it 

        return count, vectors_lengths
    
    def fit(self, x_train: MatrixLike | ArrayLike, y_train: MatrixLike | ArrayLike) -> None:

        self.probabilities_given_y0: List[float]= []
        self.probabilities_given_y1: List[float]= []
        # Applying the closed form probabilities
        for key in super.dictionary.keys():

            temp_y0: List[int]= list([self._indicator(x_train[np.where(y_train == 0)], key)])
            temp_y1: List[int]= list([self._indicator(x_train[np.where(y_train == 1)], key)])

            self.probabilities_given_y0.append((temp_y0[0] + 1) / (temp_y0[1] + len(super.dictionary)))
            self.probabilities_given_y1.append((temp_y1[0] + 1) / (temp_y1[1] + len(super.dictionary)))
            self.probabilities_y0: float= len(y_train[np.where(y_train == 0)]) / len(y_train)
            self.probabilities_y1: float= len(y_train[np.where(y_train == 1)]) / len(y_train)
    
    def predict(self, x_test: MatrixLike | ArrayLike) -> np.ndarray:

        if self.probabilities_y0 is None:
            raise ValueError('Fit method was not called , parameters are not trained yet.')
        
        predictions: List[float]= []
        for i in range(len(x_test)):
            # temporary variables to store probabilities
            temp0: List[float]= []
            temp1: List[float]= []
            for j in range(x_test.shape[1]):
                # storing probabilities while accounting for frequencies
                temp0.append(self.probabilities_given_y0[x_test[i,j]])
                temp1.append(self.probabilities_given_y1[x_test[i,j]])
            temp0= sum(np.log(temp0)) + np.log(self.probabilities_y0)
            temp1= sum(np.log(temp1)) + np.log(self.probabilities_y1)

            predictions.append(0 if temp0 > temp1 else 1)
        return np.array(predictions)
