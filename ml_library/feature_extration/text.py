import numpy as np
from matrix_operations import *
from numpy.typing import ArrayLike
"""
text.py

This Submodule includes classes and methods for common text
preprocessing and vectorization tasks

Classes
-------

CountVectorizer: Converts a collection of text documents to a matrix of token counts.
TfidfVectorizer: Converts text to a matrix of TF-IDF features.
TfidfTransformer: Transforms a count matrix to a normalized TF-IDF representation.
HashingVectorizer: A vectorizer that uses the hashing trick for efficient,
                    fixed-dimension representation.
"""
class CountVectorizer:
    
    def __init__(self)-> None : #Initializer
        self.dictionary=None

    # Defining the fit method
    def fit(self, x_train: MatrixLike | ArrayLike) -> None:
        temp=[] # collects all rows into one big list
        x_train=change_type(x_train) # setting x_train to np.array incase

        for i in range(x_train.shape[0]):
            try :
                temp.append(x_train[i][0])
            except :
                try :
                    temp.append(x_train[i])
                except : 
                    raise ValueError('The input Training data does not fit the requirement shape')
        
        combined_content=' '.join(temp) # combines all strings in the list into one string

        content_words=combined_content.split(' ') # splits content by words

        content_words=set(content_words) # removes all duplicates 
        
        self.dictionary={k:v for v,k in enumerate(content_words)}
    
    # Defining the Transform method
    def transform()->np.ndarray:
        ...

    # Defining the fit_transform method
    def fit_transform()->np.ndarray:
        ...