import numpy as np
from matrix_operations import *
from numpy.typing import ArrayLike
from typing import List
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

        content_words= combined_content.split(' ') # splits content by words

        content_words= set(content_words) # removes all duplicates 
        # filling the dictionary with obtained words , with values representing current index
        self.dictionary= {v:k for k,v in enumerate(content_words)}
    
    # Defining the Transform method
    def transform(self, x_transform: MatrixLike | ArrayLike) -> np.ndarray:

        if x_transform.shape[1] > 1: 
            raise ValueError ('The given inputs do not satisfy the required conditions')
            
        x_transform= change_type(x_transform) # changing x_transform type to ndarray incase
        
        # Supposed transformation of (x_transform)
        holder_matrix= np.zeros((x_transform.shape[0],len(self.dictionary)))

        for i in range(x_transform.shape[0]):
            words= x_transform[i].split(' ')
            words= set(words) # removes duplicates
            
            for word in words :
                if word in self.dictionary.keys():
                    holder_matrix[i,self.dictionary[word]]= 1

        return holder_matrix

    # Defining the fit_transform method
    def fit_transform(self, x_transform: MatrixLike | ArrayLike) -> np.ndarray:
        self.fit(x_transform)
        return self.transform(x_transform)
    
    # Defining the 'toarray' method
    def toarray():
        ...

    # Defining the get_feature_names_out method
    def get_feature_names_out(self) -> List[str]:
        return self.dictionary.keys()
    
class IndexVectorizer:
    
    def __init__(self) -> None:
        self.dictionary=None

    def fit(self, x_train: MatrixLike | ArrayLike) -> None:

        if x_train.shape[1]> 1:
            raise ValueError ('Invalid input shape')
        
        x_train=change_type(x_train)
        temp= [] # temporary dictionary
        for i in range(len(x_train)):
            temp.extend(x_train[i])
        
        combined= ' '.join(temp) # one big email
        combined= combined.split(' ') # split to words
        combined= set(combined)

        self.dictionary= {v:k for k,v in enumerate(combined)}
