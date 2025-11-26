from typing import Union, Sequence, Optional, Literal, Tuple, List
import numpy as np
import pandas as pd
import re

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float]]

def check_inputs(x: ArrayLike) -> np.ndarray:
    #check validity of x and y (try/except)
    return np.array(x)

def clean_msg(x: ArrayLike) -> list[str]:
    
    temp= x.copy()
    temp= temp.translate(str.maketrans('', '', ',!?;.:^$*&=)àç("')) #clean
    temp= temp.split(' ') #list of strings
    
    return temp


class CountVectorizer:

    def __init__(self) -> None:
        
        self.dictionary: dict[str, int]= None

    def fit(self, x) -> None:

        x= check_inputs(x)
        temp: List[str]= []
        for i in range(len(x)):
            temp.append(x[i]) #appending all messages together

        temp= ' '.join(temp) #long string
        temp = temp.translate(str.maketrans('', '', ',!?;.:^$*&=)àç("')) #cleaning the string
        temp= list(set(temp.split(' '))) #converts string to list then set(to remove dup) then list
        
        self.dictionary= {k: v for v,k in enumerate(temp)}

    def fit_transform(self, x) -> ArrayLike:

        #----------------creating dictionary-----------------------------------
        x= check_inputs(x)
        temp: List[str]= []
        for i in range(len(x)):
            temp.append(x[i]) #appending all messages together

        temp= ' '.join(temp) #long string
        temp = temp.translate(str.maketrans('', '', ',!?;.:^$*&=)àç("')) #cleaning the string
        temp= list(set(temp.split(' '))) #converts string to list then set(to remove dup) then list
        
        self.dictionary= {k: v for v,k in enumerate(temp)}
        #------------------------Transform--------------------------------------
        temp_row= np.zeros(len(self.dictionary))
        result_matrix= temp_row.copy()
        for i in range(len(x)):
            
            clean= clean_msg(x[i])
            for j in clean:
                if j in self.dictionary.keys():
                    temp_row[self.dictionary[j]]+= 1
            
            result_matrix= np.vstack((result_matrix, temp_row))
            temp_row= temp_row * 0
        result_matrix= result_matrix[1:, :]

        return result_matrix


    def transform(self, x) -> ArrayLike:
        
        if self.dictionary is None:
            raise ValueError('Method fit() was not called')
        temp_row= np.zeros(len(self.dictionary))
        result_matrix= temp_row.copy()
        for i in range(len(x)):
            
            clean= clean_msg(x[i])
            for j in clean:
                if j in self.dictionary.keys():
                    temp_row[self.dictionary[j]]+= 1
            
            result_matrix= np.vstack((result_matrix, temp_row))
            temp_row= temp_row * 0
        result_matrix= result_matrix[1:, :]

        return result_matrix



    def toarray(self) -> np.ndarray:
        ...