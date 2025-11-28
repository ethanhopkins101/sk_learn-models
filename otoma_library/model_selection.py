import numpy as np
import random
import pandas as pd
from typing import Union, Sequence, Tuple,List, Optional

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float]]

def check_inputs(x: ArrayLike, y: ArrayLike, intercept: Optional[bool]= True) -> Tuple[ArrayLike, ArrayLike]:
    #check validity of x and y (try/except)
    intercept= np.ones((len(x),1))
    return np.hstack((intercept, np.array(x))), np.array(y)

def train_test_split(inputs: ArrayLike, target: ArrayLike
                     , random_state: int= 1, test_size: float= 0.3
                     , stratify: ArrayLike= None) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    inputs, target= check_inputs(inputs, target)
    seed: int= random.seed(random_state)
    indexes: List[int]= []
    ilen: int= len(inputs)
    if ilen % 2 == 0:
        indexes= list(range(ilen//2))
    else:
        indexes= list(range(ilen//2 + 1))
    indexes= random.sample(indexes, len(indexes))

    for i in indexes:
        try:
            tempx= inputs[indexes[i], :]
            tempy= target[indexes[i]]
            inputs[indexes[i], :]= inputs[-indexes[i],: ]
            target[indexes[i]]= target[-indexes[i]]
            inputs[-indexes[i], :]= tempx
            target[-indexes[i]]= tempy
        except:
            break
    size: int= int(test_size * len(inputs))
    return inputs[:size,:], inputs[size:,:], target[:size], target[size:]

class GridSearchCV:

    def __init__(self, estimator, param_grid, scoring, cv= 5) -> None:
        self.estimator= estimator
        self.param_grid= param_grid
        self.scoring= scoring
        self.cv: int= cv


    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x, y= check_inputs(x, y)
        eval= []
        inputs= x
        target= y
        for i in range(self.cv):
            
            x_train= inputs[:len(inputs)//self.cv, :]
            x_test= inputs[len(inputs)//self.cv :, :]
            y_train= target[: len(target)//self.cv]
            y_test= target[len(target)//self.cv :]
            
            model= self.estimator
            model.fit(x_train, y_train)
            y_pred= model.predict(x_test)
            eval.append(self.scoring(y_test, y_pred))
            
            inputs= np.vstack(x_test, x_train)
            target= np.vstack(y_test, y_train)

            