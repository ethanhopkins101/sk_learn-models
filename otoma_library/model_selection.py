import numpy as np
import random
import pandas as pd
from typing import Union, Sequence, Tuple,List

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float]]

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
    size= int(test_size * len(inputs))
    return inputs[:size,:], inputs[size:,:], target[:size], target[size:]