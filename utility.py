import datetime
import json
import os
import pickle

import numpy as np
import pandas as pd


def use_dummies(
    df,
    x_features = ['Quantity','Exchange','Side','News'],
    dummy_features = ['Exchange','Side','News'],
    ):
    X = pd.get_dummies(df[x_features], columns= dummy_features)
    return X

def pickle_obj(obj, file_path = None):
    if file_path is not None:
        with open(file_path, 'wb') as f:
            pickle.dump(obj,f)


def read_obj(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def msg_to_dict(msg, dtypes = {'OrderID':int,'Exchange':int,'Quantity':int,'News':int,'Price':float}):
    
    if isinstance(msg,str):
        d = json.loads(msg)
        new_d = {}
        for key, val in d.items():
            if key in dtypes:
                new_d[key] = dtypes[key](val)
            else:
                new_d[key] = val
        return new_d
    elif isinstance(msg,dict):
        return msg
    else:
        raise TypeError('Unacceptable Type: '+ str(type(msg)))


def prepare_X(trade_message, model):
    X = pd.DataFrame(np.nan, index = [0], columns = model.feature_names_in_)


    transaction_df = pd.Series(msg_to_dict(trade_message)).to_frame().T
    return X.fillna(use_dummies(transaction_df)).fillna(0)

