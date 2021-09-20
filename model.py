import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import ADASYN
import pickle
#load pickle model
dt = pickle.load(open('dt_smote1', 'rb'))
def predict_dt(df):
    predictions = dt.predict(df)
    df['predictions']=predictions
    return(df)
