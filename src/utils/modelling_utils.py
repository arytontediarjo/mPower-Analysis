from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import pandas as pd
import numpy as np
import scipy.stats as stats

def create_pc_dataframe(data, target_df, n_components = 2):
    """
    function to create pca dataframe 
    concatenate each dataframe with each of the principal components and
    designated target variables

    parameters: 
    `data`         : dataframe
    `target_df`    : target variables dataframe (could be version, PD status etc)
    `n_components` : number of principal components

    Returns dataframe of principal components and target variables
    """
    
    scaler         = StandardScaler()
    X_scaled       = scaler.fit_transform(data)
    pca            = PCA().fit(X_scaled)
    pca            = PCA(n_components = n_components)
    principal_comp = pca.fit_transform(X_scaled)
    principal_df   = pd.DataFrame(data = principal_comp)
    principal_df   = pd.concat([principal_df, target_df], axis = 1)
    return principal_df

def one_hot_encode(data, *target_vars):
    """
    Function to one-hot encode dataframe
    it will encode factorial and categorical variables (phonetype, gender and versions)
    returns the encoded dataset, and the result of the encoded columns
    
    parameters:
    `data`         : dataframe
    `target_vars`  : list of target variables (categorical variables)

    Returns one-hot encoded dataset
    """
    metadata_df     = data
    for feature in target_vars:
        metadata_df = pd.concat([metadata_df, pd.get_dummies(metadata_df[feature], prefix = "OHE_is", drop_first = True)], axis = 1) 
    return metadata_df


## larssson function ## 
def QaD_correlation(values, classes, isFactor=None):
    """Given a set of values or class assignments determines correlation/enrichment 
    in list of classificiations.
    Uses, correlation or fisher tests
    
    Arguments:
    - `values`:  Vector of values (either numeric or factors)
    - `classes`:  a two dimensional array or pandas DataFrame of (either numeric or factors)
    - `isFactor`: list of same length as classes of T/F indicating weather each 
                  class is a factor or continuous variable (defaults to None) meaning
                  strings are used factors and numbers as continuous
    Returns a list of p-values one for each classification in classes
    """
    isFactor = [None]*len(classes) if isFactor==None else isFactor
    pVals = []
    classes = pd.DataFrame(classes)
    for i, key in enumerate(classes):
        classification = classes[key]
        #If it is a factor perform ANOVA across groups
        if ((classification.dtype in [np.string_,  np.object_, np.bool_, np.unicode_]) or 
            isFactor[i]):
            groupLabels = list(set(classification.dropna()))
            groups = [values[np.where(classification==l)] for l in groupLabels]
            f_val, p_val = stats.f_oneway(*groups) 
            pVals.append(p_val)
        else:
            m, b, r_val, p_val, slope_std_error = stats.linregress(values, classification)
            pVals.append(p_val)
    pVals = np.asarray(pVals)
    pVals[np.isnan(pVals)] = np.inf
    series = pd.Series(data = pVals, index = classes.columns)
    return series

def run_QaD_classification(data, target, test_split_percentage):
    ## split datasets ## 
    X_train, X_test, y_train, y_test = \
        train_test_split(data, 
                        data[target], 
                        test_size    = test_split_percentage, 
                        random_state = 100)

    gb_model = GradientBoostingClassifier()
    return 

