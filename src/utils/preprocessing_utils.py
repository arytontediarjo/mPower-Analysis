from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd



""""
class for log transformation
returns if value is lower than 1 or zero, 
it will return zero instead of negative infinity
"""
class logTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transformer."""
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X["%s_log"%feature] = np.where(X[feature] < 1, 0, np.log(X[feature]))
        return X

"""
class for dropping features
returns a transformed dataframe with features dropped
"""  
class dropFeatures(BaseEstimator, TransformerMixin):
    
    """
    Feature selector transformer
    """
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X

   
class collapseFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, aggregation_type = None):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        record_count = ((X.groupby("healthCode").count()["recordId"]).rename("nrecords").reset_index())
        
        ## take most recent date on metadata
        metadata = X[['MS', 'PD', 'age', 'appVersion', 'createdOn', 
                    'gender', 'healthCode', 'phoneInfo', 'recordId', 
                    'version', 'duration', "is_control", "class"]].sort_values("createdOn", ascending = True).\
                        drop_duplicates(subset = "healthCode", keep = "last")
        feat_columns = [feat for feat in X.columns if ("." in feat) or ("healthCode" in feat) or ("version" in feat)]

        ## groupby the features based on aggregation given by user
        X = X[feat_columns].groupby("healthCode").agg(["mean", "median", "max", "std", iqr, q25, q75, valrange, kurtosis, skew]).fillna(0)
        new_cols = []
        for feat, agg in X.columns:
            new_cols_name = "{}_{}".format(agg, feat)
            new_cols.append(new_cols_name)
        X.columns = new_cols

        # for feature in filter(lambda feature: ("." in feature), X.columns): 
        #     X  = X.rename({feature: "{}_{}".format(self.aggregation_type.upper(), feature)}, axis = 1)
        ## merge the dataset with metadata and nrecords
        X = pd.merge(X, metadata, on = "healthCode", how = "left").reset_index(drop = True)
        X = pd.merge(X, record_count, on = "healthCode", how = "left").reset_index(drop = True)
        return X


class addFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        
        ## log transform the frequency of peaks 
        X = logTransformer(variables = [feat for feat in X.columns if ("frequency_of_peaks" in feat) and ("log" not in feat)]).transform(X)
        
        ## number of steps that is based on resultant steps, not based on the resultant signals ##
        X["FC.Number_of_steps"] = np.sqrt(X["x.no_of_steps"] ** 2 + \
                                        X["y.no_of_steps"] ** 2 + \
                                        X["z.no_of_steps"] ** 2)

        ## the speed of gait based on the resultant speed of gait of X,Y,and Z not based on the speed of 
        ## gait assessed from the resultant signals ##
        X["FC.Speed_of_gait"] = np.sqrt(X["x.speed_of_gait"] ** 2 + X["y.speed_of_gait"] ** 2 + X["z.speed_of_gait"] ** 2)
        
        ## per second basis features as mPower previous versions have discrepancies in recording data ##
        X["FC.Number_of_steps_per_seconds"] = X["FC.Number_of_steps"]/X["duration"]
        X["FC.Gait_freezes_per_seconds"] = (np.sqrt(X["x.freeze_occurences"] ** 2 + \
                                                X["y.freeze_occurences"] ** 2 + \
                                                X["z.freeze_occurences"] ** 2))/X["duration"]
        X["FC.Max_freeze_index"] = (np.sqrt(X["x.max_freeze_index"] ** 2 + X["y.max_freeze_index"] ** 2 + X["z.max_freeze_index"] ** 2))
        X = logTransformer(variables = ["FC.Max_freeze_index"]).transform(X)
        X["x.freeze_occurences_per_sec"] = X["x.freeze_occurences"]/X["duration"]
        X["y.freeze_occurences_per_sec"] = X["y.freeze_occurences"]/X["duration"]
        X["z.freeze_occurences_per_sec"] = X["z.freeze_occurences"]/X["duration"]
        X["AA.freeze_occurences_per_sec"] = X["AA.freeze_occurences"]/X["duration"]
        return X
    
def preprocess(X, aggregator, is_feature_engineered):
    X = X.copy()
    if is_feature_engineered == False:
        X = collapseFeatures(aggregation_type = aggregator).transform(X)
        return X
    else:
        X = addAdditionalFeatures().transform(X)
        X = collapseFeatures(aggregation_type = aggregator).transform(X)
        X = logTransformer(variables = [feat for feat in X.columns if ("frequency_of_peaks" in feat)]).transform(X)
        X = dropFeatures(variables_to_drop = [feat for feat in X.columns if ("stride_regularity" in feat)]).transform(X)
        X = dropFeatures(variables_to_drop = ["%s_x.freeze_occurences" %aggregator.upper(), 
                                              "%s_y.freeze_occurences" %aggregator.upper(), 
                                              "%s_z.freeze_occurences" %aggregator.upper(),
                                              "%s_AA.freeze_occurences" %aggregator.upper(),
                                              "%s_x.no_of_steps" %aggregator.upper(), 
                                              "%s_y.no_of_steps" %aggregator.upper(), 
                                              "%s_z.no_of_steps" %aggregator.upper(),
                                              "%s_FC.no_of_steps" %aggregator.upper(),
                                              "%s_AA.no_of_steps" %aggregator.upper(),
                                              "%s_x.frequency_of_peaks" %aggregator.upper(), 
                                              "%s_y.frequency_of_peaks" %aggregator.upper(), 
                                              "%s_z.frequency_of_peaks" %aggregator.upper(),
                                              "%s_AA.frequency_of_peaks" %aggregator.upper(),
                                              ]).transform(X)
        return X


def iqr(x):
    return q75(x) - q25(x)
def q25(x):
    return x.quantile(0.25)
def q75(x):
    return x.quantile(0.75)
def valrange(x):
    return x.max() - x.min()
def kurtosis(x):
    return x.kurt()
def skew(x):
    return x.skew()