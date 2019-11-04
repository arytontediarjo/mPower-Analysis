from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

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
        self.aggregation_type = aggregation_type
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X = X.groupby("healthCode").agg(self.aggregation_type)
        for feature in filter(lambda feature: ("." in feature), X.columns): 
            X  = X.rename({feature: "{}_{}"\
                            .format(self.aggregation_type.upper(), feature)}, axis = 1)
        return X


class addAdditionalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        ## number of steps that is based on resultant steps, not based on the resultant signals ##
        X["FC.no_of_steps"] = np.sqrt(X["x.no_of_steps"] ** 2 + \
                                        X["y.no_of_steps"] ** 2 + \
                                        X["z.no_of_steps"] ** 2)


        ## the speed of gait based on the resultant speed of gait of X,Y,and Z not based on the speed of 
        ## gait assessed from the resultant signals ##
        X["FC.speed_of_gait"] = np.sqrt(X["x.speed_of_gait"] ** 2 + \
                                        X["y.speed_of_gait"] ** 2 + \
                                        X["z.speed_of_gait"] ** 2)
        
        ## per second basis features as mPower previous versions have discrepancies in recording data ##
        X["FC.no_of_steps_per_sec"] = X["FC.no_of_steps"]/X["duration"]
        X["x.freeze_occurences_per_sec"] = X["x.freeze_occurences"]/X["duration"]
        X["y.freeze_occurences_per_sec"] = X["y.freeze_occurences"]/X["duration"]
        X["z.freeze_occurences_per_sec"] = X["z.freeze_occurences"]/X["duration"]
        X["AA.freeze_occurences_per_sec"] = X["AA.freeze_occurences"]/X["duration"]
        return X
    
def preprocess(X, is_feature_engineered):
    X = X.copy()
    if is_feature_engineered == False:
        X = collapseFeatures(aggregation_type = "max").transform(X)
        return X
    else:
        X = addAdditionalFeatures().transform(X)
        X = collapseFeatures(aggregation_type = "max").transform(X)
        X = logTransformer(variables = [feat for feat in X.columns if ("frequency_of_peaks" in feat)]).transform(X)
        X = dropFeatures(variables_to_drop = [feat for feat in X.columns if ("stride_regularity" in feat)]).transform(X)
        X = dropFeatures(variables_to_drop = ["MAX_x.freeze_occurences", 
                                              "MAX_y.freeze_occurences", 
                                              "MAX_z.freeze_occurences",
                                              "MAX_AA.freeze_occurences",
                                              "MAX_x.no_of_steps", 
                                              "MAX_y.no_of_steps", 
                                              "MAX_z.no_of_steps",
                                              "MAX_FC.no_of_steps",
                                              "MAX_AA.no_of_steps",
                                              "MAX_x.frequency_of_peaks", 
                                              "MAX_y.frequency_of_peaks", 
                                              "MAX_z.frequency_of_peaks",
                                              "MAX_AA.frequency_of_peaks",
                                              "duration"]).transform(X)
        return X
