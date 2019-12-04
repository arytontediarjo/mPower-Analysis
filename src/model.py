## imports ##
import sys
import time
from utils import preprocessing_utils as preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import svm, decomposition, tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn.model_selection import learning_curve, GridSearchCV, cross_val_score, validation_curve
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import RFECV, SelectKBest, chi2, SelectFromModel, RFE
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import synapseclient as sc
# from sklearn.base import BaseEstimator, TransformerMixin

warnings.simplefilter("ignore")
np.random.seed(100)


syn = sc.login()

def logreg_fit(X_train, y_train):
    pipe = Pipeline(steps=[
        ("feature_selection", SelectFromModel(ExtraTreesClassifier(n_estimators = 300,
                                                                   random_state  = 100))),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state = 100))
        ])
    param = [{'feature_selection__threshold' : ["1.1*mean", "1.2*mean", "mean"], 
                'classifier__penalty': ['l2'], 
                'classifier__solver': [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}, 
             {'feature_selection__threshold' : ["1.1*mean", "1.2*mean", "mean"], 
                'classifier__penalty': ['l1'], 
                'classifier__solver': [ 'liblinear', 'saga']}]
    CV = GridSearchCV(estimator = pipe, param_grid = param , 
                      scoring= "roc_auc", n_jobs = -1, cv = 10, verbose = True)
    CV.fit(X_train, y_train)
    return CV


def xgb_fit(X_train, y_train):
    pipe = Pipeline(steps=[
        ("feature_selection", SelectFromModel(XGBClassifier(n_estimators = 300,
                                                            random_state  = 100))),
        ('classifier', XGBClassifier(seed = 100))
        ])
    param = {
        'feature_selection__threshold' : ["1.1*mean", "1.2*mean", "mean"], 
        "classifier__learning_rate" : [0.005, 0.01, 0.05, 0.1],
        "classifier__tree_method"   : ["hist", "auto"],
        "classifier__max_depth"     : [3, 6, 8, 10],
        "classifier__gamma"         : [0, 1],
        "classifier__colsample_bytree": [0.8, 0.9, 0.1],
        "classifier__subsample"     : [0.8, 0.9, 1],
        "classifier__n_estimators"  : [100, 200, 300]
    }
    CV = GridSearchCV(estimator = pipe, param_grid = param, 
                      scoring= "roc_auc", n_jobs = -1, cv = 10, verbose = True)
    CV.fit(X_train, y_train)
    return CV
    

def gradientboost_fit(X_train, y_train):
    pipe = Pipeline(steps=[
        ("feature_selection", SelectFromModel(estimator = XGBClassifier(n_estimators = 300,
                                                                              random_state  = 100))),
        ('classifier', GradientBoostingClassifier(random_state = 100))
        ])
    param = {
        'feature_selection__threshold' : ["1.1*mean", "1.2*mean", "mean"],
        'classifier__learning_rate': [0.005, 0.01, 0.05, 0.1],
        'classifier__max_depth':[3, 6, 8, 10, 12],
        'classifier__loss': ["deviance", "exponential"], ## exponential will result in adaBoost
        "classifier__n_estimators"  : [100, 200, 300]
    }
    CV = GridSearchCV(estimator = pipe, param_grid = param , 
                      scoring= "roc_auc", n_jobs = -1, cv = 10, verbose = True)
    CV.fit(X_train, y_train)
    return CV

def randomforest_fit(X_train, y_train):
    pipe = Pipeline(steps=[
        ("feature_selection", SelectFromModel(estimator = ExtraTreesClassifier(n_estimators = 100,
                                                                              random_state = 100))),
        ('classifier', RandomForestClassifier(random_state = 100))
    ])
    param = {
        'feature_selection__threshold' : ["1.1*mean", "1.2*mean", "mean"], 
        'classifier__max_depth':[3, 6, 8, 10, 12],
        'classifier__criterion': ["gini", "entropy"],## exponential will result in adaBoost
        'classifier__max_features': ["auto", "sqrt", "log2", None], 
        'classifier__n_estimators'  : [100, 200, 300]
    }
    CV = GridSearchCV(estimator = pipe, param_grid = param, 
                      scoring= "roc_auc", n_jobs = -1, cv = 10, verbose = True)
    CV.fit(X_train, y_train)
    return CV


def printPerformance(model, X_test, y_test):
    print("Mean AUC score on K-folds: {}".format(model.best_score_))
    print("Parameter Used: {}".format(model.best_params_))
    y_true, y_pred = y_test, model.predict(X_test)
    print("ROC-AUC on Test-Set: {}".format(metrics.roc_auc_score(y_true, y_pred)))
    print("log-loss: {}".format(metrics.log_loss(y_true, y_pred)))
    print("Precision: {}".format(metrics.precision_score(y_true, y_pred)))
    print("Recall: {}".format(metrics.recall_score(y_true, y_pred)))
    print("F1-Score: {}".format(metrics.f1_score(y_true, y_pred)))
def savePerformances(models, X_test, y_test):
    pred_result_dict = {}
    pred_result_dict["MODEL"] = []
    pred_result_dict["BEST_PARAMS"] = []
    pred_result_dict["CV_ROC_AUC"] = []
    pred_result_dict["TEST_ROC_AUC"] = []
    pred_result_dict["LOG_LOSS"] = []
    pred_result_dict["PRECISION"] = []
    pred_result_dict["RECALL"] = []
    pred_result_dict["F1_SCORE"] = []
    for model_tuple in models:
        model, model_name = model_tuple[0], model_tuple[1]
        y_true, y_pred = y_test, model.predict(X_test)
        pred_result_dict["MODEL"].append(model_name)
        pred_result_dict["BEST_PARAMS"].append(model.best_params_)
        pred_result_dict["CV_ROC_AUC"].append(model.best_score_)
        pred_result_dict["TEST_ROC_AUC"].append(metrics.roc_auc_score(y_true, y_pred))
        pred_result_dict["LOG_LOSS"].append(metrics.log_loss(y_true, y_pred))
        pred_result_dict["PRECISION"].append(metrics.precision_score(y_true, y_pred))
        pred_result_dict["RECALL"].append(metrics.recall_score(y_true, y_pred))
        pred_result_dict["F1_SCORE"].append(metrics.f1_score(y_true, y_pred))
        
        # persist models #
        pkl_filename = "{}.pkl".format(model_name)
        joblib.dump(model.best_estimator_, pkl_filename)
        print("persisted {} model on cd directory".format(model_name)) 
    return pred_result_dict
    
def main():
    ## split training and test ##
    entity = syn.get("syn21046180")
    data   = pd.read_csv(entity["path"], index_col = 0)
    
    ## preprocess dataset using sklearn base estimator ##
    data = preprocess(data, "max", is_feature_engineered = True)
    
    
    ## feature columns ##
    feature_columns = [feat for feat in data.columns if ("." in feat) 
                   or ("duration" in feat) or ("healthCode" in feat)]
    
    ## split datasets ## 
    walking_X_train, walking_X_test, walking_y_train, walking_y_test = \
            train_test_split(data[feature_columns], 
                             data["PD"], test_size=0.25, random_state = 100)

    ## models ##
    lr_walking_model = logreg_fit(walking_X_train, walking_y_train)
    rf_walking_model = randomforest_fit(walking_X_train, walking_y_train)
    gb_walking_model = gradientboost_fit(walking_X_train, walking_y_train)
    xgb_walking_model = xgb_fit(walking_X_train, walking_y_train)
    
    
    print("\n### Gradient Boosting Walking ###")
    printPerformance(gb_walking_model, walking_X_test, walking_y_test)
    print("\n### XTreme Gradient Boosting Walking ###")
    printPerformance(xgb_walking_model, walking_X_test, walking_y_test)
    print("\n### Random Forest Walking ###")
    printPerformance(rf_walking_model, walking_X_test, walking_y_test)
    print("\n### Logistic Regression Walking ###")
    printPerformance(lr_walking_model, walking_X_test, walking_y_test)
    
    # models = [(lr_walking_model, "LOGISTIC_REGRESSION"),
    #           (rf_walking_model, "RANDOM_FOREST"),
    #           (gb_walking_model, "SKLEARN_GRADIENT BOOSTING"),
    #           (xgb_walking_model, "XTREME_GRADIENT BOOSTING")]
    # predictions = savePerformances(models, walking_X_test, walking_y_test)
    
    # ## save to synapse file ##
    # syn = sc.login()
    # file_path = "../Data/prediction_results.csv"
    # data = pd.DataFrame.from_dict(predictions).to_csv(file_path)
    # data = File(path = file_path, parentId = "syn20816722")
    # data = syn.store(data)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    
    

    
    
