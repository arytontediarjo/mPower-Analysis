## required dependencies ##
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import svm, decomposition, tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn.model_selection import learning_curve, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from synapseclient import Entity, Project, Folder, File, Link
import synapseclient as sc
import time

warnings.simplefilter("ignore")
np.random.seed(100)


def logreg_fit(X_train, y_train):
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state = 100, n_jobs = -1))
        ])
    param = [{'classifier__penalty': ['l2'], 
              'classifier__solver': [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}, 
             {'classifier__penalty': ['l1'], 
              'classifier__solver': [ 'liblinear', 'saga']}  
            ]

    CV = GridSearchCV(estimator = pipe, param_grid = param , scoring= "roc_auc", n_jobs = -1, cv = 10)
    CV.fit(X_train, y_train)
    return CV



def xgb_fit(X_train, y_train):
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(seed = 100, nthread = -1))
        ])
    param = {
        "classifier__learning_rate" : [0.01, 0.05, 0.1],
        "classifier__tree_method"   : ["hist", "auto"],
        "classifier__max_depth"     : [3, 6, 8],
        "classifier__gamma"         : [1],
        "classifier__subsample"     : [0.8],
        "classifier__n_estimators"  : [100]
    }
    CV = GridSearchCV(estimator = pipe, param_grid = param , scoring= "roc_auc", cv = 10)
    CV.fit(X_train, y_train)
    return CV
    

def gradientboost_fit(X_train, y_train):
    # pca = decomposition.PCA()
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state = 100))
        ])
    param = {
        'classifier__learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        'classifier__max_depth':[1, 2, 3, 4, 5, 6],
        'classifier__loss': ["deviance", "exponential"], ## exponential will result in adaBoost
        "classifier__n_estimators"  : [100]
    }
    CV = GridSearchCV(estimator = pipe, param_grid = param , scoring= "roc_auc", n_jobs = -1, cv = 10)
    CV.fit(X_train, y_train)
    return CV

def randomforest_fit(X_train, y_train):
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state = 100, n_jobs = -1))
        ])
    param = {
        'classifier__max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__criterion': ["gini", "entropy"],## exponential will result in adaBoost
        'classifier__max_features': ["auto", "sqrt", "log2", None], 
        'classifier__n_estimators'  : [100, 200]
    }
    CV = GridSearchCV(estimator = pipe, param_grid = param , scoring= "roc_auc", n_jobs = -1, cv = 10)
    CV.fit(X_train, y_train)
    return CV


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
    return pred_result_dict
    
def main():
    
    # split training and test # 
    walking_train = pd.read_csv("../Data/walking_data_training.csv", index_col=0)
    # balance_train = pd.read_csv("../Data/balance_data_training.csv", index_col=0).dropna()
    # balance_X_train, balance_X_test, balance_y_train, balance_y_test = train_test_split(balance_train.drop(["healthCode", "PD"], axis = 1), balance_train["PD"], test_size=0.20, random_state = 100)
    walking_X_train, walking_X_test, walking_y_train, walking_y_test = train_test_split(walking_train.drop(["healthCode", "PD"], axis = 1), walking_train["PD"], test_size=0.20, random_state = 100)

    # model #
    lr_walking_model = logreg_fit(walking_X_train, walking_y_train)
    rf_walking_model = randomforest_fit(walking_X_train, walking_y_train)
    gb_walking_model = gradientboost_fit(walking_X_train, walking_y_train)
    xgb_walking_model = xgb_fit(walking_X_train, walking_y_train)
    models = [(lr_walking_model, "LOGISTIC REGRESSION"),
              (rf_walking_model, "RANDOM FOREST"),
              (gb_walking_model, "SKLEARN GRADIENT BOOSTING"),
              (xgb_walking_model, "XTREME GRADIENT BOOSTING")]
    predictions = savePerformances(models, walking_X_test, walking_y_test)
    
    ## save to synapse file ##
    syn = sc.login()
    file_path = "../Data/prediction_results.csv"
    data = pd.DataFrame.from_dict(predictions).to_csv(file_path)
    data = File(path = file_path, parentId = "syn20816722")
    data = syn.store(data)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    
    

    
    