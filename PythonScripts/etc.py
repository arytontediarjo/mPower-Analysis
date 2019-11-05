import utils
import ML_utils
import synapseclient as sc
import pandas as pd

syn = sc.login()
data = syn.get("syn21046180")
data = pd.read_csv(data["path"], index_col = 0)

data = ML_utils.preprocess(data, is_feature_engineered = True)


utils.save_to_synapse(data, 
                      __file__,
                      "syn21046180",
                      "syn21118104",
                      "syn20987850",
                      "walking_model_data_v1.csv"
                      )