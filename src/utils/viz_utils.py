import seaborn as sns
import matplotlib.pyplot as plt


def ageGroups(params):
    if params <= 30:
        return "<30"
    elif (params > 30) & (params <= 40):
        return "30-40"
    elif (params > 40) & (params <= 50):
        return "40-50"
    elif (params > 50) & (params <= 60):
        return "50-60"
    else:
        return ">60"    


def visualize_agegroups(data, features):
    """
    Function to visualize agegroups
    parameter: data    : takes in a dataset of featurized columns
               features: takes in feature target for subgroupings
    returns a plot of distribution in different age subgroups
    """
    data["ageGroups"] = data["age"].apply(ageGroups)
    male = data[(data["gender"] == "male") & (data["version"] != "PD_Passive")]
    female = data[(data["gender"] == "female") & (data["version"] != "PD_Passive")]

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize= (15, 5))
    for ageGroup, color in zip(["<30", "30-40", "40-50", "50-60", ">60"], 
                                ["yellow", "orange", "purple", "blue", "red"]):
        sns.distplot(male[features][(male["ageGroups"] == ageGroup)], 
                     kde_kws={"shade": True}, hist = False, label = ageGroup,
                    ax = axes[0], color = color)
        axes[0].grid()
        axes[0].set_title("Male All AgeGroup Distribution")
        axes[0].legend()  
    for ageGroup, color in zip(["<30", "30-40", "40-50", "50-60", ">60"], 
                                ["yellow", "orange", "purple", "blue", "red"]):
        sns.distplot(female[features][(female["ageGroups"] == ageGroup)], 
                     kde_kws={"shade": True}, hist = False, label = ageGroup,
                    ax = axes[1], color = color)
        axes[1].grid()
        axes[1].set_title("Female All AgeGroup Distribution")
        axes[1].legend()

    fig.tight_layout()
    fig.show()
    
    
def visualize_groupComparisons(data, features):
    """
    function to plot distribution of control vs ms patient and pd patient
    ######## Parameters #################  
    data: dataset of featurized columns            
    features: the target feature for subgroupings

    ######## Returns ##################
    returns plots of subgroup of controls vs PD vs MS based on different PDKIT features
    """
    plt.figure(figsize = (10,5))
    sns.distplot(data[features][(data["is_control"] == 0)], kde_kws={"shade": True}, label = "Control", hist = False)
    sns.distplot(data[features][(data["PD"] == 1)], kde_kws={"shade": True}, label = "PD", hist = False)
    sns.distplot(data[features][(data["MS"] == 1)], kde_kws={"shade": True}, label = "MS", hist = False)
    # sns.distplot(data[features][(data["PD"] == 0)], kde_kws={"shade": True}, label = "NON-PD", hist = False)
    # sns.distplot(data[features][(data["MS"] == 0)], kde_kws={"shade": True}, label = "NON-MS", hist = False)
    plt.legend()
    plt.grid()
    plt.show()
    
    
def visualize_passive_active(data, features):
    """
    function to plot distribution of PD-active vs PD-passive
    parameter:  data: dataset of featurized columns
                features: the target feature for subgroupings
    returns plots of subgroup of PD_PASSIVE vs PD_ACTIVE
    """
    plt.figure(figsize = (10,5))
    sns.distplot(data[features][(data["PD"] == 1) & (data["version"] == "PD_passive")], kde_kws={"shade": True}, label = "PASSIVE-PD", hist = False)
    sns.distplot(data[features][(data["PD"] == 1) & (data["version"] == "V2")], kde_kws={"shade": True}, label = "ACTIVE-PD", hist = False)
    plt.legend()
    plt.grid()
    plt.show()