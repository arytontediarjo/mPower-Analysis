import seaborn as sns
import matplotlib.pyplot as plt


## helper functions ## 
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
    
    data["ageGroups"] = data["age"].apply(ageGroups)
    
    male = data[(data["gender"] == "male") & (data["version"] != "Passive")]
    female = data[(data["gender"] == "female") & (data["version"] != "Passive")]

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize= (15, 5))
    
    # sns.distplot(male[features][(male["ageGroups"] == "<30")], kde_kws={"shade": True},
    #                  hist = False, ax = axes[0,0], color = "yellow")
    # axes[0,0].grid()
    # axes[0,0].set_title("Male 20-30 Years Old")

    # sns.distplot(male[features][(male["ageGroups"] == "30-40")], kde_kws={"shade": True}, hist = False,
    #                  ax = axes[1,0], color = "orange")
    # axes[1,0].grid()
    # axes[1,0].set_title("Male 30-40 Years Old")

    # sns.distplot(male[features][(male["ageGroups"] == "40-50")], kde_kws={"shade": True}, hist = False,
    #                  ax = axes[2,0], color = "purple")
    # axes[2,0].grid()
    # axes[2,0].set_title("Male 40-50 Years Old")


    # sns.distplot(male[features][(male["ageGroups"] == "50-60")], kde_kws={"shade": True}, hist = False,
    #                  ax = axes[3,0], color = "blue")
    # axes[3,0].grid()
    # axes[3,0].set_title("Male 50-60 Years Old")
    
    
    # sns.distplot(male[features][(male["ageGroups"] == ">60")], kde_kws={"shade": True}, hist = False,
    #                  ax = axes[4,0], color = "red")
    # axes[4,0].grid()
    # axes[4,0].set_title("Male >60 Years Old")
    
    for ageGroup, color in zip(["<30", "30-40", "40-50", "50-60", ">60"], 
                                ["yellow", "orange", "purple", "blue", "red"]):
        sns.distplot(male[features][(male["ageGroups"] == ageGroup)], 
                     kde_kws={"shade": True}, hist = False, label = ageGroup,
                    ax = axes[0], color = color)
        axes[0].grid()
        axes[0].set_title("Male All AgeGroup Distribution")
        axes[0].legend()
        
    # sns.distplot(female[features][(female["ageGroups"] == "<30")], kde_kws={"shade": True},
    #                  hist = False, ax = axes[0,1], color = "yellow")
    # axes[0,1].grid()
    # axes[0,1].set_title("Female 20-30 Years Old")

    # sns.distplot(female[features][(female["ageGroups"] == "30-40")], kde_kws={"shade": True}, hist = False,
    #                  ax = axes[1,1], color = "orange")
    # axes[1,1].grid()
    # axes[1,1].set_title("Female 30-40 Years Old")

    # sns.distplot(female[features][(female["ageGroups"] == "40-50")], kde_kws={"shade": True}, hist = False,
    #                  ax = axes[2,1], color = "purple")
    # axes[2,1].grid()
    # axes[2,1].set_title("Female 40-50 Years Old")


    # sns.distplot(female[features][(female["ageGroups"] == "50-60")], kde_kws={"shade": True}, hist = False,
    #                  ax = axes[3,1], color = "blue")
    # axes[3,1].grid()
    # axes[3,1].set_title("Female 50-60 Years Old")
    
    
    # sns.distplot(female[features][(female["ageGroups"] == ">60")], kde_kws={"shade": True}, hist = False,
    #                  ax = axes[4,1], color = "red")
    # axes[4,1].grid()
    # axes[4,1].set_title("Female >60 Years Old")
    
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
    plt.figure(figsize = (15,5))
    sns.distplot(data[features][(data["is_control"] == 1)], kde_kws={"shade": True}, label = "Control", hist = False)
    sns.distplot(data[features][(data["PD"] == 1)], kde_kws={"shade": True}, label = "PD", hist = False)
    sns.distplot(data[features][(data["MS"] == 1)], kde_kws={"shade": True}, label = "MS", hist = False)
    plt.legend()
    plt.grid()
    plt.show()
    
    
def visualize_passive_active(data, features):
    plt.figure(figsize = (15,5))
    sns.distplot(data[features][(data["PD"] == 1) & (data["version"] == "Passive")], kde_kws={"shade": True}, label = "ACTIVE-PD", hist = False)
    sns.distplot(data[features][(data["PD"] == 1) & (data["version"] == "V2")], kde_kws={"shade": True}, label = "PASSIVE-PD", hist = False)
    plt.legend()
    plt.grid()
    plt.show()