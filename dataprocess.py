import pandas as pd
import numpy as np
from collections import Counter
import math
import numpy as np

def Entropy(DataList):
    '''
        Calculate the entropy of a random variable
    '''
    counts = len(DataList)      # Total number
    counter = Counter(DataList) # Number of occurrences of each variable
    prob = {i[0]:i[1]/counts for i in counter.items()}      # Calculate p*log(p) for each variable
    H = - sum([i[1]*math.log2(i[1]) for i in prob.items()]) # Calculate entropy
    return H 


def fun_dot(data):
    Orientation = np.array([data['Orientation1'],data['Orientation2'],data['Orientation3']])
    Headway_Distance = np.array([data['Headway_Distance1'],data['Headway_Distance2'],data['Headway_Distance3']])
    return np.dot(Orientation,Headway_Distance)

##Required feature parameters ###Add the vector product of Orientation_Headway here as feature input

params = ["Headway_Distance1abs","Headway_Distance2abs","Headway_Distance3abs",
        "Preceding_Vel1abs","Preceding_Vel2abs","Preceding_Vel3abs",
        "Orientation1abs","Orientation2abs","Orientation3abs",
        "RPM_diff","Orientation_Headway"] 

# Window size set to 60 
win_size = 60
def get_train_xy(df):
    df1 = df.dropna()

    def fun_sep_values(data,key):
        datas = data[key].replace("(","").replace(")","").split(" ")
        return datas

    df1[["Orientation1","Orientation2","Orientation3"]] = df1.apply(fun_sep_values,args=(" Orientation",),axis=1,result_type="expand")
    df1[["Velocity1","Velocity2","Velocity3"]] = df1.apply(fun_sep_values,args=(" Velocity",),axis=1,result_type="expand")
    df1[["Position1","Position2","Position3"]] = df1.apply(fun_sep_values,args=(" Position",),axis=1,result_type="expand")
    df1[["Headway_Distance1","Headway_Distance2","Headway_Distance3"]] = df1.apply(fun_sep_values,args=(" Headway_Distance",),axis=1,result_type="expand")
    df1[["Preceding_Vel1","Preceding_Vel2","Preceding_Vel3"]] = df1.apply(fun_sep_values,args=(" Preceding_Vel",),axis=1,result_type="expand")

    df1 = df1.drop(columns=[' Orientation',' Position',' Velocity',' Headway_Distance',' Preceding_Vel'])


    #Data type conversion
    df1['Orientation1'] = df1['Orientation1'].astype(float)
    df1['Orientation2'] = df1['Orientation2'].astype(float)
    df1['Orientation3'] = df1['Orientation3'].astype(float)

    df1['Headway_Distance1'] = df1['Headway_Distance1'].astype(float)
    df1['Headway_Distance2'] = df1['Headway_Distance2'].astype(float)
    df1['Headway_Distance3'] = df1['Headway_Distance3'].astype(float)

    df1['Velocity1'] = df1['Velocity1'].astype(float)
    df1['Velocity2'] = df1['Velocity2'].astype(float)
    df1['Velocity3'] = df1['Velocity3'].astype(float)

    df1['Position1'] = df1['Position1'].astype(float)
    df1['Position2'] = df1['Position2'].astype(float)
    df1['Position3'] = df1['Position3'].astype(float)

    #Calculate Velocity's first-order difference
    param = 'Velocity'
    for i in range(3):
        df1[f"{param}{i+1}abs"] = df1[f"{param}{i+1}"].diff().abs()

    #Calculate Headway_Distance's first-order difference
    param = 'Headway_Distance'
    for i in range(3):
        df1[f"{param}{i+1}abs"] = df1[f"{param}{i+1}"].diff().abs()

    #Calculate  preceding's first-order difference
    param = 'Preceding_Vel'
    for i in range(3):
        df1[f"{param}{i+1}"] = df1[f"{param}{i+1}"].astype(float)
        df1[f"{param}{i+1}abs"] = df1[f"{param}{i+1}"].diff().abs()

    param = 'Orientation'
    
    for i in range(3):
        df1[f"{param}{i+1}"] = df1[f"{param}{i+1}"].astype(float)
        df1[f"{param}{i+1}abs"] = df1[f"{param}{i+1}"].diff().abs()

    df1['RPM_diff'] = df1[' RPM'].diff().abs() 
    
    #Add the cross product here ##Used to distinguish 
    
    df1['RPM_diff'] = df1[' RPM'].diff().abs() 
    
    df1['Orientation_Headway'] =  df1.apply(fun_dot,axis=1)

    train_datas = []
    mydf = df1.dropna() 
    info_entropy = []
    y = []
    for i in range(mydf.shape[0]-win_size):
        entropies = []
        train_data = []
        for param in params:
            entropies.append(Entropy(mydf[param].values[(i):(i+win_size)]))
            train_data.append(mydf[param].values[(i):(i+win_size)])
        train_datas.append(train_data)
        if sum(entropies)>40:
            y.append(3)
        elif sum(entropies)>38:
            y.append(2)
        else:
            y.append(1)
        info_entropy.append(sum(entropies))
    return train_datas,y

from tqdm import tqdm
import os
train_X = []
train_y = []
count = 0
for root, dirs, files in tqdm(os.walk("./DataLog", topdown=False)):
    for name in files:
        if name.endswith(".csv"):
            df = pd.read_csv(os.path.join(root, name)) 
            trainx,y = get_train_xy(df)
            train_X = train_X + trainx
            train_y = train_y + y
            count = count + 1
         
train_X = np.array(train_X)
train_y = np.array(train_y)

np.save("X2.npy", train_X)
np.save("y2.npy", train_y)

print("Finish processing ")

 
