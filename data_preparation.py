#imports 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


#constants
crop_species = ["Maize", "Rice", "Wheat", "Soybean"]



#functions

def splitting_dataset(pretraining, same_locations, train_on_all, Adaptation_Type):
    testing = []
    training = []

    for i in range(len(pretraining)):
        ds = pretraining[i]

        #choose the id of the chosen location
        if same_locations:
            ID = np.arange(1, len(ds['Loc']) + 1)
            DATA_sample = ds
            DATA_sample.index = ID
            DATA_sample['ID'] = ID
            
            # Remove locations including one data only
            Loc_1 = DATA_sample['Loc'].value_counts().loc[lambda x: x < 2].index.tolist()
            DATA_sample2 = DATA_sample.copy()
            if len(Loc_1) > 0:
                ID_1 = DATA_sample.loc[DATA_sample['Loc'].isin(Loc_1), 'ID']
                DATA_sample2 = DATA_sample[~DATA_sample['ID'].isin(ID_1)]
            
            # Sample one data per location
            new_data = DATA_sample2.groupby('Loc').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
            ID = new_data['ID'].tolist()
            #print(ID)
            #print(len(ID))

        else:
            # Split using different locations
            UnLoc = np.unique(ds['Loc'])
            ID_loc = np.random.choice(UnLoc, size=int(round(0.25 * len(UnLoc))), replace=False)
            ID = np.arange(1, len(ds['Loc']) + 1)[np.isin(ds['Loc'], ID_loc)]
            #print(ID)
            #print(len(ID))
    

        # Splitting the dataset into the Training set and Test set
        if train_on_all:
            testing.append(pretraining[i].drop(['Loc', 'ID'], axis=1))
            training.append(pretraining[i].drop(['Loc', 'ID'], axis=1))
            print(training[i].shape)
            print(testing[i].shape)
        else:
            training.append(ds[~ds['ID'].isin(ID)].drop(['Loc', 'ID'], axis=1))
            testing.append(ds[ds['ID'].isin(ID)].drop(['Loc', 'ID'], axis=1))
            print(training[i].shape)
            print(testing[i].shape)  
    return training, testing 