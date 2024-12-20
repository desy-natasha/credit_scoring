from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks 
from imblearn.over_sampling import SMOTE 
from collections import Counter
import pandas as pd
import numpy as np

def convert_to_object(data,columns):
    for col in columns:
        data[col] = data[col].astype(object)

    return data

class data_preparation:

    def sample_data(self,dataframe,columns_to_convert=[],seed=1,pprint=True):
        converted_df = convert_to_object(dataframe,columns_to_convert)

        self.data_train,self.data_test = train_test_split(converted_df, test_size=0.2,random_state=seed,stratify=converted_df['Label'])

        if pprint:
            print('Data Shape: ',converted_df.shape)
            print('Data Train Shape: ',self.data_train.shape)
            print('Data Test Shape: ',self.data_test.shape)


    def fe_data(self,seed=1,scaling=True,pprint=True):

        object_cols = list(self.data_train.select_dtypes(include='object'))
        numeric_cols = list(self.data_train.select_dtypes(include='int'))
        numeric_cols.remove('Label')
        
        # log transform for numeric column
        for i in numeric_cols:
            var_name = "Log_"+str(i)
            
            # transform amount into log
            self.data_train[var_name]=self.data_train[i].apply(lambda x: np.log(x))
            self.data_test[var_name]=self.data_test[i].apply(lambda x: np.log(x))
            
            # drop initial column as it is now transformed
            self.data_train = self.data_train.drop(i,axis = 1)
            self.data_test = self.data_test.drop(i,axis = 1)
        
        # scaling for numeric column
        if scaling:
            numeric_cols = ['Log_' +str(i) for i in numeric_cols]
            
            scaler = StandardScaler()
            self.data_train[numeric_cols] = scaler.fit_transform(self.data_train[numeric_cols])
            self.data_test[numeric_cols] = scaler.transform (self.data_test[numeric_cols])
            
        # one hot encoding for object column
        test_index = self.data_test.index
        df_all = pd.concat([self.data_train,self.data_test],axis=0)
        
        # categorice type columns
        df_all=pd.get_dummies(df_all,columns=object_cols)

        self.data_test = df_all.loc[test_index]
        self.data_train = df_all.loc[~df_all.index.isin(test_index)]

        # make train and test dataset
        self.X_train = self.data_train.drop('Label',axis=1)
        self.y_train = self.data_train[['Label']]

        self.X_test = self.data_test.drop('Label',axis=1)
        self.y_test = self.data_test['Label']

        if pprint:
            print(f'train -  {np.bincount(self.y_train.Label)}   |   test -  {np.bincount(self.y_test)}')

# SMOTE-TomekLink
def smote_tomek(X,y,print_result=False):
    df_train = pd.concat([X,y],axis=1)

    smt = SMOTETomek(random_state=1,tomek=TomekLinks(sampling_strategy='majority'),smote=SMOTE(random_state=1,sampling_strategy=0.5))
    X_train_sam, y_train_sam = smt.fit_resample(X, y)
    
    if print_result:
        print('Original dataset shape:', '0: ', Counter(df_train.Label)[0],'1: ', Counter(df_train.Label)[1])
        print('Sampling dataset shape:', '0: ', Counter(y_train_sam.Label)[0],'1: ', Counter(y_train_sam.Label)[1])
        print(f'majority data reduce: {round(100*(Counter(df_train.Label)[0]-Counter(y_train_sam.Label)[0])/Counter(df_train.Label)[0],2)}%')
        print(f'minority data generate: {round(100*(Counter(y_train_sam.Label)[1]-Counter(df_train.Label)[1])/Counter(df_train.Label)[1],2)}%')
    return X_train_sam, y_train_sam