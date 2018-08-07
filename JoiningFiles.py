import pandas as pd

df2=pd.read_csv('User_id_folds.csv')
df1 = pd.read_csv('skill_builders_16_17_all_features.csv')
print(len(df1.index))
#df_final=df1.join(df2,  lsuffix='', rsuffix='',on='user_id')
df_final=df1.merge(df2,on='user_id',how='left')
print(len(df_final.index))
print(df_final.columns)