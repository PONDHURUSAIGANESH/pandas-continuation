
import pandas as pd
dat = pd.read_csv('C:/Users/Dell/Downloads/Data_Set.csv')

A = pd.read_csv('C:/Users/Dell/Downloads/Data_Set.csv',header=2)

B = A.rename(columns={'Temperature':'Temp'})

C = B.drop('No. Occupants',axis=1)

B.drop('No. Occupants',axis=1,inplace = True)

D = C.drop(2,axis=0)

E = D.reset_index(drop = True)

E.describe()

mn = E['E_Heat'].min()

E['E_Heat'][E['E_Heat'] == mn]

E['E_Heat'].replace(-4,9, inplace = True)

E.info()

#co-variance

E.cov()

import seaborn as sn

sn.heatmap(E.corr())


'''
missing values

'''

import numpy as np
import pandas as pd

F = E.replace('!',np.NaN)
 
F.info()

F = F.apply(pd.to_numeric)

F.info()

F.isnull()

F.mean()

G = F.fillna(method = 'ffill')#METHOD = 'ffill' frontfill

#METHOD = 'bfill' backfill

x = F['E_Plug'].mean()

F['E_Plug'].fillna(x, inplace =True)

print(F.to_string())

#                   (OR)

from sklearn.impute import SimpleImputer as sm

ss = sm(missing_values=np.NaN,strategy= 'mean')
ss.fit(F)

H = ss.transform(F)

F.boxplot()
F['E_Plug'].quantile(0.25)


F['E_Plug'].quantile(0.75)

J = pd.read_csv("C:/Users/Dell/Downloads/Data_New.csv")

I = pd.concat([F,J], axis=1)
I.info()
I.replace(np.NaN,34,inplace=True)
K = pd.get_dummies(I)
