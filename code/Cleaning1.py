import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

data_path = r'C:\Users\jeron\OneDrive\Desktop\903group\data\framingham.csv'
figures_path = r'C:\Users\jeron\OneDrive\Desktop\903group\code\visualization'

#####load data#####
data_heart = pd.read_csv(data_path)
df_heart = data_heart.copy()

#############exploratory analysis############
len(df_heart)

print(df_heart.isnull().sum()) # check number of nan values
df_heart.dropna(inplace=True)

len(df_heart)

######undestanding the data###############
data.dtypes
data.nunique()

label_count = data['TenYearCHD'].value_counts()
ratio = f'approximately {label_count[0]/label_count[1]:.0f} : 1'
ratio
################plotting label######################
sns.histplot(data=df_heart, x='TenYearCHD',
            shrink = 1, discrete=True, hue='TenYearCHD')

plt.xticks([0,1])
plt.title('Balance in output')
plt.savefig(os.path.join(figures_path, 'label balance'))
################data type########################
attributes = {'nominal': ['sex','currentSmoker','BPMeds',
                          'prevalentStroke','prevalentHyp',
                          'diabetes','TenYearCHD'],
               'ordinal': 'education',
               'continious': ['age', 'cigsPerDay','totChol',
                              'sysBP','diaBP','BMI',
                              'heartRate','glucose']}

df_heart[attributes['continious']].nunique().plot.bar(figsize=(12, 6))
plt.ylabel('number of unique variables')
plt.xlabel('variables')
plt.title('cardinallity of continious variables')
#plt.savefig(os.path.join(figures_path, r'continious cardinallity'))

##################comparing/ visualizing ################################
df_heart['age range'] = pd.cut(df_heart['age'], bins=3)

def percentage_perclass(variable):
    freq = (df_heart[variable].value_counts()/ len(df_heart)) * 100
    fig = freq.sort_values(ascending=False).plot.bar()
    fig.set_ylabel(f'percentage of patients within each {variable}')
    fig.set_xlabel(variable)
    plt.show()

#percentage_perclass('age range')

################checking distribution of variables #######################
def continus_distribution():
    for element in attributes['continious']:
        sns.displot(df_heart[element], kde=True)
        plt.title(f'{element} distribution')
        plt.savefig(os.path.join(figures_path, f'{element} distribution.png'))
        plt.show()

def line_distribution():
    for element in attributes['continious']:
        stats.probplot(df_heart[element], dist='norm', plot=plt)
        plt.title(f'{element} line distribution')
        plt.savefig(os.path.join(figures_path, f'{element} line distribution.png'))
        plt.show()
line_distribution()
