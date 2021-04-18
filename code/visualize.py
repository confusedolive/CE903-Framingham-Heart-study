import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from matplotlib import cm

data_path = r'C:\Users\jeron\OneDrive\Desktop\903group\data\framingham.csv'
figures_path = r'C:\Users\jeron\OneDrive\Desktop\903group\code\visualization'

#####load data#####
data_heart = pd.read_csv(data_path)
df_heart = data_heart.copy()
#############exploratory analysis############
len(df_heart) #length of data set before dropping nas

print(df_heart.isnull().sum()) # check number of nan values
df_heart.dropna(inplace=True) #drop na

len(df_heart) #length after dropping NAN
######undestanding the data###############
df_heart.dtypes #check the type of data
df_heart.nunique() #checking the cardinallity of each variable

attributes = {'categorical': ['male', 'currentSmoker','BPMeds',
                          'prevalentStroke','prevalentHyp',
                          'diabetes','education'],
               'continious': ['age', 'cigsPerDay','totChol',
                              'sysBP','diaBP','BMI',
                              'heartRate','glucose', 'TenYearCHD']}
################plotting label######################
####visualize the inbalance in label 'TenYearCHD'
def label_balance():
    sns.histplot(data=df_heart, x='TenYearCHD',
                shrink = 1, discrete=True, hue='TenYearCHD')

    plt.xticks([0,1])
    plt.title('Balance in output')
    plt.savefig(os.path.join(figures_path, 'label balance'))
    plt.show()
label_balance()
#visualize label inbalance by gender
def label_balance_gender():
    df = df_heart.copy()
    df.rename(columns={'male':'gender'}, inplace=True)
    df['gender'].replace([0,1],['female', 'male'], inplace=True)

    face_gender = sns.FacetGrid(data=df, col='gender', hue='TenYearCHD', palette='muted')

    face_gender.map(sns.histplot, 'TenYearCHD')
    plt.xticks([0,1])
    plt.savefig(os.path.join(figures_path, 'label balance gender'))
    plt.show()

#label balance per categorical features, found in dict attributes['categorical']
def label_balance_variable():

    df = df_heart.copy()
    path = os.path.join(figures_path, 'label variable balance')
    if not os.path.exists(path):
        os.mkdir(path)

    for x in attributes['categorical']:
        facet = sns.FacetGrid(data=df, col=x, hue='TenYearCHD')
        facet.map(sns.histplot, 'TenYearCHD')
        plt.xticks([0,1])
        plt.savefig(os.path.join(path, f'{x} label distribution'),
        bbox_inches='tight')
        plt.show()

##################comparing/ visualizing ################################

#correlation heatmap utilizing pearson correlation in full dataset
def corr_plot():
    correlation = df_heart.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(14,10))
    plt.title('correlation')
    sns.heatmap(correlation, mask=np.zeros_like(correlation, dtype=np.bool),
                cmap=cm.cividis_r, square=True, ax=ax, annot=True)
    plt.savefig(os.path.join(figures_path, ' variables correlation heatmap'))
    plt.show()

#visualize the cardinallity of the continious variables in the dataset
def card_plot():
    df_heart[attributes['continious']].nunique().plot.bar(figsize=(12, 6))
    plt.ylabel('number of unique variables')
    plt.xlabel('variables')
    plt.title('cardinallity of continious variables')
    plt.savefig(os.path.join(figures_path, r'continious cardinallity'))
    plt.show()


df_heart['age range'] = pd.cut(df_heart['age'], bins=3)
def percentage_perclass(variable):
    freq = (df_heart[variable].value_counts()/ len(df_heart)) * 100
    fig = freq.sort_values(ascending=False).plot.bar()
    fig.set_ylabel(f'percentage of patients within each {variable}')
    fig.set_xlabel(variable)
    plt.savefig(os.path.join(figures_path,'age ranges'))
    plt.show()

percentage_perclass('age range')

################checking distribution of variables #######################

def continus_distribution():

    path = os.path.join(figures_path, 'distribution')

    if not os.path.exists(path):
        os.mkdir(path)

    for element in attributes['continious']:
        sns.displot(df_heart[element], kde=True)
        plt.title(f'{element} distribution')
        plt.savefig(os.path.join(path, f'{element} distribution.png'),
                                                bbox_inches='tight')
        plt.show()

def line_distribution():

    path = os.path.join(figures_path, 'line dist')

    if  not os.path.exists(path):
        os.mkdir(path)

    for element in attributes['continious']:
        stats.probplot(df_heart[element], dist='norm', plot=plt)
        plt.title(f'{element} line distribution')
        plt.savefig(os.path.join(path, f'{element} line distribution.png'))
        plt.show()
line_distribution()
#############################looking for outliers#######################################
def box_plots():
    path = os.path.join(figures_path, 'boxplot')
    if not os.path.exists(path):
        os.mkdir(path)
    for elements in attributes['continious']:
        sns.boxplot(df_heart[elements])
        plt.title(f'{elements} box plot')
        plt.savefig(os.path.join(path, f'{elements} boxplot'))
        plt.show()

#########################################################################################
def violin_plots():

    path = os.path.join(figures_path, 'violinplot')
    if not os.path.exists(path):
        os.mkdir(path)
    for elements in attributes['continious']:
        sns.violinplot(x=df_heart[elements])
        plt.title(f'{elements} violinplot')
        plt.savefig(os.path.join(path,f'{elements} violinplot'))
        plt.show()
violin_plots()
#############joyplot########################################
from matplotlib import cm
from joypy import joyplot

def joyplotting():
    test = df_heart.copy()
    test['male'] = test['male'].replace([0,1], ['female','male'])
    plt.tight_layout()
    plt.figure(figsize=(16,10), dpi=600)
    fig, axes = joyplot(test, column='TenYearCHD', by='male', ylim='own', figsize=(14, 10),
    colormap = cm.autumn_r)
    plt.title('Label distribution per gender')
    plt.savefig(os.path.join(figures_path, 'label per gender'), bbox_inches='tight')
    plt.show()

def categorical_cardinallity():
    path = os.path.join(figures_path, 'caregorical cardinallity')
    data_heart.nunique().plot.bar(figsize=(12,8))
    plt.ylabel('count categories')
    plt.xlabel('variables')
    plt.show()

def scatters():
    path = os.path.join(figures_path, 'scatter continious')
    if not os.path.exists(path):
        os.mkdir(path)
    for x in attributes['continious']:
        if x not in ['age', 'cigsPerDay']:
            sns.scatterplot(data = data_heart,y=list(range(len(data_heart))), x=x, hue='TenYearCHD')
            plt.title(f'{x} distribution')
            plt.savefig(os.path.join(path,f'{x} distribution'))
            plt.show()

def join_boxplot():
    plt.figure(figsize=(16,8))
    sns.boxplot(data=data_heart[attributes['continious']])
    plt.title('Continuous variables boxplots')
    plt.savefig(os.path.join(figures_path,'box plot all'))
    plt.show()


def education_risk():
    df_heart.dropna(inplace=True)
    sns.set_style('darkgrid')
    sns.set_context('paper', font_scale=1.4)
    sns.violinplot(x='education',y=list(range(1, len(df_heart)+1)), data=df_heart, hue='TenYearCHD', split=True)
    plt.legend( loc=0)
    plt.title('Education violin_plots')
    plt.show()







#fail to reject h0 ergo indepedent
