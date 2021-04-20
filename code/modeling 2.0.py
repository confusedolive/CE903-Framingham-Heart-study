import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from boruta import BorutaPy
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from scipy.stats import chi2_contingency, randint
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, classification_report, plot_confusion_matrix,
                             precision_recall_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                      Loading Dataset                                                                               #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

data_path = r'C:\Users\jeron\OneDrive\Desktop\903group\data\framingham.csv'

data = pd.read_csv(data_path)
data_heart = data.copy()
data_heart.dropna(inplace=True)
data_heart['male'].isnull().sum()
# fill missing values with the means
#data_heart = data_heart.apply(lambda x: x.fillna(x.mean()), axis=0)
# Drop current smoker and education ,
# CigsperDay covers current smoker and education is not relevant
data_heart.drop(columns=['currentSmoker', 'education'], axis=1, inplace=True)

# CurrentSmoker is irrelevant considering we have cigsperday, therefore is dropped.
print(len(data_heart))
print(data_heart['TenYearCHD'].value_counts())
output = 'TenYearCHD'
features = ['male', 'age',
            'cigsPerDay', 'BPMeds', 'prevalentStroke',
            'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
            'diaBP', 'BMI', 'heartRate', 'glucose']

# No scaling are categorical features [0,1] that dont need scaling
no_scaling = ['male', 'BPMeds', 'prevalentStroke',
              'prevalentHyp', 'diabetes', 'TenYearCHD']

# list of features that need scaling
need_standard = [x for x in features if x not in no_scaling]
features_to_scale = data_heart[need_standard]

data_heart[need_standard].describe()
print(need_standard)

standard = StandardScaler()
scaled_features = standard.fit_transform(features_to_scale)
data_heart[need_standard] = scaled_features
scaler = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler.fit_transform(data_heart[features])

data_heart[features] = features_scaled
# separate features and output
X = data_heart[features]
y = data_heart[output].values
print(X.describe())

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                               preprocessing and feature selection functions                                                                      #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#


def boruta_selected():
    randomclf = RandomForestClassifier(n_jobs=-1,
                                       max_depth=6, n_estimators=1000,
                                       class_weight='balanced')

    boruta_select = BorutaPy(randomclf, n_estimators='auto',
                             verbose=2, random_state=1)

    boruta_select.fit(np.array(X_train), np.array(y_train))

    features_importance = [X.columns[i]
                           for i, boolean in enumerate(boruta_select.support_) if boolean]

    not_important = [X.columns[i]
                     for i, boolean in enumerate(boruta_select.support_) if not boolean]
    return features_importance, not_important


def ChiSquare(data_heart, output, alpha=0.05):
    '''
      ----------------------------------------------------
       Utilizes the chi squared test to assest relevance
       in features, if a feature's p value is below or
       equal to  alpha it is considered relevant
       ----------------------------------------------------
            * data_heart = Dataset
            * output = Label class
            * alpha = p value threshold
       ----------------------------------------------------
       returns a list of relevant and not relevant features
       '''
    relevant = []
    relevant_pval = []
    not_relevant = []
    not_relevant_pval = []
    for column in data_heart.columns:
        if column != output:
            cross = pd.crosstab(data_heart[column], data_heart[output])
            chi_square_value, p_value, _, _ = chi2_contingency(cross)
            if p_value <= alpha:
                relevant.append(column)
                relevant_pval.append(p_value)
            else:
                not_relevant.append(column)
                not_relevant.append(p_value)

    return relevant, relevant_pval, not_relevant, not_relevant_pval


# get relevant features and their p_values to understand which features are statisticially significant
relevant, relevant_p, not_relevant, not_relevant_p = ChiSquare(
    data_heart, output)
relevant_pvals_chi = list(zip(relevant, relevant_p))
for tup in relevant_pvals_chi:
    print(tup)
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                      Evaluation  functions                                                                         #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#


def pre_recall_auc(y_p, y_t, label):
    '''
    -------------------------------------------------
    Plots the precision recall curve
    and returns the area under the curve(auc) for it
    -------------------------------------------------
        * y_p = Predicted labels
        * y_t = True labels
        * label = model label
    -------------------------------------------------
    '''
    prec, rec, threshold = precision_recall_curve(y_p, y_t)
    auc_score = auc(rec, prec)

    plt.plot(rec, prec, marker='.', label=f'{label} (auc = {auc_score})')
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision recall curve')
    plt.show()

    return auc_score


def plot_conf(model):
    ''' plots and shows confusion matrix for model'''
    conf = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=[0, 1],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    plt.title('Confusion matrix')
    plt.show()


def evaluate_model(model, modelname):
    '''
    ------------------------------------------------------------------------------------------
    Function to ease the process of evaluating many models , prints a form of resume
    of the relevant metrics and performance of the model/models wiht a focus in
    precision and recall , prints:

               * Confusion matrix
               * Classification report
               * Precision recall curve and area under curve
               * Prints and returns Accuracy, F1 score, Precision, Recall

    ------------------------------------------------------------------------------------------
    Model is the model to be evaulated e.g. modle=LogisticRegression(),
    modelname is the name of the model to be printed and to identify it in the Evaluation
    e.g. modelname=logregression
    -------------------------------------------------------------------------------------------
    '''
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('#' * 80, '\n')
    print(' ' * (40 - len(modelname)), modelname, ' evaluation\n')
    print('-' * 80, '\n')
    print(f'{modelname} fited in train set and evaluated in test set')
    print(f'Unbalance label being  a problem the focus of evaluation is in precision and recall\n')
    print('-' * 80, '\n')
    print('#' * 80, '\n')
    print('-' * 80, '\n')
    print(' ' * 25, f'Confusion matrix of {modelname}\n')
    print('-' * 80, '\n')
    try:
        plot_conf(model)
        con = tf.math.confusion_matrix(
            labels=y_test, predictions=y_predict).numpy()
        print(con)
    except:
        con = tf.math.confusion_matrix(
            labels=y_test, predictions=y_predict).numpy()
        print(con)
    print('-' * 80, '\n')
    print(' ' * 20, f'Classification report of  {modelname}\n')
    print('-' * 80, '\n')
    print(classification_report(y_test, y_predict))
    print('-' * 80, '\n')
    print(' ' * 20, f'Precision-recall curve of {modelname}\n')
    print('-' * 80, '\n')
    model_auc = pre_recall_auc(y_predict, y_test, modelname)
    print('\n')
    print('-' * 80, '\n')
    print(' ' * 20, 'Metrics\n')
    print('-' * 80, '\n')
    print(
        f'The area under the curve for the precision recall curve is : {model_auc:.2f}')
    print(f"{modelname}'s acuracy is {model.score(X_test,y_test):.2f}")
    print(f"{modelname}'s f1 score is {metrics.f1_score(y_test, y_predict):.2f}")
    print(f"{modelname}'s precision is {metrics.precision_score(y_test, y_predict):.2f}")
    print(f"{modelname}'s recall is {metrics.recall_score(y_test, y_predict):.2f}")
    print('-' * 80, '\n')
    print('#' * 80, '\n')
    return metrics.recall_score(y_test, y_predict), metrics.precision_score(y_test, y_predict), model_auc, metrics.f1_score(y_test, y_predict), model.score(X_test, y_test)


def evaluate_n_models(models, type_test):
    '''
    -------------------------------------------------------
    Tests different models , prints a report of each model
    utilizing the evaluate_model function found in line 157
    -------------------------------------------------------

        * models = list of tuples containing (modelname, model)
         e.g. (['Random Forest', RandomForestClassifier()'])

         *typetest is for clarification as to what parameters
         are being tested , it will print a message before anything else

    -------------------------------------------------------
    returns a pandas dataframe with
    index =
        * Recall, precision, area under the curve and f1 score
    columns =
        * Model names
    prints that same dataframe
    ------------------------------------------------------
    '''
    print(type_test, '\n')
    scores_dict = {}
    for modelname, model in models:
        print(modelname, '\n')
        recall, prec, auc_score, f1, acc = evaluate_model(model, modelname)
        scores_dict[modelname] = {
            'Recall': recall, 'Precision': prec,
            'Area under curve': auc_score, 'f1 score': f1, 'Accuracy': acc}
    scores = pd.DataFrame(scores_dict)
    print('-' * 80, '\n')
    print('All scores\n')
    print(scores)
    return scores
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                               Set selection, processing and models functions                                                       #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#


def get_models():
    '''returns list of models containing
       tuples with (modelname, model)'''
    models = [
        ('Random Forest', RandomForestClassifier()),
        ('Logistic regression', LogisticRegression()),
        ('Decision tree ', DecisionTreeClassifier()),
        ('Support vector maching', SVC()),
        ('Naive Bayes', GaussianNB()),
        ('k nearest', KNeighborsClassifier(n_neighbors=2)),
    ]
    return models


def put_features(X_train, X_test, chi=False, boruta=False):
    '''Takes X train and X test and converts the features
    to features selected by either Chi Squared test if chi is set to True
    or features selected by boruta if boruta is set to True,
    returns train set and test set'''
    if chi:
        features_importance, not_important, _, __ = ChiSquare(
            data_heart, output)
    if boruta:
        features_importance, not_important = boruta_selected()
    train = X_train[features_importance]
    test = X_test[features_importance]
    return train, test


def get_train_test(X, y, oversample=False, undersample=False, over_sampling=None, test_size=0.20, n=8):
    '''
      --------------------------------------------------------------------------
       Utilizes sklearn train and split function to split the dataset
       this functions is used to facilitate testing different oversampling,
       undersampling ratios, test sizes and train sizes.
       --------------------------------------------------------------------------

              *  X,y are the paramters for x= features y=label
              *  If oversample is True the X_train, Y_train gets oversampled utilizing SMOTE
              *  If undersample is True  the X_train, Y_train gets undersampled
                 utilizing RandomUnderSampler
              *  over_sampling sets the sampling strategy for SMOTE over sampling
              *  under_sampling sets the sampling strategy for RandomUnderSampler under sampling
              *  test_size sets the size of the test set

        --------------------------------------------------------------------------
       '''
    if oversample:
        over = SMOTETomek(random_state=42)
    if undersample:
        undersample = NearMiss(version=2, n_neighbors_ver2=2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    if oversample:
        X_train, y_train = over.fit_resample(X_train, y_train)
        if undersample:
            X_train, y_train = under.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def visualize_performances(df, description):
    '''Creates a new folder inside the visualization folder
      and saves figures that visualize the performance of different metrics
      per model
      -----------------------------------------------------------------------
      df= pandas dataframe containing the models and its metric scores
      description = titles to be identified when figures are saved
      -----------------------------------------------------------------------
      '''
    path = r'code\visualization\performance'
    if not os.path.exists(path):
        os.makedirs(path)
    df = df.T
    for col in df.columns:
        sns.barplot(x=df[col], y=df.index)
        plt.xlabel(" ")
        plt.title(col)
        plt.savefig(os.path.join(
            path, f'{col} {description}'), bbox_inches='tight')
        plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                         Testing                                                                                    #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

# combinations of features selected by boruta and features selected by ChiSquare


# Split with get_train_test ,no features selected, test size 20% train size 20%
models = get_models()
X_train, X_test, y_train, y_test = get_train_test(X, y)
no_features = evaluate_n_models(
    models, 'No feature selection no sampling techniques')
visualize_performances(no_features, 'No features')
# chi features

X_train, X_test = put_features(X_train, X_test, chi=True)
mod = get_models()
chi_feature_scores = evaluate_n_models(
    mod, 'features selected with chi squared test')
visualize_performances(chi_feature_scores, 'chi features')

X_train, X_test, y_train, y_test = get_train_test(X, y)
X_train, X_test = put_features(X_train, X_test, boruta=True)
mod = get_models()
boruta_feature_scores = evaluate_n_models(
    mod, 'features selected with Borutapy')
visualize_performances(boruta_feature_scores, 'boruta features')
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                        Findings mondel tuning                                                                      #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#


# Logistic regression and support vector machine seems to be performing the best overall,
# with featuers selected using the chi squared test

X_train, X_test, y_train, y_test = get_train_test(X, y, oversample=False)
X_train, X_test = put_features(X_train, X_test, chi=True)

# Hyperparameters to be tested for Logistic Regression
params = [{'penalty': ['l2'],
          'C': np.logspace(-4, 4, 50),
           'class_weight': ['balanced', None],
           'solver':['newton-cg', 'lbfgs', 'sag', 'saga']},
          {'penalty': ['elasticnet'],
           'C': np.logspace(-4, 4, 50),
           'class_weight': ['balanced', None],
           'solver':['saga'], 'l1_ratio':[0, 1]},
          {'penalty': ['l1'],
              'C': np.logspace(-4, 4, 50),
           'class_weight': ['balanced', None],
           'solver':['liblinear', 'saga']}]

log_model = LogisticRegression()
grid = GridSearchCV(log_model, params, cv=10)
grid.fit(X_train, y_train)
print(grid.best_estimator_)

# cross validating performance of logistic regression
optimal_logistic = LogisticRegression(
    C=0.18420699693267145, solver='newton-cg', penalty='l2')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(optimal_logistic, X_train, y_train, cv=cv)

print(scores.mean())
###get classification report
optimal_logistic.fit(X_train, y_train)
y_pred_logistic = optimal_logistic.predict(X_test)
print(classification_report(y_test, y_pred_logistic))

# Support vector machine Hyperparameter tuning
svc = SVC()
params_svc = {'C': [0.01, 0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'sigmoid'],
              'class_weight': ['balanced', None]}

grid_svc = GridSearchCV(svc, params_svc, verbose=2, cv=10)
grid_svc.fit(X_train, y_train)
print(grid_svc.best_estimator_)

optimal_svc = SVC(C=1, gamma=1, kernel='poly')
cv1 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores1 = cross_val_score(optimal_svc, X_train, y_train, cv=cv1)
print(scores1.mean())

# Decision Tree hyperparameter tuning

tree = DecisionTreeClassifier()
params = param_dist = {"max_depth": [3, 6, 8, 12],
                       "max_features": [2, 4, 6, 9],
                       "min_samples_leaf": [2, 4, 5, 9],
                       "criterion": ["gini", "entropy"]}
tree_grid = GridSearchCV(tree, params, cv=10)
tree_grid.fit(X_train, y_train)
print(tree_grid.best_estimator_)
optimal_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, max_features=4,
                                      min_samples_leaf=2)
cv2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores2 = cross_val_score(optimal_tree, X_train, y_train, cv=cv2)
print(scores2.mean())

# Comparing performance of all three hypertuned parameters
optimized_models = [('Decision Tree', optimal_tree), ('Support Vector Machine',
                                                      optimal_svc), ('Logistic Regression', optimal_logistic)]
optimals = evaluate_n_models(optimized_models, 'all models')
visualize_performances(optimals, 'optimal models test')
optimals
# visualizing the area under the curve for the optimized models
for name, model in optimized_models:
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    prec, rec, threshold = precision_recall_curve(y_test, y_predict)
    auc_score = auc(rec, prec)
    plt.plot(rec, prec, marker='.', label=f'{name} (auc = {auc_score:.2f})')
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision recall curve')
plt.savefig('COMPARE auc optima testl')
plt.show()
optimals
