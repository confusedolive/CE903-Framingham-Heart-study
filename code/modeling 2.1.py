import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.classifier import EnsembleVoteClassifier
from scipy.stats import chi2_contingency
from sklearn import metrics
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
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
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from collections import Counter
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                      Loading Dataset                                                                               #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

data_path = r'C:\Users\jeron\OneDrive\Desktop\903group\data\framingham.csv'

data = pd.read_csv(data_path)
data_heart = data.copy()
#data_heart_means = data_heart.apply(lambda x: x.fillna(x.mean()), axis=0)
data_heart.dropna(inplace=True)
data_heart.drop('currentSmoker', axis=1, inplace=True)
# CurrentSmoker is irrelevant considering we have cigsperday, therefore is dropped.

output = 'TenYearCHD'
features = ['male', 'age', 'education',
            'cigsPerDay', 'BPMeds', 'prevalentStroke',
            'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
            'diaBP', 'BMI', 'heartRate', 'glucose']

not_standard = ['male', 'BPMeds', 'prevalentStroke',
                'prevalentHyp', 'diabetes', 'TenYearCHD', 'education']

# features that need standarizing i.e. continuous featuers
need_standard = [x for x in features if x not in not_standard]
features_to_scale = data_heart[need_standard]

#check whether any of the categorical varaibles needs to be one hot encoded
for feature in not_standard:
    print(data_heart[feature].value_counts())

print(need_standard)

#standarize all continuous features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_to_scale.values)
#encodes education
data_heart = pd.get_dummies(data_heart, columns=['education'], prefix='dumm')
data_heart[need_standard] = features_scaled
features = [feat for feat in data_heart.columns if feat not in output]

#separate features and output
X = data_heart[features]
y = data_heart[output]


# Label balance
#     15.2269% = 1 | heart risk
#     84.7731% = 0 | no heart risk
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

    boruta_select.fit(np.array(X), np.array(y))

    features_importance = [X.columns[i]
                           for i, boolean in enumerate(boruta_select.support_) if boolean]

    not_important = [X.columns[i]
                     for i, boolean in enumerate(boruta_select.support_) if not boolean]
    return features_importance, not_important


def ChiSquare(data_heart, output, alpha=0.01):
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

    relevant.append('totChol')
    return relevant,relevant_pval, not_relevant, not_relevant_pval

#get relevant features and their p_values to understand which features are statisticially significant
relevant, relevant_p, not_relevant, not_relevant_p = ChiSquare(data_heart, output)
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
    return metrics.recall_score(y_test, y_predict), metrics.precision_score(y_test, y_predict)\
                    ,model_auc, metrics.f1_score(y_test, y_predict),model.score(X_test, y_test)


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
             'Area under curve': auc_score, 'f1 score': f1, 'Accuracy':acc }
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
        ('Logistic regression', LogisticRegression(C=11.288378916846883, penalty='l1', solver='saga', max_iter=1000)),
        ('Decision tree ', DecisionTreeClassifier()),
        ('Support vector maching', SVC(kernel='linear', probability=True, C=100, gamma=1)),
        ('Naive Bayes', GaussianNB(var_smoothing=2e-9)),
    ]
    return models

def put_features(X_train, X_test, chi=False, boruta=False):
    '''Takes X train and X test and converts the features
    to features selected by either Chi Squared test if chi is set to True
    or features selected by boruta if boruta is set to True,
    returns train set and test set'''
    if chi:
        features_importance, not_important, _, __ = ChiSquare(data_heart, output)
    if boruta:
        features_importance, not_important = boruta_selected()
    train = X_train[features_importance]
    test = X_test[features_importance]
    return train, test

def get_train_test(X, y, oversample=False, undersample=False, over_sampling=.2, under_sampling=.5, test_size=.15, n=8):
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
        over = SMOTE(sampling_strategy=over_sampling, k_neighbors = n)
    if undersample:
        under = RandomUnderSampler(sampling_strategy=under_sampling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    if oversample:
        X_train, y_train = over.fit_resample(X_train, y_train)
        if undersample:
            X_train, y_train = under.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                         Testing                                                                                    #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

#combinations of features selected by boruta and features selected by ChiSquare

# Split with get_train_test , oversample and undersample are both is false, no boruta selected features
models = get_models()
X_train, X_test, y_train, y_test = get_train_test(X, y)
noboruta_sampling = evaluate_n_models(
    models, 'No feature selection no sampling techniques')

#chi testing no oversampling
X_train, X_test, y_train, y_test = get_train_test(X, y, oversample=False,
                                                  undersample=False, over_sampling=.2, under_sampling=.5, test_size=.2)

X_train , X_test = put_features(X_train, X_test,  chi=True)
models1 = get_models()
chi_feat = evaluate_n_models(get_models(), 'chi squared features no oversampling')


X_train, X_test, y_train, y_test = get_train_test(X, y, oversample=True,
                                                  undersample=False, over_sampling=.9, under_sampling=.8, test_size=.2, n=8)

print(y_train.value_counts())
X_train, X_test = put_features(X_train, X_test, chi=True)

g
#Boruta testing
# Testing with boruta selected and just oversampling
