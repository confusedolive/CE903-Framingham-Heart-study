import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.classifier import EnsembleVoteClassifier
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

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                      Loading Dataset                                                                               #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

data_path = r'C:\Users\jeron\OneDrive\Desktop\903group\data\framingham.csv'

data = pd.read_csv(data_path)
data_heart = data.copy()
data_heart.dropna(inplace=True)
data.drop('currentSmoker', axis=1, inplace=True)
# CurrentSmoker is irrelevant considering we have cigsperday, therefore is dropped.

output = 'TenYearCHD'
features = ['male', 'age', 'education',
            'cigsPerDay', 'BPMeds', 'prevalentStroke',
            'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
            'diaBP', 'BMI', 'heartRate', 'glucose']

not_standard = ['male', 'BPMeds', 'prevalentStroke',
                'prevalentHyp', 'diabetes', 'TenYearCHD']
need_standard = [x for x in features if x not in not_standard]
features_to_scale = data_heart[need_standard]

scaler = StandardScaler()

features_scaled = scaler.fit_transform(features_to_scale.values)
data_heart[need_standard] = features_scaled

X = data_heart[features]
y = data_heart[output]

# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                      preprocessing  functions                                                                      #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# boruta py recommends using pruned forest with a depth between 3-7


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


def get_train_test(X, y, oversample=False, undersample=False, over_sampling=.2, under_sampling=.5, test_size=.15):
    if oversample:
        over = SMOTE(sampling_strategy=over_sampling)
    if undersample:
        under = RandomUnderSampler(sampling_strategy=under_sampling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    if oversample:
        X_train, y_train = over.fit_resample(X_train, y_train)
        if undersample:
            x_train, y_train = under.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
#                                                      Evaluation  functions                                                                         #
# ---------------------------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------------------------#

def pre_recall_auc(y_p, y_t, label):
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
    conf = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=[0,1],
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

               -Confusion matrix
               -Classification report
               -Precision recall curve and area under curve
               -Prints and returns Accuracy, F1 score, Precision, Recall

    ------------------------------------------------------------------------------------------
    Model is the model to be evaulated e.g. modle=LogisticRegression(),
    modelname is the name of the model to be printed and to identify it in the Evaluation
    e.g. modelname=logregression
    -------------------------------------------------------------------------------------------
    '''
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('#' * 80, '\n')
    print(' '*(40-len(modelname)), modelname, ' evaluation\n')
    print('-' * 80, '\n')
    print(f'{modelname} fited in train set and evaluated in test set')
    print(f'Unbalance label being  a problem the focus of evaluation is in precision and recall\n')
    print('-' * 80, '\n')
    print('#' * 80, '\n')
    print('-' * 80, '\n')
    print(' '*25, f'Confusion matrix of {modelname}\n')
    print('-' * 80, '\n')
    try:
        plot_conf(model)
    except:
        con = tf.math.confusion_matrix(
        labels=y, predictions=y_pred).numpy()
        print(con)
    print('-' * 80, '\n')
    print(' '*20, f'Classification report of  {modelname}\n')
    print('-' * 80, '\n')
    print(classification_report(y_test, y_predict))
    print('-' * 80, '\n')
    print(' '*20, f'Precision-recall curve of {modelname}\n')
    print('-' * 80, '\n')
    model_auc = pre_recall_auc(y_predict, y_test, modelname)
    print('\n')
    print('-' * 80, '\n')
    print(' '*20, 'Metrics\n')
    print('-' * 80, '\n')
    print(
        f'The area under the curve for the precision recall curve is : {model_auc:.2f}')
    print(f"{modelname}'s acuracy is {model.score(X_test,y_test):.2f}")
    print(f"{modelname}'s f1 score is {metrics.f1_score(y_test, y_predict):.2f}\n")
    print(f"{modelname}'s precision is {metrics.precision_score(y_test, y_predict):.2f}\n")
    print(f"{modelname}'s recall is {metrics.recall_score(y_test, y_predict):.2f}\n")
    print('-' * 80, '\n')
    print('#' * 80, '\n')
    return metrics.recall_score(y_test, y_predict), metrics.precision_score(y_test, y_predict)\
                ,model_auc, metrics.f1_score(y_test, y_predict)


####################################################################################################

X_train, X_test, y_train, y_test = get_train_test(X,y)

models = [
          ('Random Forest', RandomForestClassifier()),
          ('Logistic regression', LogisticRegression()),
          ('Decision tree ', DecisionTreeClassifier()),
          ('Support vector maching', SVC())]

#########RESULTS WITH no boruta select or any form of oversampling.
scores_noboruta_nosampliing = {}
for modelname, model  in models:
    print(modelname,'\n')
    recall, prec, auc_score, f1 = evaluate_model(model, modelname)
    scores_noboruta_nosampliing[modelname] ={'Recall':recall, 'Precision':prec, 'Area under curve': auc_score, 'f1 score': f1}

print(scores_noboruta_nosampliing)
noboruta_sampling = pd.DataFrame(scores_noboruta_nosampliing)

noboruta_sampling


features_importance, not_important = boruta_selected()

X_train = X_train[features_importance]
X_test = X_test[features_importance]
features_importance
