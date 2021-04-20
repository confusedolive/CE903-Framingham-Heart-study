import numpy   as np
import pandas  as pd
import seaborn as sns
from boruta  import BorutaPy
from sklearn import metrics
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from imblearn.over_sampling  import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline       import Pipeline
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import plot_confusion_matrix
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import BaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import classification_report
from mlxtend.classifier      import EnsembleVoteClassifier
from sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.utils           import compute_class_weight

data_path = r'C:\Users\jeron\OneDrive\Desktop\903group\data\framingham.csv'

data = pd.read_csv(data_path)
data_heart = data.copy()
data_heart.dropna(inplace=True)
data.drop(columns=['currentSmoker', 'education'],axis=1, inplace=True)

output = 'TenYearCHD'
features = ['male', 'age',
        'cigsPerDay', 'BPMeds', 'prevalentStroke',
        'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
        'diaBP', 'BMI', 'heartRate', 'glucose']

not_standard = ['male', 'BPMeds', 'prevalentStroke',
                'prevalentHyp', 'diabetes', 'TenYearCHD']
need_standard = [x for x in features if x not in not_standard]
features_to_scale = data_heart[need_standard]

normalize =MinMaxScaler(feature_range=(0,1))
features_scaled = normalize.fit_transform(features_to_scale.values)

data_heart[need_standard] = features_scaled
X = data_heart[features]
y = data_heart[output]
#boruta py recommends using pruned forest with a depth between 3-7
randomclf = RandomForestClassifier(n_jobs=-1,
                                  max_depth=6,
                                  class_weight='balanced')

boruta_select = BorutaPy(randomclf, n_estimators='auto',
                              verbose=2, random_state=1)

boruta_select.fit(np.array(X), np.array(y))

features_importance = [X.columns[i] for i, boolean in enumerate(boruta_select.support_) if boolean]

not_important = [X.columns[i] for i , boolean in enumerate(boruta_select.support_) if not boolean]
print(not_important)
print(features_importance)

X = X[features_importance]

#15.2269% = 1
#84.7731% = 0

####################################################################################################

def plot_conf(model):
    conf = plot_confusion_matrix(model, X_test, y_test,
                             display_labels=y.unique(),
                             cmap= plt.cm.Blues,
                              normalize='true')
    plt.show()

####################################################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

results_acc = {}
results_f1  = {}
######################################logistic regression###############################################################
###paremeter tuning for logistic regression
params = [
        {'penalty': ['l1'],'C': np.logspace(-4, 4, 50),
        'solver': [ 'liblinear','saga'],'class_weight':['balanced'],
        'max_iter':[100, 1000, 1500,5000]},
        {'penalty': ['l2'],'C': np.logspace(-4, 4, 50),
         'solver': ['newton-cg','sag','lbfgs'],'class_weight':['balanced'],
         'max_iter':[100, 1000, 1500,5000]},
        {'penalty': ['elasticnet'],'class_weight':['balanced'],'C': np.logspace(-4, 4, 50),
         'solver': ['saga'], 'max_iter':[100, 1000, 1500,5000],
         'l1_ratio':[0.5]}]

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
log = LogisticRegression()
grid = GridSearchCV(log, params, cv=cv)
grid.fit(X, y)
print(grid.best_estimator_)

weights = {0:0.5, 1:0.95}
log = LogisticRegression(C=0.0011, class_weight='balanced', penalty='l2',
                   solver='lbfgs')
log.fit(X_train,y_train)
print(log.predict(X_test))
print(classification_report(y_train, log.predict(X_train)))
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scores = cross_val_score(log, X_test, y_test, scoring='recall', cv=cv, n_jobs=-1)

print(scores.mean())


results_acc['logmodel'] = accscore
results_f1['logmodel']  = f1scor

sns.set_style('dark')
sns.set_context('paper', font_scale=1.4)
plot_conf(logmodel)

#################################suppor vector hyperspace######################################
#parameter tuning for support vector machine
param_grid = {'C': [0.1,1, 10, 100],
            'gamma': [1,0.1,0.01,0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(), param_grid, verbose=2)
grid.fit(X_train, y_train)
print(grid.best_estimator_)

svc = SVC(C=0.1, gamma=0.1)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print(classification_report(y_test, y_pred))
f1scorsv = metrics.f1_score(y_test, y_pred)
accscorsv = svc.score(X_test, y_test)
plot_conf(svc)

results_acc['svc'] = accscorsv
results_f1['svc']  = f1scorsv
#########################Boosting#################################

log = LogisticRegression(solver='lbfgs', class_weight='balanced')
ada = AdaBoostClassifier(n_estimators=5, base_estimator=log)
grad_boost = GradientBoostingClassifier(n_estimators=100)
xgb = XGBClassifier(max_depth=8, learning_rate=0.001, use_label_encoder=False)

ensemble = EnsembleVoteClassifier(clfs = [ada, grad_boost, xgb], voting='hard')

ensemble.fit(X_train, y_train)

y_preden = ensemble.predict(X_test)
f1scoren = metrics.f1_score(y_test, y_preden)
accscoren = ensemble.score(X_test, y_test)
results_acc['ensemble'] = accscoren
results_f1['ensemble']  = f1scoren

print(classification_report(y_test, y_pred))
plot_conf(ensemble)
###############################################################################

naive = GaussianNB(var_smoothing=2e-9)
naive.fit(X_train, y_train)

y_pred  = naive.predict(X_test)
f1scornb = metrics.f1_score(y_test, y_pred)
accscornb = naive.score(X_test, y_test)
results_acc['NB']  = accscornb
results_f1['NB']   = f1scornb
plot_conf(naive)


##########################Comparing models############################
def compare_model_acc():
    df    = pd.DataFrame(results_acc, index=['acc'])
    df_t  = df.T
    df_t['model'] = df_t.index

    sns.barplot(x = 'model', y='acc', data=df_t)
    plt.title('Models accuracy')
    plt.show()

def compare_model_f1():
    df    = pd.DataFrame(results_f1, index=['f1'])
    df_t  = df.T
    df_t['model'] = df_t.index

    sns.barplot(x = 'model', y='f1', data=df_t)
    plt.title('Models F1 score')
    plt.show()

compare_model_f1()
compare_model_acc()
