import numpy   as np
import pandas  as pd
import seaborn as sns
from boruta  import BorutaPy
from sklearn import metrics
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from imblearn.over_sampling  import SMOTE
from imblearn.under_sampling import RandomUnderSampler
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
from sklearn.model_selection import GridSearchCV
from sklearn.utils           import shuffle

data_path = r'C:\Users\jeron\OneDrive\Desktop\903group\data\framingham.csv'

data = pd.read_csv(data_path)
data_heart = data.copy()
data_heart.dropna(inplace=True)
data.drop('currentSmoker',axis=1, inplace=True)

output = 'TenYearCHD'
features = ['male', 'age', 'education',
        'cigsPerDay', 'BPMeds', 'prevalentStroke',
        'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
        'diaBP', 'BMI', 'heartRate', 'glucose']

not_standard = ['male', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'TenYearCHD']
need_standard = [x for x in features if x not in not_standard]
features_to_scale = data_heart[need_standard]

scaler =StandardScaler()
normalize = MinMaxScaler()

features_scaled = scaler.fit_transform(features_to_scale.values)

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
print(features_importance)
not_important = [X.columns[i] for i , boolean in enumerate(boruta_select.support_) if not boolean]
print(not_important)

X = X[features_importance]
####################################################################################################

def plot_conf(model):
    conf = plot_confusion_matrix(model, X_test, y_test,
                             display_labels=y.unique(),
                             cmap= plt.cm.Blues,
                              normalize='true')
    plt.show()

####################################################################################################
standard =StandardScaler()
normalize = MinMaxScaler()

over = SMOTE()
under = RandomUnderSampler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
X_train, y_train = over.fit_resample(X_train, y_train)
X_train,y_train = under.fit_resample(X_train, y_train)


######################################logistic regression###############################################################
###paremeter tuning for logistic regression
params = [
        {'penalty': ['l1'],'C': np.logspace(-4, 4, 20),
        'solver': [ 'liblinear','saga'],
        'max_iter':[100, 1000, 1500,5000]},
        {'penalty': ['l2'],'C': np.logspace(-4, 4, 20),
         'solver': ['newton-cg','sag','lbfgs'],
         'max_iter':[100, 1000, 1500,5000]},
        {'penalty': ['elasticnet'],'C': np.logspace(-4, 4, 20),
         'solver': ['saga'], 'max_iter':[100, 1000, 1500,5000],
         'l1_ratio':[0.5]}]

log = LogisticRegression()
grid = GridSearchCV(log, param_grid=params, cv= 10, verbose=2, n_jobs= -1)
b = grid.fit(X_train, y_train)
print(b.best_estimator_)

logmodel = LogisticRegression(C=0.012742749857031334, penalty='l1',
                              solver='liblinear', max_iter = 1000,
                              random_state=42)

logmodel.fit(X_train, y_train)
print(classification_report(y_test, logmodel.predict(X_test)))
print(metrics.f1_score(y_test, logmodel.predict(X_test)))
print(logmodel.score(X_test, y_test))
sns.set_style('dark')
sns.set_context('paper', font_scale=1.4)
plot_conf(logmodel)

#################################suppor vector hyperspace######################################
svc = SVC(kernel='linear', C=100, gamma=1)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

plot_conf(svc)

d = classification_report(y_test, y_pred)
print(d)
print(metrics.f1_score(y_test, y_pred))

#########################Boosting#################################
svc1 = SVC(kernel='linear', probability=True, C=100, gamma=1)
ada = AdaBoostClassifier(n_estimators=5, base_estimator=svc1)
grad_boost = GradientBoostingClassifier(n_estimators=10)
xgb = XGBClassifier(max_depth=5, learning_rate=0.001, use_label_encoder=False)

ensemble = EnsembleVoteClassifier(clfs = [ada, grad_boost, xgb], voting='hard')

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
print(classification_report(y_test, y_pred))
plot_conf(ensemble)
###############################################################################


param_grid = {'C': [0.1,1, 10, 100],
            'gamma': [1,0.1,0.01,0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

print(grid.best_estimator_)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

naive = GaussianNB(var_smoothing=2e-9)
naive.fit(X_train, y_train)
print(naive.score(X_train, y_train))
print(naive.score(X_test, y_test))
y_pred = naive.predict(X_test)
print(metrics.f1_score(y_test, y_pred))
plot_conf(naive)
