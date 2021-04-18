# CE903 Group Project Framingham Heart study
> Constructing a predictive model to predict ten year risk of coronary heart disease.

## Table of contents
* [General info](#general-info)
* [Images](#images)
* [Setup](#setup)
* [How to](#how-to)
* [Code examples](#code-examples)
* [External resources](#external-resources-used)
* [Status](#status)
* [Contact](#contact)

## General info
Research with the intention to accurately predict the overall risk i.e. whether a patient is at risk or not, of coronary heart disease in 10 years
The dataset contains 4230 total patients and 15 attributes, after getting rid of missing values this is left with 3658 total samples.
The output is stored under the variable 'TenYearCHD' and shows a class imbalance before and after dropping missing values :

* Class 0 | no risk | 3101 | 84.8%
* Class 1 | ten year risk | 557 | 15.2%

This label imbalance can be seen in the following plot:<br/>
![label_balance](https://github.com/confusedolive/CE903-Framingham-Heart-study/blob/main/code/visualization/label%20balance.png?raw=true)

We explore different predictive classification models  with the goal of tuning a model which can accurately classify both classes,
for this the focus metrics of evaluation will be:

* Precision
* Recall
* F1 score
* Area under the curve (AUC) for the precision recall curve

This metrics are chosen on the basis of dealing with an imbalanced dataset , in order to deal with this imbalanced the main techniques chosen are Synthetic Minority Oversampling Technique (SMOTE) and Random Under Sampling. <br/>
It is important as well to note that  recall will be prioritised over precision as given the nature of the research False negatives (1 classified as 0) can be more dangerous than False positives(0 classified as 1).
## Images
A heatmap showing the correlation between variables in the dataset
![Examples](https://github.com/confusedolive/CE903-Framingham-Heart-study/blob/main/code/visualization/variables%20correlation%20heatmap.png?raw=true)

## Setup
* Python= 3.x.x
* Pandas=1.2.1
* ScikitLearn=0.24.1
* Numpy=1.19.5
* Matplotlib=3.2.2
* Seaborn=0.11.1
* Scipy=1.6.0
* boruta=0.3
><br/>can be installed using:<br/>
>> pip install -r /path/to/requirements.txt

## How to
This repository is separated into three folders:

* code
* data
* example

with in the code folder there is another folder called visualization which contains all graphs produced, the code folder also containes 2 .py files:

* modelling x.x.py
* visualize.py

In modelling all pre-processing, feature selecting, hyperparameter tuning and evaluating is carried out.
Visualize contains all the code utilized to create the graphs found in the folder visualization.
In the folder data the Framingham Heart Study dataset can be found.
In examples there are screen shots showing how to use functions found in modelling.py


## Code Examples
Example of the results Utilizing the evaluate_n_models function available in modelling 2.1.py, shows the first report printed in the first three images , last image shows the pandas dataframe produced out of the scores of all tested models:
![n_modles](https://github.com/confusedolive/CE903-Framingham-Heart-study/blob/main/example/example%20evaluate%20n%20models.PNG?raw=true)
![n_modles2](https://github.com/confusedolive/CE903-Framingham-Heart-study/blob/main/example/example%20evaluate%20n%20models%202.PNG?raw=true)
![n_modles3](https://github.com/confusedolive/CE903-Framingham-Heart-study/blob/main/example/example%20evaluate%20n%20models%203.PNG?raw=true)
![n_modles4](https://github.com/confusedolive/CE903-Framingham-Heart-study/blob/main/example/all%20scores.PNG?raw=true)

## External resources used
* https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset, access to dataset through Kaggle.

## To-do list:
- [x] Visualize the dataset
- [x] More research on the specific dataset
- [x] Test different models
- [x] Find optimal model
- [ ] Fine tune model
- [ ] Write report

## Status
Project is: _in progress_<br/>
Currently fine tuning model and preparing report.


## Contact
Created by: PUT NAMES HERE<br/> E-mail: PUT EMAILS HERE
