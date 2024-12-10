import numpy as np
import csv
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
heartDisease = pd.read_csv ("C:/Users/Lochan/OneDrive/Documents/P6Data.csv")
heartDisease = heartDisease.replace('?',np.nan)
print('Few examples from the dataset are given below')
print(heartDisease.head())

model=BayesianModel([('age','Heartdisease'),('gender','Heartdisease'),('exang','Heartdisease'),
('cp','Heartdisease'),('Heartdisease','restecg'),('Heartdisease','chol')])
print('\n Learning CPD using Maximum likelihood estimators')
model.fit (heartDisease,estimator=MaximumLikelihoodEstimator)
print ('\n Inferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)
print ('\n 1. Probability of HeartDisease given evidence= restecg')
q1=HeartDiseasetest_infer.query(variables=['Heartdisease'],evidence={'age':35})
print(q1)
print('\n 2. Probability of HeartDisease given evidence= cp ')
q2=HeartDiseasetest_infer.query(variables=['Heartdisease'],evidence={'chol':250})
print(q2)