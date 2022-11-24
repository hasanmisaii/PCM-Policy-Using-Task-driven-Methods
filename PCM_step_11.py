# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:34:43 2021

@author: misaiiha
generate masked data using random.multinomial (Time and cause dependent model)
"""

import numpy as np
import pandas as pd
import itertools as ite
from scipy.stats import weibull_min
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# evaluate random forest algorithm for classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.integrate import quad
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
# from matplotlib import pyplot
from numpy import exp
from sklearn.neighbors import KernelDensity
import scipy.optimize as optimize
import math
from imblearn.over_sampling import SMOTE


print("Hasan")
n = int(input("Enter number of data that should be generated ('n'):  \n"))  # number of samples

J = int(input("Enter number of system component(s): \n"))
arr_0 = input("Enter shape parameters by space: \n")   # takes the whole line of shape numbers by space
beta = list(map(float, arr_0.split(' ')))  # split those numbers with space( becomes ['2','3','6','6','5']) and then map every element into float (becomes [2,3,6,6,5])

arr_1 = input("Enter scale parameters by space: \n")   # takes the whole line of scale numbers by space
lam = list(map(float, arr_1.split(' ')))  # split those numbers with space( becomes ['2','3','6','6','5']) and then map every element into int (becomes [2,3,6,6,5])

# costs
cost = input("Enter costs for (Ins., Unavail., PCM) by space: \n")
cost = list(map(float, cost.split(' ')))  # split those numbers with space( becomes ['2','3','6','6','5']) and then map every element into float (becomes [2,3,6,6,5])
cost_ins = cost[0]
cost_unavail = cost[1]
cost_pcm = cost[2]

# %% failure cause
k_0 = list(ite.repeat(0, n))  # exact cause failure
k_1 = list(ite.repeat(0, n))  # Second cause failure
k_2 = list(ite.repeat(0, n))  # Third cause failure
t = list(ite.repeat(0, n))  # time to failure

t_0 = weibull_min.rvs(c=beta[0], loc=0, scale=lam[0], size=n)  # lifetimes of component 1
t_1 = weibull_min.rvs(c=beta[1], loc=0, scale=lam[1], size=n)  # lifetimes of component 2
t_2 = weibull_min.rvs(c=beta[2], loc=0, scale=lam[2], size=n)  # lifetimes of component 3

for i in range(n):
    t[i] = min(t_0[i], t_1[i], t_2[i])
    if i+1 >= n:
        if t[i] == t_0[i]:
            k_0[i] = "k_0"
        if t[i] == t_1[i]:
            k_0[i] = "k_1"
        if t[i] == t_2[i]:
            k_0[i] = "k_2"
        break
    if t[i] == t_0[i]:
        k_0[i] = "k_0"
        t_0[i+1] = weibull_min.rvs(beta[0], loc=0, scale=lam[0], size=1)
    else:
        t_0[i+1] = t_0[i]
    if t[i] == t_1[i]:
        k_0[i] = "k_1"
        t_1[i+1] = weibull_min.rvs(beta[1], loc=0, scale=lam[1], size=1)
    else:
        t_1[i+1] = t_1[i]
    if t[i] == t_2[i]:
        k_0[i] = "k_2"
        t_2[i+1] = weibull_min.rvs(beta[2], loc=0, scale=lam[2], size=1)
    else:
        t_2[i+1] = t_2[i]
# %% failure cause
k_1 = list(ite.repeat(0, n))  # Second cause failure
k_2 = list(ite.repeat(0, n))  # Third cause failure

for i in range(n):
    if t_0[i] < t_1[i] < t_2[i]:
        k_0[i] = "k_0"
        k_1[i] = "k_1"
        k_2[i] = "k_2"
    if t_2[i] < t_1[i] < t_0[i]:
        k_0[i] = "k_2"
        k_1[i] = "k_1"
        k_2[i] = "k_0"
    if t_1[i] < t_0[i] < t_2[i]:
        k_0[i] = "k_1"
        k_1[i] = "k_0"
        k_2[i] = "k_2"
    if t_1[i] < t_0[i] < t_2[i]:
        k_0[i] = "k_1"
        k_1[i] = "k_0"
        k_2[i] = "k_2"
    if t_2[i] < t_0[i] < t_1[i]:
        k_0[i] = "k_2"
        k_1[i] = "k_0"
        k_2[i] = "k_1"
    if t_0[i] < t_2[i] < t_2[i]:
        k_0[i] = "k_0"
        k_1[i] = "k_2"
        k_2[i] = "k_1"
    if t_1[i] < t_2[i] < t_0[i]:
        k_0[i] = "k_1"
        k_1[i] = "k_2"
        k_2[i] = "k_0"
# %% generate data as a data frame
data = {'time_to_failure': t,
        'cause_to_failure': k_0
        }
df = pd.DataFrame(data)
df['cause_to_failure'] = pd.Categorical(df.cause_to_failure)
print(df)
# visualization
print('The shape of our data is:', df.shape)
print('Describe all data:', df.describe(include='all'))
print('Describe categorical data:', df.describe(include=['category']))
print('Describe all data info:', df.info())
df.boxplot()
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 5))
df.groupby('cause_to_failure').size().plot(kind='pie', autopct='%.2f', textprops={'fontsize': 20}, colors=['tomato', 'gold', 'skyblue'], ax=ax1)
ax1.set_ylabel('cause_to_failure', size=15)
plt.tight_layout()
plt.show()
# %% Real Cost Rate


def integrand_0(x, a, b):
    return x * weibull_min.pdf(x, c=a, loc=0, scale=b)


def integrand_1(x, a, b):
    return weibull_min.pdf(x, c=a, loc=0, scale=b)


# Generate Masked sets
k = 100
M = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
MM = list(ite.repeat(0, k))
for i in range(k):
    p_val = [1/7]*7
    ii = np.random.multinomial(1, p_val, size=1)
    MM[i] = M[np.argmax(ii)]


def long_run_cost_realM(k, tau):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        a = beta[j]
        b = lam[j]
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            # print(mm, i)
            midpoint = quad(integrand_0, (i)*tau, (i+1)*tau, args=(a, b))[0]
            R = 1
            for jj in range(J):
                if jj != j:
                    R *= quad(integrand_1, (i+1)*tau, math.inf, args=(a, b))[0]
            # print(midpoint)
            # if midpoint <= tau:
            ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                quad(integrand_1, (i)*tau, (i+1)*tau, args=(a, b))[0] * R
            ex_time[j] += (i+1) * tau * quad(integrand_1, (i)*tau, (i+1)*tau, args=(a, b))[0] * R
        # print(ex_cost[j])
        # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_realM_v = np.vectorize(long_run_cost_realM)
# long_run_cost_realM_v(10, 0.2)

tau_vals = np.linspace(2.5, 30, 10)
result_real_M = long_run_cost_realM_v(k, tau_vals)
result_real_M

plt.plot(tau_vals, result_real_M)
plt.ylabel('OPTIMAL')
plt.show()
# %% Parametric without ML


def Estimate_Par(t):
    def loglik(parms):
        S = 0
        a, b = parms
        for i in range(len(t)):
            S += np.log(weibull_min.pdf(t[i], c=a, loc=0, scale=b))
        return(-1*S)

    initial_guess = [1, 1]
    result = optimize.minimize(loglik, initial_guess, method='L-BFGS-B')
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)
    return(fitted_params)


t_k_0 = np.array(df['time_to_failure'][df['cause_to_failure'] == 'k_0'])
t_k_1 = np.array(df['time_to_failure'][df['cause_to_failure'] == 'k_1'])
t_k_2 = np.array(df['time_to_failure'][df['cause_to_failure'] == 'k_2'])

plt.hist(t, bins=50, density=True)
plt.hist(t_k_0, bins=50, density=True)
plt.hist(t_k_1, bins=50, density=True)
plt.hist(t_k_2, bins=50, density=True)


def long_run_cost_E_PM(k, tau):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            a, b = Estimate_Par(t_k_0)
        if j == 1:
            a, b = Estimate_Par(t_k_1)
        if j == 2:
            a, b = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            # print(mm, i)
            midpoint = quad(integrand_0, (i)*tau, (i+1)*tau, args=(a, b))[0]
            R = 1
            for jj in range(J):
                if jj != j:
                    R *= quad(integrand_1, (i+1)*tau, math.inf, args=(a, b))[0]
            # print(midpoint)
            # if midpoint <= tau:
            ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                quad(integrand_1, (i)*tau, (i+1)*tau, args=(a, b))[0] * R
            ex_time[j] += (i+1) * tau * quad(integrand_1, (i)*tau, (i+1)*tau, args=(a, b))[0] * R
        # print(ex_cost[j])
        # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_E_P_vM = np.vectorize(long_run_cost_E_PM)
long_run_cost_E_P_vM(100, 0.2)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_EPM = long_run_cost_E_P_vM(100, tau_vals)


plt.plot(tau_vals, result_real_M)
plt.plot(tau_vals, result_estimate_EPM)
plt.ylabel('OPTIMAL')
plt.show()
# %% RF for cause to failure
X = df[['time_to_failure']]
y = df[['cause_to_failure']]
# One-hot encode the data using pandas get_dummies
# X = pd.get_dummies(X)
# Display the first 5 rows of the last 12 columns
# X.iloc[:,5:].head(5)
# reshape data
# Imbanaced Classification
# transform the dataset
X, y = oversampler = SMOTE(k_neighbors=1).fit_resample(X, y)
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# define the model
model_cause_RF = RandomForestClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model_cause_RF, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy RF with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification

# fit the model on the train dataset
model_cause_RF.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_RF.predict(X_test)
k_prob = model_cause_RF.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)
print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric RF
X_balanced = list(X['time_to_failure'])
y_balanced = list(y['cause_to_failure'])

data_balanced = {'time_to_failure': X_balanced,
                 'cause_to_failure': y_balanced
                 }
df_balanced = pd.DataFrame(data_balanced)
df_balanced['cause_to_failure'] = pd.Categorical(df_balanced.cause_to_failure)
print(df_balanced)


t_k_0 = np.array(df_balanced['time_to_failure'][df_balanced['cause_to_failure'] == 'k_0'])
t_k_1 = np.array(df_balanced['time_to_failure'][df_balanced['cause_to_failure'] == 'k_1'])
t_k_2 = np.array(df_balanced['time_to_failure'][df_balanced['cause_to_failure'] == 'k_2'])


def long_run_cost_estimate_PRFM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / partition) * model_cause_RF.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PRFM_v = np.vectorize(long_run_cost_estimate_PRFM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_PRFM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PRFM = long_run_cost_estimate_PRFM_v(k, tau_vals, 100)
result_estimate_PRFM


plt.plot(tau_vals, result_real_M)
plt.plot(tau_vals, result_estimate_PRFM)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric RF
sample = np.asarray(t)
plt.hist(sample, bins=50, density=True)
plt.show()

model = KernelDensity(bandwidth=1, kernel='exponential')
sample = sample.reshape((len(sample), 1))
model.fit(sample)

...
# sample probabilities for a range of outcomes
values = np.asarray([value for value in range(0, len(sample))])
values = values.reshape((len(values), 1))
probabilities = model.score_samples(values)
probabilities = exp(probabilities)

...
# plot the histogram and pdf
plt.hist(sample, bins=50, density=True)
plt.plot(values[:], probabilities)
plt.show()


model = KernelDensity(bandwidth=0.1, kernel='exponential')
sample = X
model.fit(X)

model_0 = KernelDensity(bandwidth=0.1, kernel='exponential')


def long_run_cost_estimate_NPRFM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / partition) * model_cause_RF.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPRFM_v = np.vectorize(long_run_cost_estimate_NPRFM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_NPRFM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPRFM = long_run_cost_estimate_NPRFM_v(k, tau_vals, 100)
result_estimate_NPRFM

plt.plot(tau_vals, result_real_M, color="b", ls='-')
plt.plot(tau_vals, result_estimate_PRFM, color="g", ls='--')
plt.plot(tau_vals, result_estimate_NPRFM, color="r", ls='-.')
plt.plot(tau_vals, result_estimate_EPM, color="black", ls=':')
plt.legend(["Real", "PRF", "NPRF", "PE"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
plt.show()
# %% KNN
# %% KNN for cause to failure
# define the model
model_cause_knn = KNeighborsClassifier(n_neighbors=5)
# evaluate the model
n_scores = cross_val_score(model_cause_knn, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy KNN with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_knn.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_knn.predict(X_test)
k_prob = model_cause_knn.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)
print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric KNN


def long_run_cost_estimate_PKNNM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / partition) * model_cause_knn.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PKNNM_v = np.vectorize(long_run_cost_estimate_PKNNM)
long_run_cost_realM_v(100, 0.2)
long_run_cost_estimate_PKNNM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PKNNM = long_run_cost_estimate_PKNNM_v(k, tau_vals, 100)
result_estimate_PKNNM


plt.plot(tau_vals, result_real_M)
plt.plot(tau_vals, result_estimate_PKNNM)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric KNN
# %% Estimate MAINTENANCE Cost Non-Parametric KNN


def long_run_cost_estimate_NPKNNM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / partition) * model_cause_knn.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPKNNM_v = np.vectorize(long_run_cost_estimate_NPKNNM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_NPKNNM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPKNNM = long_run_cost_estimate_NPKNNM_v(k, tau_vals, 100)
result_estimate_NPKNNM

plt.plot(tau_vals, result_real_M, color="b", ls='-')
plt.plot(tau_vals, result_estimate_PKNNM, color="g", ls='--')
plt.plot(tau_vals, result_estimate_NPKNNM, color="r", ls='-.')
plt.plot(tau_vals, result_estimate_EPM, color="black", ls=':')
plt.legend(["Real", "PKNN", "NPKNN", "PE"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
plt.show()
# %% NB for cause to failure
# define the model
model_cause_gnb = GaussianNB()
# evaluate the model
n_scores = cross_val_score(model_cause_gnb, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_gnb.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_gnb.predict(X_test)
k_prob = model_cause_gnb.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)
print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric GNB


def long_run_cost_estimate_PGNBM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / partition) * model_cause_gnb.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PGNBM_v = np.vectorize(long_run_cost_estimate_PGNBM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_PGNBM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PGNBM = long_run_cost_estimate_PGNBM_v(k, tau_vals, 100)
result_estimate_PGNBM

plt.plot(tau_vals, result_real_M)
plt.plot(tau_vals, result_estimate_PGNBM)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric GNB


def long_run_cost_estimate_NPGNBM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / partition) * model_cause_gnb.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPGNBM_v = np.vectorize(long_run_cost_estimate_NPGNBM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_NPGNBM_v(100, 0.2, 100)
# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPGNBM = long_run_cost_estimate_NPGNBM_v(k, tau_vals, 100)
result_estimate_NPGNBM

plt.plot(tau_vals, result_real_M, color="b", ls='-')
plt.plot(tau_vals, result_estimate_PGNBM, color="g", ls='--')
plt.plot(tau_vals, result_estimate_NPGNBM, color="r", ls='-.')
plt.plot(tau_vals, result_estimate_EPM, color="black", ls=':')
plt.legend(["Real", "PNB", "NPNB", "PE"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
# %% SVM for cause to failure
# define the model
model_cause_svm = svm.SVC(probability=True)
# evaluate the model
n_scores = cross_val_score(model_cause_svm, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy SVM with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_svm.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_svm.predict(X_test)
k_prob = model_cause_svm.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)

print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric SVM


def long_run_cost_estimate_PSVMM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / partition) * model_cause_svm.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PSVMM_v = np.vectorize(long_run_cost_estimate_PSVMM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_PSVMM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PSVMM = long_run_cost_estimate_PSVMM_v(k, tau_vals, 100)
result_estimate_PSVMM


plt.plot(tau_vals, result_real_M)
plt.plot(tau_vals, result_estimate_PSVMM)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric SVM


def long_run_cost_estimate_NPSVMM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / partition) * model_cause_svm.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPSVMM_v = np.vectorize(long_run_cost_estimate_NPSVMM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_NPSVMM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPSVMM = long_run_cost_estimate_NPSVMM_v(k, tau_vals, 100)
result_estimate_NPSVMM

plt.plot(tau_vals, result_real_M, color="b", ls='-')
plt.plot(tau_vals, result_estimate_PSVMM, color="g", ls='--')
plt.plot(tau_vals, result_estimate_NPSVMM, color="r", ls='-.')
plt.plot(tau_vals, result_estimate_EPM, color="black", ls=':')
plt.legend(["Real", "PSVM", "NPSVM", "PE"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
plt.show()
# %% LDA
# %% LDA for cause to failure
# define the model
model_cause_lda = LinearDiscriminantAnalysis()
# evaluate the model
n_scores = cross_val_score(model_cause_lda, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy LDA with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_lda.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_lda.predict(X_test)
k_prob = model_cause_lda.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)
print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric LDA


def long_run_cost_estimate_PLDAM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / partition) * model_cause_lda.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PLDAM_v = np.vectorize(long_run_cost_estimate_PLDAM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_PLDAM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PLDAM = long_run_cost_estimate_PLDAM_v(k, tau_vals, 100)
result_estimate_PLDAM


plt.plot(tau_vals, result_real_M)
plt.plot(tau_vals, result_estimate_PLDAM)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric LDA


def long_run_cost_estimate_NPLDAM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / partition) * model_cause_lda.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPLDAM_v = np.vectorize(long_run_cost_estimate_NPLDAM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_NPLDAM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPLDAM = long_run_cost_estimate_NPLDAM_v(k, tau_vals, 100)
result_estimate_NPLDAM

plt.plot(tau_vals, result_real_M, color="b", ls='-')
plt.plot(tau_vals, result_estimate_PLDAM, color="g", ls='--')
plt.plot(tau_vals, result_estimate_NPLDAM, color="r", ls='-.')
plt.plot(tau_vals, result_estimate_EPM, color="black", ls=':')
plt.legend(["Real", "PLDA", "NPLDA", "PE"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
plt.show()
# %% Decission Tree
# %% DT for cause to failure
# define the model
model_cause_dt = tree.DecisionTreeClassifier()
# evaluate the model
n_scores = cross_val_score(model_cause_dt, X_train, y_train, scoring='accuracy', cv=cv)
# report performance
print('Accuracy DT with CV: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# make predictions using random forest for classification
# fit the model on the whole dataset
model_cause_dt.fit(X_train, y_train)
# make a single prediction
k_hat = model_cause_dt.predict(X_test)
k_prob = model_cause_dt.predict_proba(X_test)
# confusion matrix
labels = np.unique(y_test)
a = confusion_matrix(y_test, k_hat, labels=labels)
pd.DataFrame(a, index=labels, columns=labels)

print("k_hat", k_hat)
print("k_prob", k_prob)
# %% Estimate MAINTENANCE Cost Parametric DT


def long_run_cost_estimate_PDTM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    a, b = Estimate_Par(t)
    for j in range(J):
        if j == 0:
            a_0, b_0 = Estimate_Par(t_k_0)
        if j == 1:
            a_0, b_0 = Estimate_Par(t_k_1)
        if j == 2:
            a_0, b_0 = Estimate_Par(t_k_2)
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * weibull_min.pdf(mid, c=a_0, loc=0, scale=b_0)
                cycle_proba += (tau / partition) * model_cause_dt.predict_proba([[mid]])[0][j] *\
                    weibull_min.pdf(mid, c=a, loc=0, scale=b)
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_PDTM_v = np.vectorize(long_run_cost_estimate_PDTM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_PDTM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_PDTM = long_run_cost_estimate_PDTM_v(k, tau_vals, 100)
result_estimate_PDTM


plt.plot(tau_vals, result_real_M)
plt.plot(tau_vals, result_estimate_PDTM)
plt.ylabel('OPTIMAL')
plt.show()
# %% Estimate MAINTENANCE Cost Nonparametric DT


def long_run_cost_estimate_NPDTM(k, tau, partition):
    ex_cost = list(ite.repeat(0, J))
    ex_time = list(ite.repeat(0, J))
    for j in range(J):
        if j == 0:
            model_0.fit(t_k_0.reshape(-1, 1))
        if j == 1:
            model_0.fit(t_k_1.reshape(-1, 1))
        if j == 2:
            model_0.fit(t_k_2.reshape(-1, 1))
        for i in range(k):
            mm = 0
            for m in range(i+1):
                mm += (j in MM[m])
            midpoint = 0
            cycle_proba = 0
            p = np.linspace(i*tau, (i+1)*tau, partition + 1)
            n = 1
            while n <= (len(p)-1):
                mid = (p[n-1] + p[n])/2
                midpoint += (tau / partition) * mid * exp(model_0.score_samples([[mid]]))
                cycle_proba += (tau / partition) * model_cause_dt.predict_proba([[mid]])[0][j] *\
                    exp(model.score_samples([[mid]]))
                # print(midpoint)
                n += 1
            if midpoint <= tau:
                ex_cost[j] += (mm * cost_ins/J + ((i+1) * tau - midpoint) * cost_unavail + cost_pcm) *\
                            cycle_proba
                ex_time[j] += (i+1) * tau * cycle_proba
                # print(ex_cost[j])
                # print(ex_time[j])
        if ex_time[j] == 0:
            ex_time[j] = 1
            ex_cost[j] = 0
            print("Hasan, Maybe there is a nan")
    res = [i / j for i, j in zip(ex_cost, ex_time)]
    return(sum(res))


long_run_cost_estimate_NPDTM_v = np.vectorize(long_run_cost_estimate_NPDTM)
# long_run_cost_realM_v(100, 0.2)
# long_run_cost_estimate_NPDTM_v(100, 0.2, 100)

# tau_vals = np.linspace(10, 200, 100)
result_estimate_NPDTM = long_run_cost_estimate_NPDTM_v(k, tau_vals, 100)
result_estimate_NPDTM

plt.plot(tau_vals, result_real_M, color="b", ls='-')
plt.plot(tau_vals, result_estimate_PDTM, color="g", ls='--')
plt.plot(tau_vals, result_estimate_NPDTM, color="r", ls='-.')
plt.plot(tau_vals, result_estimate_EPM, color="black", ls=':')
plt.legend(["Real", "PDT", "NPDT", "PE"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=4, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
plt.show()
# %% Parametric
plt.plot(tau_vals, result_real_M, color="b")
plt.axvline(x=tau_vals[result_real_M == np.min(result_real_M)], color="b")
plt.plot(tau_vals, result_estimate_PRFM, linestyle=(1, (1, 1)), color="r")
plt.axvline(x=tau_vals[result_estimate_PRFM == np.min(result_estimate_PRFM)], linestyle=(1, (1, 1)), color="r")
plt.plot(tau_vals, result_estimate_PSVMM, linestyle=(2, (2, 1)), color="g")
plt.axvline(x=tau_vals[result_estimate_PSVMM == np.min(result_estimate_PSVMM)], linestyle=(2, (2, 1)), color="g")
plt.plot(tau_vals, result_estimate_PKNNM, linestyle=(3, (3, 1)), color="k")
plt.axvline(x=tau_vals[result_estimate_PKNNM == np.min(result_estimate_PKNNM)], linestyle=(3, (3, 1)), color="k")
plt.plot(tau_vals, result_estimate_PGNBM, linestyle=(4, (4, 1)), color="y")
plt.axvline(x=tau_vals[result_estimate_PGNBM == np.min(result_estimate_PGNBM)], linestyle=(4, (4, 1)), color="y")
plt.plot(tau_vals, result_estimate_PLDAM, linestyle=(5, (5, 1)), color="m")
plt.axvline(x=tau_vals[result_estimate_PLDAM == np.min(result_estimate_PLDAM)], linestyle=(5, (5, 1)), color="m")
plt.plot(tau_vals, result_estimate_PDTM, linestyle=(6, (6, 1)), color="c")
plt.axvline(x=tau_vals[result_estimate_PDTM == np.min(result_estimate_PDTM)], linestyle=(6, (6, 1)), color="c")
plt.legend(["Real", "PRF", "PSVM", "PKNN", "PNB", "PLDA", "PDT"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=3, fancybox=True, shadow=True)
# %% Non Parametric
plt.plot(tau_vals, result_real_M, color="b")
plt.axvline(x=tau_vals[result_real_M == np.min(result_real_M)], color="b")
plt.plot(tau_vals, result_estimate_NPRFM, linestyle=(1, (1, 1)), color="r")
plt.axvline(x=tau_vals[result_estimate_NPRFM == np.min(result_estimate_NPRFM)], linestyle=(1, (1, 1)), color="r")
plt.plot(tau_vals, result_estimate_NPSVMM, linestyle=(2, (2, 1)), color="g")
plt.axvline(x=tau_vals[result_estimate_NPSVMM == np.min(result_estimate_NPSVMM)], linestyle=(2, (2, 1)), color="g")
plt.plot(tau_vals, result_estimate_NPKNNM, linestyle=(3, (3, 1)), color="k")
plt.axvline(x=tau_vals[result_estimate_NPKNNM == np.min(result_estimate_NPKNNM)], linestyle=(3, (3, 1)), color="k")
plt.plot(tau_vals, result_estimate_NPGNBM, linestyle=(4, (4, 1)), color="y")
plt.axvline(x=tau_vals[result_estimate_NPGNBM == np.min(result_estimate_NPGNBM)], linestyle=(4, (4, 1)), color="y")
plt.plot(tau_vals, result_estimate_NPLDAM, linestyle=(5, (5, 1)), color="m")
plt.axvline(x=tau_vals[result_estimate_NPLDAM == np.min(result_estimate_NPLDAM)], linestyle=(5, (5, 1)), color="m")
plt.plot(tau_vals, result_estimate_NPDTM, linestyle=(6, (6, 1)), color="c")
plt.axvline(x=tau_vals[result_estimate_NPDTM == np.min(result_estimate_NPDTM)], linestyle=(6, (6, 1)), color="c")
plt.legend(["Real", "NPRF", "NPSVM", "NPKNN", "NPNB", "NPLDA", "NPDT"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=3, fancybox=True, shadow=True)
# %% Parametric
plt.plot(tau_vals, result_real_M, color="b")
plt.plot(tau_vals, result_estimate_PRFM, linestyle=(1, (1, 1)), color="r")
plt.plot(tau_vals, result_estimate_PSVMM, linestyle=(2, (2, 1)), color="g")
plt.plot(tau_vals, result_estimate_PKNNM, linestyle=(3, (3, 1)), color="k")
plt.plot(tau_vals, result_estimate_PGNBM, linestyle=(4, (4, 1)), color="y")
plt.plot(tau_vals, result_estimate_PLDAM, linestyle=(5, (5, 1)), color="m")
plt.plot(tau_vals, result_estimate_PDTM, linestyle=(6, (6, 1)), color="c")
plt.plot(tau_vals, result_estimate_EPM, linestyle=(6, (6, 1)), color="black")
plt.legend(["Real", "PRF", "PSVM", "PKNN", "PNB", "PLDA", "PDT", "PE"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=3, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
plt.show()
# %% Non Parametric
plt.plot(tau_vals, result_real_M, color="b")
plt.plot(tau_vals, result_estimate_NPRFM, linestyle=(1, (1, 1)), color="r")
plt.plot(tau_vals, result_estimate_NPSVMM, linestyle=(2, (2, 1)), color="g")
plt.plot(tau_vals, result_estimate_NPKNNM, linestyle=(3, (3, 1)), color="k")
plt.plot(tau_vals, result_estimate_NPGNBM, linestyle=(4, (4, 1)), color="y")
plt.plot(tau_vals, result_estimate_NPLDAM, linestyle=(5, (5, 1)), color="m")
plt.plot(tau_vals, result_estimate_NPDTM, linestyle=(6, (6, 1)), color="c")
plt.plot(tau_vals, result_estimate_EPM, linestyle=(6, (6, 1)), color="black")
plt.plot(tau_vals, result_estimate_EPM, linestyle=(6, (6, 1)), color="black")
plt.legend(["Real", "PRF", "PSVM", "PKNN", "PNB", "PLDA", "PDT", "PE"], loc="upper right",
           bbox_to_anchor=(1, 1), ncol=3, fancybox=True, shadow=True)
plt.ylabel('Rate of cost')
plt.xlabel('\u03C4')
plt.show()

input("Press Enter to End:  \n")
