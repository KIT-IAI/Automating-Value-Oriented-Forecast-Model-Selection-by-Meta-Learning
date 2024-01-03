import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier


from statsmodels.tsa.seasonal import seasonal_decompose


###Import PV and load data
prefix = 'data//data'
pv = pd.read_csv(prefix+'//solar_gg_all_2010-2013.csv', index_col='time', parse_dates = ['time'], infer_datetime_format = True)
load = pd.read_csv(prefix+'//solar_gc_all_2010-2013.csv', index_col='time', parse_dates = ['time'], infer_datetime_format = True) + \
pd.read_csv(prefix+'//solar_cl_all_2010-2013.csv', index_col='time', parse_dates = ['time'], infer_datetime_format = True)


###Generate different input features for each load_factor and pv_factor for the first two years
factor_pv = [1,1,1,1,1,10,5]
factor_load = [1,5,2,1/2,1/5,1,1]
final = pd.DataFrame()
for load_factor, pv_factor in zip(factor_load, factor_pv):
    load_temp = load_factor*load[pd.Timestamp(year=2010,month=7, day=1, minute=30):pd.Timestamp(year=2012,month=7, day=1)]
    pv_temp = pv_factor*pv[pd.Timestamp(year=2010,month=7, day=1, minute=30):pd.Timestamp(year=2012,month=7, day=1)]
    prosumption = load_temp - pv_temp
    ##Basic features: Summary statistic over all days for prosumption or load/PV seprately
    #df = pd.concat([load_temp.describe().T.drop('count', axis=1).set_axis(['load_' + load_temp.describe().T.drop('count', axis=1).columns], axis=1),\
    #                pv_temp.describe().T.drop('count', axis=1).set_axis(['pv_' + pv_temp.describe().T.drop('count', axis=1).columns], axis=1)], axis=1)
    df = prosumption.describe().T.drop('count', axis=1)
    ##Feature 6: Autocorrelation for prosumption or load/PV seprately
    #pv_autocorr = []
    #load_autocorr = []
    prosumption_autocorr = []
    for col in pv.columns:
        sumi = 0
        for lag in np.arange(7*48):
    #        sumi_pv = sumi + pv_temp[col].autocorr(lag)**2
    #        sumi_load = sumi + load_temp[col].autocorr(lag)**2
             sumi_prosumption = sumi + prosumption[col].autocorr(lag)**2
    #    pv_autocorr.append(sumi_pv)
    #    load_autocorr.append(sumi_load)
        prosumption_autocorr.append(sumi_prosumption)
    #df = pd.concat([pd.DataFrame(pv_autocorr, index=df.index, columns=['pv']),df], axis=1)
    #df = pd.concat([pd.DataFrame(load_autocorr, index=df.index, columns=['load']),df], axis=1)
    df = pd.concat([pd.DataFrame(prosumption_autocorr, index=df.index, columns=['load']),df], axis=1)
    ##Feature 5: Skewness and Kurtosis for prosumption or load/PV seprately
    #df = pd.concat([pd.DataFrame(pv_temp.skew(), index=df.index, columns=['pv_skew']),df], axis=1)
    #df = pd.concat([pd.DataFrame(load_temp.skew(), index=df.index, columns=['load_skew']),df], axis=1)
    #df = pd.concat([pd.DataFrame(pv_temp.kurtosis(), index=df.index, columns=['pv_kurt']),df], axis=1)
    #df = pd.concat([pd.DataFrame(load_temp.kurtosis(), index=df.index, columns=['load_kurt']),df], axis=1)
    #df = pd.concat([pd.DataFrame(prosumption.skew(), index=df.index, columns=['prosumption_skew']),df], axis=1)
    #df = pd.concat([pd.DataFrame(prosumption.kurtosis(), index=df.index, columns=['prosumption_kurt']),df], axis=1)
    ##Feature 4: Seasonality and Trend for prosumption or load/PV seprately
    #df = pd.concat([pd.DataFrame(1 - np.var(pv_temp.values - seasonal_decompose(pv_temp.values, model='additive', period=52*48).seasonal - \
    #                           seasonal_decompose(pv_temp.values, model='additive', period=52*48, extrapolate_trend='freq').trend, axis=0)/\
    #                    np.var(pv_temp.values - seasonal_decompose(pv_temp.values, model='additive', period=52*48, extrapolate_trend='freq').trend, axis=0),\
    #                             index=df.index, columns=['pv_season']), df], axis=1)
    #df = pd.concat([pd.DataFrame(1 - np.var(pv_temp.values - seasonal_decompose(pv_temp.values, model='additive', period=52*48).seasonal - \
    #                           seasonal_decompose(pv_temp.values, model='additive', period=52*48, extrapolate_trend='freq').trend, axis=0)/\
    #                   np.var(pv_temp.values - seasonal_decompose(pv_temp.values, model='additive', period=52*48).seasonal, axis=0), \
    #                             index=df.index, columns=['pv_trend']), df], axis=1)
    #df = pd.concat([pd.DataFrame(1 - np.var(load_temp.values - seasonal_decompose(load_temp.values, model='additive', period=52*48).seasonal - \
    #                           seasonal_decompose(load_temp.values, model='additive', period=52*48, extrapolate_trend='freq').trend, axis=0)/\
    #                    np.var(load_temp.values - seasonal_decompose(load_temp.values, model='additive', period=52*48, extrapolate_trend='freq').trend, axis=0), \
    #                             index=df.index, columns=['load_season']), df], axis=1)
    #df = pd.concat([pd.DataFrame(1 - np.var(load_temp.values - seasonal_decompose(load_temp.values, model='additive', period=52*48).seasonal - \
    #                           seasonal_decompose(load_temp.values, model='additive', period=52*48, extrapolate_trend='freq').trend, axis=0)/\
    #                    np.var(load_temp.values - seasonal_decompose(load_temp.values, model='additive', period=52*48).seasonal, axis=0), \
    #                             index=df.index, columns=['load_trend']), df], axis=1)
    #df = pd.concat([pd.DataFrame(1 - np.var(prosumption.values - seasonal_decompose(prosumption.values, model='additive', period=52*48).seasonal - \
    #                           seasonal_decompose(prosumption.values, model='additive', period=52*48, extrapolate_trend='freq').trend, axis=0)/\
    #                   np.var(prosumption.values - seasonal_decompose(prosumption.values, model='additive', period=52*48).seasonal, axis=0), \
    #                             index=df.index, columns=['prosumption_trend']), df], axis=1)
    #df = pd.concat([pd.DataFrame(1 - np.var(prosumption.values - seasonal_decompose(prosumption.values, model='additive', period=52*48).seasonal - \
    #                           seasonal_decompose(prosumption.values, model='additive', period=52*48, extrapolate_trend='freq').trend, axis=0)/\
    #                    np.var(prosumption.values - seasonal_decompose(prosumption.values, model='additive', period=52*48, extrapolate_trend='freq').trend, axis=0), \
    #                             index=df.index, columns=['prosumption_season']), df], axis=1)
    ##Feature 3: Mean over all days of Min/Max for each day for prosumption or load/PV seprately
    #df = pd.concat([pd.DataFrame(load_temp.groupby(np.arange(len(load_temp))//48).min().swapaxes(0,1).mean(axis=1), \
    #                            columns=['load_min']),df], axis=1)
    #df = pd.concat([pd.DataFrame(load_temp.groupby(np.arange(len(load_temp))//48).max().swapaxes(0,1).mean(axis=1), \
    #                            columns=['load_max']),df], axis=1)
    #df = pd.concat([pd.DataFrame(pv_temp.groupby(np.arange(len(load_temp))//48).min().swapaxes(0,1).mean(axis=1), \
    #                            columns=['pv_min']),df], axis=1)
    #df = pd.concat([pd.DataFrame(pv_temp.groupby(np.arange(len(load_temp))//48).max().swapaxes(0,1).mean(axis=1), \
    #                            columns=['pv_max']),df], axis=1)
    #df = pd.concat([pd.DataFrame(prosumption.groupby(np.arange(len(prosumption))//48).max().swapaxes(0,1).mean(axis=1), \
    #                            columns=['prosumption_max']),df], axis=1)
    #df = pd.concat([pd.DataFrame(prosumption.groupby(np.arange(len(prosumption))//48).min().swapaxes(0,1).mean(axis=1), \
    #                            columns=['prosumption_min']),df], axis=1)
    ##Feature 2: Mean/Max/Min of each day for prosumption or load/PV seprately
    #df = pd.concat([load_temp.groupby(np.arange(len(load_temp))//48).min().swapaxes(0,1).set_axis(['load_min_'+str(i) for i in np.arange(0,365)], axis=1), df], axis=1)
    #df = pd.concat([pv_temp.groupby(np.arange(len(load_temp))//48).min().swapaxes(0,1).set_axis(['pv_min_'+str(i) for i in np.arange(0,365)], axis=1), df], axis=1)
    #df = pd.concat([load_temp.groupby(np.arange(len(load_temp))//48).max().swapaxes(0,1).set_axis(['load_max_'+str(i) for i in np.arange(0,365)], axis=1), df], axis=1)
    #df = pd.concat([pv_temp.groupby(np.arange(len(load_temp))//48).max().swapaxes(0,1).set_axis(['pv_max_'+str(i) for i in np.arange(0,365)], axis=1), df], axis=1)
    #df = pd.concat([load_temp.groupby(np.arange(len(load_temp))//48).mean().swapaxes(0,1).set_axis(['load_mean_'+str(i) for i in np.arange(0,365)], axis=1), df], axis=1)
    #df = pd.concat([pv_temp.groupby(np.arange(len(load_temp))//48).mean().swapaxes(0,1).set_axis(['pv_mean_'+str(i) for i in np.arange(0,365)], axis=1), df], axis=1)
    #df = pd.concat([prosumption.groupby(np.arange(len(prosumption))//48).min().swapaxes(0,1).set_axis(['prosumption_min_'+str(i) for i in np.arange(0,731)], axis=1), df], axis=1)
    #df = pd.concat([prosumption.groupby(np.arange(len(prosumption))//48).max().swapaxes(0,1).set_axis(['prosumption_max_'+str(i) for i in np.arange(0,731)], axis=1), df], axis=1)
    #df = pd.concat([prosumption.groupby(np.arange(len(prosumption))//48).mean().swapaxes(0,1).set_axis(['prosumption_mean_'+str(i) for i in np.arange(0,731)], axis=1), df], axis=1)
    ##Feature 1: Mean over all days for each hour for prosumption or load/PV seprately
    #df = pd.concat([pd.DataFrame(load_temp.values.reshape((-1, 48, 300)).mean(axis=0).swapaxes(0,1),\
    #                    index=df.index, columns=['load_'+str(i) for i in np.arange(0,48)]),df], axis=1)
    #df = pd.concat([pd.DataFrame(pv_temp.values.reshape((-1, 48, 300)).mean(axis=0).swapaxes(0,1),\
    #                    index=df.index, columns=['pv_'+str(i) for i in np.arange(0,48)]),df], axis=1)


    df = pd.concat([pd.DataFrame(prosumption.values.reshape((-1, 48, 300)).mean(axis=0).swapaxes(0,1), index=df.index),df], axis=1)
    final = pd.concat([final, df])



###Generate input and output data for classification
path = 'data//data_analysis//target_imb2.csv'
results = pd.read_csv(path, index_col=['data','bldg'])
y = results['target']
final = final.set_index(y.index)

##Split in train (building 1 to 199) and test (building 200 to 300) data
y_train = y.loc[(slice(None), list(range(1,200)))]
y_test = y.loc[(slice(None), list(range(200,301)))]
x_train = final.loc[y_train.index]
x_test = final.loc[y_test.index]

##Scale input data
object= StandardScaler()
x_tr = final.loc[y_train.index]
x_te = final.loc[y_test.index]
x_train = object.fit_transform(x_tr)
x_test = object.transform(x_te)

##Perform PCA to reduce dimensionality of input features
pca = PCA(0.70)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)



###Perform classification
classifier = [XGBClassifier(), KNeighborsClassifier(), SVC(), MLPClassifier(max_iter=1500), DecisionTreeClassifier(),
              GaussianNB()]
result = {}
for c, classi in enumerate(classifier):
    scores = []
    costs_mean = []
    runtime = []
    for i in range(5):
        classi.fit(x_train, y_train)
        st = datetime.datetime.now()
        y_hat = classi.predict(x_test)
        et = datetime.datetime.now()
        runtime.append(et - st)
        scores.append(f1_score(y_hat, y_test, average='micro'))
        costs = []
        for i, j in enumerate(y_hat):
            costs.append(results.loc[y_test.index].values[i, j])
        costs_mean.append(np.mean(costs))
    result[str(c)] = {'costs': costs_mean, 'scores': scores, 'runtime': runtime}