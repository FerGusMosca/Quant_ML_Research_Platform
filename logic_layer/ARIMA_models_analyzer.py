import time
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
class ARIMAModelsAnalyzer():
    def __init__(self, p_logger):
        self.logger = p_logger



    #region privste methods

    def __aggregate_weekly_basis__(self,df,symbol,period):
        df.set_index('date', inplace=True)
        df_period = df.resample(period).mean()
        df_period = df_period[[symbol]]
        df_period.head()
        return df_period

    def __add_weekly_returns__(self,df_period,symbol):
        df_period['period_ret'] = np.log(df_period[symbol]).diff()
        df_period.head()

        # drop null rows
        df_period.dropna(inplace=True)

        #En caso que lo queramos --> weekly plot
        #df_week.weekly_ret.plot(kind='line', figsize=(12, 6))

        udiff = df_period.drop([symbol], axis=1)
        udiff.head()

        return udiff


    def __plot_rolling_mean_std_dev__(self,udiff):
        rolmean = udiff.rolling(20).mean()
        rolstd = udiff.rolling(20).std()

        plt.figure(figsize=(12, 6))
        orig = plt.plot(udiff, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std Deviation')
        plt.title('Rolling Mean & Standard Deviation')
        plt.legend(loc='best')
        plt.show(block=False)


    #The ACF gives us a measure of how much each "y" value is correlated to the previous n "y" values prior.
    #Helps us choose the q parameter
    def __acf_auto_corr_plot__(self,udiff):
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_acf(udiff.values, lags=10, ax=ax)
        plt.show(block=False)


    #he PACF is the partial correlation function gives us (a sample of) the amount of correlation between two "y" values
    # separated by n lags excluding the impact of all the "y" values in between them.
    # Helps us chose the p parameter
    def __pacf_auto_corr_plot__(self,udiff):
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_pacf(udiff.values, lags=10, ax=ax)
        plt.show(block=False)



    def __perf_dickey_fuller_test__(self,udiff):
        # Perform Dickey-Fuller test
        dftest = sm.tsa.adfuller(udiff.period_ret, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

        dickey_fuller_test_dict={}
        for key, value in dftest[4].items():
            dickey_fuller_test_dict[key]=value


        return  dickey_fuller_test_dict


    def __build_ARIMA__(self,udiff,p,d,q):
        # Notice that you have to use udiff - the differenced data rather than the original data.
        ar1 = ARIMA(udiff.values, order=(p,d,q)).fit()
        ar1.summary()
        return ar1



    def __run_forecast__(self,udiff,ar1,steps):

        forecast = ar1.forecast(steps=steps)
        preds = ar1.fittedvalues
        # plt.figure(figsize=(12, 8))
        # plt.plot(udiff.values, color='blue')
        # plt.plot(preds, color='red')
        #
        # plt.plot(pd.DataFrame(np.array([preds[-1], forecast[0]]).T,
        #                       index=range(len(udiff.values) + 1, len(udiff.values) + 3)), color='green')
        # plt.plot(pd.DataFrame(forecast, index=range(len(udiff.values) + 1, len(udiff.values) + 1 + steps)),
        #          color='green')
        # plt.title('Display the predictions with the ARIMA model')
        # plt.show()
        return forecast

    #endregion


    def build_ARIMA_model(self,series_df,symbol,period,show_graphs=True):
        df_period=self.__aggregate_weekly_basis__(series_df,symbol,period)
        udiff=self.__add_weekly_returns__(df_period,symbol)

        if show_graphs:
            self.__plot_rolling_mean_std_dev__(udiff)
            self.__acf_auto_corr_plot__(udiff)
            self.__pacf_auto_corr_plot__(udiff)

        #Hay que poner un breakpoint para ver bien todos los gráficos!
        #input("Presiona Enter para continuar...")
        pass
        dickey_fuller_test_dict=self.__perf_dickey_fuller_test__(udiff)

        return dickey_fuller_test_dict


    def build_and__predict_ARIMA_model(self,series_df,symbol,p,d,q,period,step):
        df_period=self.__aggregate_weekly_basis__(series_df,symbol,period)
        udiff=self.__add_weekly_returns__(df_period,symbol)
        ar1=self.__build_ARIMA__(udiff,p,d,q)

        preds=self.__run_forecast__(udiff,ar1,step)

        forecast_list = preds.tolist()

        return forecast_list