# Traffic and Sales Analysis for US shop

<p align="center">
  <img src="https://github.com/matteomm/sales-traffic-us-store/blob/master/figures/cover.jpg" width=750>
</p>


Contacts:
* [Linkedin](https://www.linkedin.com/in/matteo-tortella-0a4274130/)
* [e-mail] matteotortella4@gmail.com 


# Table of Contents

1. [ File Descriptions ](#file_description)
2. [ Technologies Used ](#technologies_used)
3. [ Executive Summary ](#executive_summary)
    * [ Data Cleaning and EDA ](#datacleaning)
    * [ Modelling ](#modelling)
    * [ Model Evaluation and Dashboard ](#insights)
 4. [ Limitations and Future Work ](#futurework)


<a name="file_description"></a>
## File Descriptions
> ipynb_checkpoints: different notebooks version going from preprocessing to modelling

> notebooks: contains all the different notebooks used throughout the project from data cleaning to modelling

> data: contains dataset used for the analysis both processed and raw

> figures: jpg images taken from the jupyter notebook 


<a name="technologies_used"></a>
## Technologies used
- Python
- Pandas
- Keras
- Matplotlib

<a name="executive_summary"></a>
## Executive Summary

A US shop is trying to forecast future traffic and sales based on historical data gathered over the years. The initial data provided is split into two different datasets, the traffic one contains data from roughly 2015 to 2018 while the sales csv file presents more data points starting from 2013 and also finishing in 2018. 

Successfully predicting future income and affluence can be extremely important for companies in order to come up with effective strategies encompassing internal logistics and marketing initiatives.

Given the nature of the two datasets, it seemed like **time series analysis** was the most appropriate choice for this type of problem.

An external dataset containing federal US holidays was also merged with the two initial ones in order to provide some additional insights. The external dataset can be found [here](https://www.kaggle.com/gsnehaa21/federal-holidays-usa-19662020) on Kaggle. It simply states all the federal holidays from 1966 to 2020. This dataset was combined with the other two only for EDA purposes but not for modelling ones. That is because modelling already proved quite challenging even without the presence of exogenous variables.

Data Cleaning and EDA was performed simultaneously on both datasets in the data_cleaning notebook. Modelling was done separately for traffic and sales, the former mainly using SARIMA iterations and TBATS modelling while the latter with a Recursive Neural Network approach (LSTM).

Throughout the modelling, the main evaluation metric will be **Mean Squared Error** (MSE) which measures the average of the error squares between the predictions made by the model and the actual observations. 

Our best model for traffic (TBATS) had a final MSE of 43.95 while our the best iteration of LSTM returned a final 32.13 MSE.
Both figures refer to our test set results.


<a name="datacleaning"></a>
## Data Cleaning and EDA

During the first stage we work on both datasets to understand the overall structure of the data while also pre-processing it for time series analysis. For both datasets we only have two columns, the dates and values respectively for traffic of people and sales.

There are no missing values in the sense of 'Nan' values (so not recorded data) but only missing values in terms of 0s which correctly represent the lack of people and sales at specific moment in times (for example during closing times at nights).

However, dates are still reported with a 15 min granularity and since the final goal is to predict traffic/sales per hour, resampling the entire sets by hour instead is an effective way to reduce intra-hours 'missing' values during the day. '0' values for closing times are left as they are as we ideally want the model to pick up on that daily seasonality.

The second step is the merging and analysis of the federal US holidays with both datasets.

The snapshot below tells us quite a bit of information on the average sale and amount of people at the shop broken down by hour, day of the week and months. **Unsurprisingly, peak days and times are during the weekend around lunchtime. However, the increase in traffic does not translate into a significant increase in sales as you can see from the graph on the top right.** 

This is an insight which would need further investigation and also led me to the conclusion that it is probably more appropriate to do modelling of the two variables separately instead of having traffic as an independent variable for sales.

Also, there is a clear yearly/monthly seasonality shown the both graphs in the bottom:

<p align="center">
  <img src="https://github.com/matteomm/sales-traffic-us-store/blob/master/figures/eda_1.png" width=750>
</p>


Additionally, find below same breakdown (pivot of hours and days of the week) for all federal US holidays plus Saturdays and Sundays. Generally speaking, the shop is open every day of the week from 10 to 21 (only Sundays 19 instead) and also during all federal US holidays but Thanksgiving. **Also, it seems to be a very bad idea to be opened on Christmas Day as sales are  consistently lower than any other holiday with an average of 7 dollars an hour**.

<p align="center">
  <img src="https://github.com/matteomm/sales-traffic-us-store/blob/master/figures/eda_traffic.png" width=750>
</p>

<p align="center">
  <img src="https://github.com/matteomm/sales-traffic-us-store/blob/master/figures/eda_sales.png" width=750>
</p>

In the last section, I wanted to perform series decomposition which broke down the original series into three sub-sets showcasing overall trend, seasonality and what is left of the time series (residuals) after stripping out the first two components. You'll find in the notebook that there's no significant trend for both datasets and seasonality is not properly recognised by the decomposition. In the end, I also performed a Dickey-Fuller test to check for stationarity before proceeding to the modelling. Stationarity is a required assuption for most of the ARMA family of models. The Dickey-Fuller test rejects the H0 for both datasets and declares both as stationary. However, multiple seasonality is still there for both series. 

## Modelling

As mentioned before, my intent was to try out two different approaches and see which one yielded better results.

- Traffic SARIMA
- Sales LSTM

### Traffic SARIMA

So far, we have identified two main problems with this task. First the fact that it presents at the same time yearly, monthly, weekly and daily seasonality. Secondly, it shows intermittent demand which means that there are many 0 values for when the shop is closed.

Given this premise, I thought that the most appropriate model in this case would have been a SARIMA which stands for Seasonal AutoRegressive Integrated Moving Averages. The (S)ARIMA needs usually three(four) values to operate on p,d,q(m).

p is the auto-regressive part of the model. It allows us to incorporate the effect of past values into our model. Intuitively, this would be similar to stating that it is likely to rain tomorrow if it has been raining for past 3 days. AR terms are just lags of dependent variable. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5).

D is just the number of the differences and it will be set to 1 because there is still seasonality to de-trend.

q is the moving average part of the model which is used to set the error of the model as a linear combination of the error values observed at previous time points in the past. MA terms form lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.

m is the seasonality parameter which the model takes into account.

In the first section I analysed Autocorrelation and Partial Autocorrelation plot and it seemed like there was a strong relationship with values lagged 24 hours and weekly.

I would have liked to play around with the the p and d parameters but the memory of my laptop constrained me to use values in between 0 and 1 for both. I had to run a dummy model with p=1 d=1 and q=1  with daily seasonality m=24 which returned an initial 127 for MSE. It picked daily patterns perfectly as expected but any variations of traffic during the week was completely ignored.


In the second iteration I basically replicated the same SARIMA model but changing the model seasoanlity to weekly m=53. I tried to reduce the number of datapoints just by looking at the last 6 months but still computation for anything above p and q more than 2 would take too long. MSE actually got worse in thsi case and the model stopped picking up any valuable pattern.


Unfortunately, any iteration of SARIMA would be unable to deal with multiple seasonalities at once and I had to resort to a different statistical package which is specifically designed to take multiple seasonalities into the modelling at once (TBATS). 

Each seasonality is modeled by a trigonometric representation based on Fourier series.
One of the limitations of this model is that it does not take any exogenous variables like the SARIMAX counterpart would do. Having said that, the SARIMA model was already struggling quite a bit in this context and adding exogenous variables would have not helped by a lot in this case.

It is true that we could have used exogenous variables to determine opening hours for the shop. Still the multiple seasonality problem would have persisted even in that scenario, probably leading to poor performance. 

The final TBATS model however still had negative values for hours overnight and I decided to round negative values to 0. The final winning model has a 43 MSE on test set and I produced forecasts for the next month outside the dataset.

Please find below predictions on the test set while forecast of additional month June 2018:





