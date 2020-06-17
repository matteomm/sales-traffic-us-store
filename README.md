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
## Data Cleaning

During the first stage we work on both datasets to understand the overall structure of the data while also pre-processing it for time series analysis. For both datasets we only have two columns, the dates and values respectively for traffic of people and sales.

There are no missing values in the sense of 'Nan' values (so not recorded data) but only missing values in terms of 0s which correctly represent the lack of people and sales at specific moment in times (for example during closing times at nights).

However, dates are still reported with a 15 min granularity and since the final goal is to predict traffic/sales per hour, resampling the entire sets by hour instead is an effective way to reduce intra-hours 'missing' values during the day. '0' values for closing times are left as they are as we ideally want the model to pick up on that daily seasonality.

The second step is the merging and analysis of the federal US holidays with both datasets.

The snapshot below tells us quite a bit of information on the average sale and amount of people at the shop broken down by hour, day of the week and months. Unsurprisingly, peak days and times are during the weekend around lunchtime. However, the increase in traffic does not translate into a significant increase in sales as you can see from the graph on the top right. 

This is an insight which would need further investigation and also led me to the conclusion that it is probably more appropriate to do modelling of the two variables separately instead of having traffic as an independent variable for sales.

Also, there is a clear yearly/monthly seasonality shown the both graphs in the bottom:
