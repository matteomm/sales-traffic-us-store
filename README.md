# Traffic and Sales Analysis for US shop

<p align="center">
  <img src="https://github.com/matteomm/sales-traffic-us-store/blob/master/figures/cover.jpg" width=750>
</p>


Contacts:
* [e-mail](matteotortella4@gmail.com)
* [Linkedin](https://www.linkedin.com/in/matteo-tortella-0a4274130/)


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

> references: links to the source material referenced in the notebook

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

Data Cleaning and EDA was performed simultaneously on both datasets in the first notebook. Modelling was done separately for traffic and sales, the former mainly using SARIMA iterations and TBATS modelling while the latter with a Recursive Neural Network approach (LSTM).

Throughout the modelling, the main evaluation metric will be Mean Squared Error which 


<a name="datacleaning"></a>
## Data Cleaning and Feature Engineering

As mentioned above the initial dataset was designed through the combination of two distinct sets from the web. The raw initial dataset was designed to have no class imbalance as we have selected exactly 21421 positive and 20610 negative tweets. Given the balanced nature of the training set, this project will mainly look at accuracy and f1 score as success metrics. 

Cleaning was performed with a some iterations of regex s
