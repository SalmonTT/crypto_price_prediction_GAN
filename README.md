# ML in Practice Project Report (IEOR4579)

## Abstract
Our aim for this project is twofold: 1) we aim to first replicate the project detailed in the paper 
[Stock price prediction using Generative Adversarial Networks](https://paperswithcode.com/paper/stock-price-prediction-using-generative) by H.Lin *et al.* and 2) we attempt to improve some of the potential shortcomings found in the paper. 

## Paper Summary
The paper conducts Generative Adversarial Network(GAN) to predict stock price. 
## Shortcomings

### Data shortcomings

### Model Architecture
 1. replicate the paper's model
 2. pseudo gridsearch


## Extensions

### Datasets and Features

#### Dataset Descriptions
Data are downloaded from three main data sources: 1) [Polygon.io](https://polygon.io/), 2) [FRED](https://fred.stlouisfed.org/) and 3) [Nasdaq Data Link](https://data.nasdaq.com/). We downloaded the tick-level trades data for the top 5 cryto-currencies (ranled by market cap) and converted them into bar data. Two kinds of bar data are used: time bar (1 hour frequency) and dollar-value bar ($1MM sampling frequency). The target price for the model is Bitcoin (BTCUSD) closing price. 
