# Stock Price Prediction Using AI


# 1. Introduction: <a class="anchor" id="introduction"></a>

We attempt to reproduce the methods and results shown in this [Github Repo](https://github.com/borisbanushev/stockpredictionai) by Boris Banushev. If time allows, we will extend upon his work in serveral ways:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Multiple stocks prediction  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Additional features  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Modification to model architecture  

The author (Boris Banushev) attempts to predict the stock price movement of **Goldman Sachs (NYSE: GS)**. 
- Data sample: 01/01/2010 to 31/12/2018 (7 years training and 2 years for validation)

For our project


# 2. Data: <a class="anchor" id="thedata"></a>

### Below are data sources (and transformation methods) used by the author:
#### 1. Stock's historical data:
   - 1585 days to train the model, and predict the next 680 days
   - start_date = '2010-01-01', end_date = '2018-12-31'
#### 2. Correlated assets:
   - Related companies: these include JPMorgan, Morgan Stanley etc.
       - Downloaded the Adj Closing price of all firms in the S&P500 during the training sample period and checked the log return correlation against underlying stock (GS)
       - Obtain the top 10 most correlated stocks and use as correlated assets
   - Global macro indicators:
       - LIBOR (USD and GBP) etc.
       - Complete list: fed funds rate (DFF), TIPS (DFII10), non-farm payroll (PAYEMS), inflation (USA, GBR, EUR, JPN), CPI (USA, GBR, EUR, JPN), US fixed income...) need to be summarized in a table. 
   - VIX and other daily volatility index
   - Composite indices:
       - NASDAQ, NYSE, FTSE100, Nikkei225, Hang Seng, BSE Sensex indices
   - Currencies: 
       - Basket of currencies including USDJPY, GDPUSD etc.
#### 3. Technical Indicators:

#### 4. Fundamental analysis:
- Financial ratios (PE ratios etc.)
- Sentiment analysis: 
    - daily news of the underlying stock (GS), transformed into sentiment indicators using BERT
#### 5. Fourier transforms:
- Transform the closing price of the underlying stock with Fourier transforms.
- Purpose is to generalize several long- and short-term trends.
#### 6. Autoregressive Integrated Moving Average (ARIMA):
- Used on closing price of underlying stock also
#### 7. Deep unsupervised learning for anomaly detecton in options pricing
- For everyday, add the price for 90-days call option on the underlying stock
- Spot anomalies in pricing using deep unsupervised learning




# 3. Data Preparation: <a class="anchor" id="dataprep"></a>



# Model: <a class="anchor" id="model"></a>
The project uses **Generate Adversarial Network (GAN)** combined with **LSTM** as generator and **Convolutional Neural Network (CNN)** as discriminator. 
- Why are these models chosen?

For GAN, the tricky part is getting the right set of hyperparameters.
- **Bayesian optimization** and **RL** are used to decide when and how to change GAN's hyperparameters (*exploration vs. exploitation dilemma*).
    - What are these methods and what is the dilemma?
