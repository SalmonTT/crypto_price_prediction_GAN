# ML in Practice Project Report (IEOR4579)
#### Tianmeng Tie (tt2877), Jinghan Zhang (jz3526), Chloe Zhu (nz2365）

## 1. Introduction
Our aim for this project is twofold: 1) replicate the paper [Stock price prediction using Generative Adversarial Networks](https://paperswithcode.com/paper/stock-price-prediction-using-generative) by H.Lin *et al.* and 2) extend the paper by changing the dataset and improving shortcomings found in paper. 

### 1.1 Paper Summary

Lin et al. used a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) to predict the next 3-day stock prices of Apple Inc. (AAPL) using the past 30 days’ data. Gated Recurrent Units (GRU) was used as the generator and Convolutional Neural Network (CNN) was used as the discriminator.

The performance of the proposed WGAN-GP model with baseline models consisting of Long-short Memory Networks (LSTM), GRU, and GAN model. Using Root Mean Square Error (RMSE) as the evaluation metric, the authors conclude that both GAN and WGAN-GP models outperform LSTM and GRU models. In addition, WGAN-GP model exhibits better performance during shock events such as COVID-19 in comparison to GAN model. 

## 2. Replication
### 2.1 Dataset Description
The dataset used consists of 36 features and 2497 days of observations from 01/07/2010 to 06/30/2020. Three categories of features were chosen: 1) technical indicators generated by AAPL's close price, 2) prices of correlated securities and, 3) prices of market index and commodities (Table 2). A 7:3 ratio was used for splitting training and testing data.

### 2.2 Model Architecture

The Basic GAN and WGAN-GP models share similar model architectures in general but with 3 differences: 1) the **number of layers** in their generators is different; 2) the **number of neurons** in each generator layer is different; 3) **WGAN-GP has a regularizer** for each generator layer whereas Basic GAN doesn't. Specifically, in the WGAN-GP model, a L2 Regularization is embedded into each of the GRU layers as a recurrent regularizer as well as into the dense layers as a kernel regularizer. Figure 1 illustrates the architecture of the Basic GAN model utilized in Lin et al.'s paper and Table 5 demonstrates the architecture of the generator in the WGAN-GP model. Configurations of models used in the original paper are summarized in Table 4.

### 2.3 Replication Results
This section discusses the results of our replication. While the paper adopted RMSE as the evaluation metric (Table 6), we proceed to normalize the RMSE (Table 1) to account for the differences in the scale of the training and test data.
The out-sample price prediction results for all 4 models are shown in Figure 2 and the loss functions for GAN and WGAN-GP are shown in Figure 3.
<p style="text-align: center;">Table 1: Normalized RMSE (%) </p>

| Paper/Replication result | LSTM | GRU | Basic GAN | WGAN-GP |
| -------- | -------- | -------- |-------- |-------- |
| **Train**  | 4.99/**2.20**     | 3.28/**2.17**   |5.38/**12.90**    |5.71/**15.56**     |
| **Test**    | 11.77/**17.37**    | 9.50/**28.37**     |9.56/**16.37**     |8.50/**14.14**    |
| **Test (excl. 2020)**    | 16.85/**8.99**     | 7.27/**16.65**     |5.51/**10.57**     |6.92/**7.86**     |





### 2.4 Shortcomings
Several shortcomings were found in the data processing methodology upon closer inspection with the source codes. We discuss the implications of these shortcomings here. 

During the data imputation process, two questionable methods were used: 1) backward fill and 2) taking the average of the previous and next available data. These methods introduce future data and are generally frowned upon during machine learning application in finance. The number of observations affected by this is non-trivial and therefore would affect the fidelity of the results in our opinions. 

In addition, we felt that the choice of features was also quite arbitrary. Market indices are often noisy signals since it reflects the performances of many sectors. The correlation between commodities and price of AAPL is also unclear both economically and empirically. Consequently, we felt that the choice of features potentially added a lot of noise to the dataset. 

### 2.5 Conclusion
  

Based on our replication results shown in the table above, the WGAN-GP model outperforms the Basic GAN model in predicting equity prices, both during normal times and the COVID-19 period. The WGAN-GP model has lower RMSE values in both test periods, with and without the inclusion of 2020. The inconsistency may result from the difference in the data preprocessing procedure or randomness in model training. Apart from this variation, our replication results are consistent with those presented in the original paper.



## 3. Extension
Our extension to the paper addresses some of the aforementioned shortcomings by swapping the entire dataset. Instead of equity price prediction, we decided to predict the price of Ethereum (ETHUSD) and use other cryptocurrencies and macroeconomic variables as features. Cryptocurrencies are highly correlated assets and exhibit a high correlation to macroeconomic conditions (Sovbetov and Yhlas 2018). In addition, we avoided the aforementioned data imputation methods. 

### 3.1 Dataset Description
The dataset used consists of 43 features from three categories and 1860 observations starting from 02/26/2018 to 03/31/2023 (Table 3). The three categories are: 1) cryptocurrencies' prices, 2) foreign exchange rate, 3) fixed income instruments, 4) Although we are able to obtain intraday tick-level trades data for cryptocurrencies, we can only obtain daily data for our macroeconomic features. Therefore, our final dataset contains only daily data.

We obtained historical prices of cryptocurrencies from [polygon.io](https://polygon.io/) and macroeconomic features from [FRED](https://fred.stlouisfed.org/docs/api/fred/) and [NASDAQ](https://data.nasdaq.com/). 

### 3.2 Experiments
In our experiments, we used the same configurations as stated in the original paper for each of the models. For each model, we further test the results of different data structures. Recall that the original paper uses previous 30 days to predict next 3 days, we further extend our experiments to try 3-to-1 (3 days to 1 day), 15-to-1, 15-to-3, and 30-to-1. For the many-to-many data structures, we take the average of the predicted prices and use that as the predicted price of the first day. We provide a visual representation of this process in Figure 9. 


### 3.3 Results
We used two metrics to evaluate our model performances during the extension process (Table 7), namely the normalized RMSE and price trend accuracy. The latter indicates whether the direction of the predicted price (up/down) is the same as the actual price. As shown in Table 7, Basic GAN outperforms all other models in terms of normalized RMSE whereas GRU has the highest price trend accuracy. However, price trend accuracies for all different configurations of the 4 models are around 50%, indicating that these machine learning models are only able to predict the price trend correctly half of the time. 

Consequently, while GAN and WGAN-GP models demonstrate good capability in generating future prices with low normalized RMSE, there is little economic value as they cannot be used as trading signals due to the poor price trend accuracy. 


### 3.4 Conclusions
In both our replication and extension process, we discovered that GAN and WGAN-GP models outperform GRU and LSTM models significantly. In addition, GAN models are more suitable for performing price predictions of crypto currencies as compared to equities since the normalized RMSEs in our extension are significantly lower than those in our replication. However, the economic value of using GAN models to predict equity/crypto currency prices is questionable as the price trend accuracies for all models are floating around 50%, making it hard to convert prediction into profitable trade signals. 

While stock price prediction has always been an important and intriguing topic, few papers have touched on it and even fewer had reached solid results. Our project could serve as a baseline for further research on this topic, which might include the detection of anomalies in stock/crypto currency prices and predicting returns rather than prices.

## 4. References

Lin, H., Chen, C., Huang, G. & Jafari, A. (2021). Stock price prediction using Generative Adversarial Networks. Journal of Computer Science, 17(3), 188-196. https://doi.org/10.3844/jcssp.2021.188.196

Sovbetov, Yhlas, Factors Influencing Cryptocurrency Prices: Evidence from Bitcoin, Ethereum, Dash, Litcoin, and Monero (February 17, 2018). Journal of Economics and Financial Analysis, 2(2), 1-27, Available at SSRN: https://ssrn.com/abstract=3125347



## 5. Appendix

### 5.1 Features:
<p style="text-align: center;">Table 2. Features used in original paper</p>

| Price Bar Data  | Technical indicators | Correlated Assets | Others             |
| ----------------- | -------------------- | ---------------- | ------------------------ |
| Open    | MA7       | NASDAQ      |     News               |
| High    | MA21      | NYSE         |                    |
| Low     | 20SD      | S&P500        |                   |
| Close   | MACD      |           RUSSELL2000       |                          |
| Volume |  upper_band     |           BSE SENSEX       |                          |
|         |  lower_band            | FTSE100      |                  |
|         |  EMA             | Nikki225    |  |
|         | Logmomentum           |             HENG SENG Index     |  |
|         |  absolute_3_comp          |            SSE      |  |
|             |  angle_3_comp           |          Crude Oil        |                     |
|                   | absolute_6_comp                  |      Gold            |              |
|                   |  angle_6_comp                  |       VIX           |                |
|                   |absolute_9_comp   |        USD Index          |                |
|                   | angle_9_comp        |          Amazon      |            |
|                   |     |          Google        |                    |
|                   |            |        Microsoft          |                    |





<p style="text-align: center;">Table 3. Features used in extension
</p>

| Correlated Assets/VIX | Technical indicators | FX/Bullions | Fixed Income             |
| ----------------- | -------------------- | ---------------- | ------------------------ |
| BTCUSD_vwap       | ma7                  | usd_nominal      | 5_YR                     |
| USDTUSD_vwap       | ma21                 | usd_real         | 7_YR                     |
| XRPUSD_vwap     | 26ema                | usd_euro         | 10_YR                    |
| ADAUSD_vwap       | 12ema                | gold_london      | 20_YR                    |
| DOGEUSD_vwap       | MACD                 | silver_london    | 4_Wk_Bank_Discount_Rate  |
| LTCUSD_vwap      | 20sd                 |                  | 13_Wk_Bank_Discount_Rate |
| VIX       | upper_band           |                  | 52_Wk_Bank_Discount_Rate |
|             | lower_band           |                  | tips                     |
|                   | ema                  |                  | uk_20y_nif               |
|                   | RSI                  |                  | uk_10y_nif               |
|                   | absolute_3_comp      |                  | uk_5y_nif                |
|                   | angle_3_comp         |                  | uk_bank_rate             |
|                   | absolute_6_comp      |                  | EM_HY                    |
|                   | angle_6_comp         |                  | US_Corp                  |
|                   | absolute_9_comp      |                  |                          |
|                   | angle_9_comp         |                  |                          |




### 5.2 Model Configurations:

<p style="text-align: center;">Table 4. Configurations of models in original paper
</p>

| Model         | Learning Rate | Batch Size | Epochs |
| ------------- | ------------- | ---------- | ------ |
| Baseline Gru  | 0.0001        | 128        | 50     |
| Baseline LSTM | 0.001         | 64         | 50     |
| Basic GAN     | 0.00016       | 128        | 165    |
| WGAN-GP       | 0.0001        | 128        | 100    |



### 5.3 Model Architectures:

![](https://i.imgur.com/8EjzAJ4.png)
Figure 1. Basic GAN architecture in original paper


<p style="text-align: center;">Table 5. WGAN-GP architecture in original paper
</p>

| Architecture of Generator in WGAN-GP |
|:-------------------------------------:|
|        GRU Layer, 256 neurons, l2($1e^{-3}$), dropout=0.2         |
|        GRU Layer, 128 neurons, l2($1e^{-3}$), dropout=0.2          |
|       Dense Layer, 64 neurons, l2($1e^{-3}$)        |
|       Dense Layer, 32 neurons, l2($1e^{-3}$)         |



### 5.4 Results - Replication:

![](https://i.imgur.com/wPqh2YZ.png)
Figure 2. Out-sample price predictions from replication

![](https://i.imgur.com/7L6q3kB.png)
Figure 3. Basic GAN and WGAN-GP loss functions from replication

<p style="text-align: center;">Table 6. Unnormalized RMSE
</p>

| Paper/Replication result | LSTM | GRU | Basic GAN | WGAN-GP |
| -------- | -------- | -------- |-------- |-------- |
| **Train**  | 1.52/**0.67**     | 1.00/**0.66**   |1.64/**3.93**    |1.74/**4.74**     |
| **Test**    | 6.60/**9.74**    | 5.33/**15.91**     |5.36/**9.18**     |4.77/**7.93**    |
| **Testing (excl. 2020)**    | 9.45/**5.04**     | 4.08/**9.34**     |3.09/**5.93**     |3.88/**4.41**     |

### 5.5 Results - Extension:

<p style="text-align: center;">Table 7. Performances of models during extension with different configurations 
</p>

![Results from Extension](https://i.imgur.com/KS1CtzL.png)

*\*Normalized RMSE = RMSE / (max value / min value)*
*\*Price Trend Accuracy: Whether the direction of the predicted price (up/down) is the same as actual price*


![](https://i.imgur.com/qyLme6j.png)
Figure 4. GAN, WGAN-GP loss function during extension


<p float="left">
  <img src="https://i.imgur.com/QeNXPVk.png" width="250" />
  <img src="https://i.imgur.com/mPbdwGf.png" width="250" /> 
  <img src="https://i.imgur.com/oO5DMV6.png" width="250" />
  <img src="https://i.imgur.com/hgvWEyf.png" width="250" />
  <img src="https://i.imgur.com/9EZoSkm.png" width="250" />
</p>

Figure 5. GRU model out-sample price predictions during extension


<p float="left">
  <img src="https://i.imgur.com/eNNIQvF.png" width="250" />
  <img src="https://i.imgur.com/8D7ZJm2.png" width="250" /> 
  <img src="https://i.imgur.com/44hocwX.png" width="250" />
  <img src="https://i.imgur.com/dAb9336.png" width="250" />
  <img src="https://i.imgur.com/cofpa0L.png" width="250" />
</p>

Figure 6. LSTM model out-sample price predictions during extension



<p float="left">
  <img src="https://i.imgur.com/a8oFwvr.png" width="250" />
  <img src="https://i.imgur.com/3QngLpN.png" width="250" /> 
  <img src="https://i.imgur.com/gI6Kddy.png" width="250" />
  <img src="https://i.imgur.com/enTXjCy.png" width="250" />
  <img src="https://i.imgur.com/BR5aV7Y.png" width="250" />
</p>

Figure 7. GAN model out-sample price predictions during extension


<p float="left">
  <img src="https://i.imgur.com/zcWrFO4.png" width="250" />
  <img src="https://i.imgur.com/CDFweD2.png" width="250" /> 
  <img src="https://i.imgur.com/EZksAeb.png" width="250" />
  <img src="https://i.imgur.com/rS1xCXz.pngg" width="250" />
  <img src="https://i.imgur.com/Gh92jD5.png" width="250" />
</p>

Figure 8. WGAN-GP model out-sample price predictions during extension


![](https://i.imgur.com/5MFG1ij.jpg)
Figure 9. Data structure for many-to-many prediction





