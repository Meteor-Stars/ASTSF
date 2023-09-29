# AS_TSF

# Update
  .
# Environment
 Pytorch 1.12.1

 Numpy 1.22.4

## Downloading of Datasets
  You can download all datasets from https://drive.google.com/drive/folders/1I819ARskzUCvDS76f8OqLuVaVFqsZ0LK?usp=sharing.

  The downloaded folders 'Fund_all' and 'Fund_all_2' should be placed at the 'dataset' folder. Then you can following the annotation in 'run_baselines_fund.py' and run it to reproduce the paper results. 'VS' denotes Vanilla scale scaling. 'SC' denotes only using our scale calibrating sub-module. 'SS+SC' is equivalent to out AS module, using both our scale calibrating and scaling selection sub-modules.
  
## Introduction of of Datasets
   We collect fund sales datasets of different customers from Ant Fortune, which is an online wealth management platform on the Alipay APP."transaction_date" represents the transaction date of the current fund product. The time span of each dataset extends from the past to November 2022. The maximum observation point is two years.  **The sales of different fund products exhibit
    scale heterogeneity and are suitable to validate our method.** For example, in the downloaded fund datasets, the sales volumes (apply_amt, applying transaction amount; redeem_amt, redemption transaction amount) for product IDs 50, 48, 6, and 52 vary significantly in magnitude, ranging from hundreds, tens, units, to decimal places, indicating scale heterogeneity. This presents challenges for model training convergence.
   
   The characteristics of each fund product are as follows. "product_pid" indicates the fund ID of the current fund product. "is_summarydate" indicates whether the transaction date of the current fund product is a summary date (fund products do not trade on holidays and weekends, and the trading volume during these periods is aggregated to the next non-holiday or non-weekend, known as a summary date). "apply_amt" represents the applying transaction amount of the current fund product. "redeem_amt" represents the redemption transaction amount of the current fund product. "during_days" indicates the holding period of the current fund product (the number of days to hold the fund product before it can be traded). "is_trade" indicates whether the current day is a trading day. "is_weekend_delay" indicates whether it is a weekend before a trading day. "holiday_num" indicates how many statutory holidays occur before the trading day.