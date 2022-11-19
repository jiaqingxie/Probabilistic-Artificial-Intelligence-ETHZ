# ETH Zurich PAI HS 2022
Chong Zhang @Tsinghua, Ziheng Chi @HPoly, Jiaqing Xie @Edinburgh. We are all master student at ETH Zurich. This repo includes four projects.

## 1. Gaussian Process Regression for Air Pollution Prediction
<img></img>
We have trained several local GPs with clusters to include all of the training points. As part of training local GPs, we used KMeans to fit the data to each cluster (each cluster trains an individual GP model). It is estimated that the final cost would be approximately 15.18.
<br/><br/>

## 2. Bayesian NNs and measurements of predictive uncertainty
As part of this project, we implemented all four techniques: Monte Carlo with dropout, deep ensemble models, stochastic gradient Langevin dynamics (SGLD), and Bayesian neural networks (BNNs). 
<br/><br/>

## 3. Hyperparameter tuning with Bayesian optimization
We implemented 1) the next recommendation.  2) the acquisition function. (EI) 3) add_data_point. We concatenated the given input x, v, f & trained two GP. 4) get_solution. We would prefer the point that satisfies both conditions: trying to be the maximum of f and standing above v_min.
<br/><br/>

## 4. To be announced

