# Python-Option-Pricing
Finding an accurate price for a dynamic, constantly changing object like an option is nearly impossible (unless you can see into the future). Instead, there are numerous methods to estimate the price of an option based on some simple metrics.  

One of the most well known metthods of pricing options is the Black-Scholes-Merton Model (BSM). First coined in the early 70's, it helped make it easier to price options and a boom in options trading around the world. Another method of pricing opions is using Monte Carlo Simulations, where the future prices are predicted using probability distributions. The third method used in this project is the binomial lattice method, where a tree of potential future prices is used to determine the value of an option.

## Black-Scholes-Merton Model
BSM is the simplest of the three models and simply uses the Black-Scholes equation.  
One downside to BSM is that it makes assumptions about the underlying asset and broader market which are not always true.  

One primary assumption of BSM is that stock prices follow geometric brownian motion, or more simply, that the returns of a stock follow a random walk-like pattern. This is not always the case due to unpredictable volalitility or behavioural factors. Some smaller assumptions made in BSM are how the risk free rate is constant throughout the life of the option and that the underlying stock does not pay dividends. The Black-Scholes formula for a basic european call option is:  
https://github.com/user-attachments/assets/ad69e24e-e797-4404-ab3c-afca3da2c811

