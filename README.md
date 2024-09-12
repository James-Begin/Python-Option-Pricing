# Python-Option-Pricing
Finding an accurate price for a dynamic, constantly changing object like an option is nearly impossible (unless you can see into the future). Instead, there are numerous methods to estimate the price of an option based on some simple metrics.  

One of the most well known metthods of pricing options is the Black-Scholes-Merton Model (BSM). First coined in the early 70's, it helped make it easier to price options and a boom in options trading around the world. Another method of pricing opions is using Monte Carlo Simulations, where the future prices are predicted using probability distributions. The third method used in this project is the binomial lattice method, where a tree of potential future prices is used to determine the value of an option.

## Black-Scholes-Merton Model
BSM is the simplest of the three models and simply uses the Black-Scholes equation.  
One downside to BSM is that it makes assumptions about the underlying asset and broader market which are not always true.  

One primary assumption of BSM is that stock prices follow geometric brownian motion, or more simply, that the returns of a stock follow a random walk-like pattern. This is not always the case due to unpredictable volalitility or behavioural factors. Some smaller assumptions made in BSM are how the risk free rate is constant throughout the life of the option and that the underlying stock does not pay dividends. The Black-Scholes formula for a basic european call option is:  
### **$C = S_0 * N(d1) - K * e^{-rT} * N(d2)$**  

In this equation, 
- $C$ represents the price of the option
- $K$ represents the strike price
- $S_0$ represents the price of the stock
- $T$ represents the option's time to expiry in years
- $r$ represents the annualized risk free interest rate
- $e^(-rT)$ represents the discount factor (current value of future amount)
- $N(d1)$ and $N(d2)$ together are the special sauce of the equation and represent probabilities from the CDF of a normal distribution at $d1$ and $d2$ (see below)  
### $d_1 = \frac{ln(\frac{S_0}{K}) + (r + \frac{\sigma^2}{2}) * T}{\sigma * \sqrt{T}}$
### $d_2 = d_1 - \sigma * \sqrt{T}$  

Here, $\sigma$ represents the volatility of the underlying asset.  
Although, $d1$ is complex to derive and difficult to understand from its equation form, it can be thought of as a probability for how likely the option is to end up in "in the money" based on the underlying price, the risk free rate, and the underlying volatility.  

Further, $d2$ helps to adjust $d1$ for the time until expiry, it represents the probability that the option will expire in the money based on the volatility of the underlying and the time until expiry.  

Below is the code for BSM in python, it is important to note that scipy is used to generate the cumulative distribution functions (CDF) and has the fastest runtime of the three models.
```
d1 = ((np.log(price / strike)) + (exp * (rfr + ((vol**2)/2)))) / (vol * (exp**0.5))
d2 = d1 - (vol * (exp)**0.5) 
return [(price * norm.cdf(d1) - strike * np.exp(-rfr*exp) * norm.cdf(d2)),
        (strike * math.exp(-rfr*exp) * norm.cdf(-d2) - price * norm.cdf(-d1))]
```

## Monte Carlo Simulation
Using Monte Carlo Simulation to price options is typically more accurate than BSM as we are simulating the future prices of the underlying asset over many iterations to help determine the price of the option.  

The first step in simulation is to model the movements of the underlying asset. In this case, we assume that stock prices follow Geometric Brownian Motion, similar to the above BSM. Using this assumption, the formula for the price of an underlying stock at time $t$ is:  
$S_t = S_0 * exp[(r - \frac{\sigma^2}{2}) * t + \sigma * W_t]$
