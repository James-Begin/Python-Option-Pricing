import math
import pandas as pd
import numpy as np
import datetime
from scipy.stats import norm

def blackscholes(price, strike, exp, rfr, vol):
    #first intermediate step
    #
    exp = exp/365
    d1 = ((np.log(price / strike)) + (exp * (rfr + ((vol**2)/2)))) / (vol * (exp**0.5))

    #second step
    d2 = d1 - (vol * (exp)**0.5)
    #[call price, put price]
    return [(price * norm.cdf(d1) - strike * np.exp(-rfr*exp) * norm.cdf(d2)),
            (strike * math.exp(-rfr*exp) * norm.cdf(-d2) - price * norm.cdf(-d1))]

def montecarlo(price, strike, exp, rfr, vol):
    #first simulate prices
    #init price movements over time (1000 simulations can be changed)
    timeperstep = (exp**2) / 365
    numsim = 1000
    simulate = np.zeros((exp, numsim))
    simulate[0] = price

    for time in range(1, exp):
        #random values
        rand = np.random.standard_normal(numsim)
        #fill in prices at each time
        simulate[time] = simulate[time-1] * np.exp((rfr - ((vol**2)/2)) *
                        (timeperstep) + (vol * ((timeperstep)**0.5) * rand))

    return [np.exp(-rfr * exp/365) * (1 / numsim) * np.sum(np.maximum(simulate[-1] - strike, 0)),
            np.exp(-rfr * exp/365) * (1 / numsim) * np.sum(np.maximum(strike - simulate[-1], 0))]

def binomial(price, strike, exp, rfr, vol):
    timeperstep = (exp**2) / 365
    upfactor = np.exp(vol * timeperstep**0.5)
    downfactor = 1/upfactor
    Vcall = np.zeros(exp+1)
    Vput = np.zeros(exp+1)
    prices = np.array([((price * upfactor**i * downfactor**(exp-i)) for i in range(exp+1))])
    compound = np.exp(rfr * timeperstep)
    upprob = (compound-downfactor) / (upfactor - downfactor)
    downprob = 1.0-upprob

    Vcall[:] = np.maximum(prices - strike, 0.0)
    Vput[:] = np.maximum(strike - prices, 0.0)
    for i in range(exp-1, -1, -1):
        Vcall[:-1] = np.exp(-rfr * timeperstep) * (upprob * Vcall[1:] + downprob * Vcall[:-1])
        Vput[:-1] = np.exp(-rfr * timeperstep) * (upprob * Vput[1:] + downprob * Vput[:-1])

    return [Vcall[0], Vput[0]]

price = float(input("Price of Underlying: "))
strike = float(input("Option Strike Price: "))
exp = int(input("Days to expiration: "))
rfr = float(input("Risk Free Rate: "))
vol = float(input("Underlying Volatility (Sigma): "))

print("Black Scholes Call/Put: " + str(round(blackscholes(price, strike, exp, rfr, vol)[0], 2)) + "/" +
      str(round(blackscholes(price, strike, exp, rfr, vol)[1], 2)))
print("Monte Carlo Call/Put: " + str(round(montecarlo(price, strike, exp, rfr, vol)[0], 2)) + "/" +
      str(round(montecarlo(price, strike, exp, rfr, vol)[1], 2)))
print("Binomial Call/Put: " + str(round(binomial(price, strike, exp, rfr, vol)[0], 2)) + "/" +
      str(round(binomial(price, strike, exp, rfr, vol)[1], 2)))

