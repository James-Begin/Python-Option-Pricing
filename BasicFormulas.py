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
    years = exp/365
    timeperstep = years / exp
    numsim = 1000
    simulate = np.zeros((exp, numsim))
    simulate[0] = price

    for time in range(1, exp):
        #random values
        rand = np.random.standard_normal(numsim)
        #fill in prices at each time
        simulate[time] = simulate[time-1] * np.exp((rfr - 0.5 * vol**2) *timeperstep + (vol * np.sqrt(timeperstep) * rand))

    return [np.exp(-rfr * years) * (1 / numsim) * np.sum(np.maximum(simulate[-1] - strike, 0)),
            np.exp(-rfr * years) * (1 / numsim) * np.sum(np.maximum(strike - simulate[-1], 0))]

def binomialput(price, strike, exp, rfr, vol):
    steps = 1000
    exp = exp/365
    dt = exp / steps
    upfactor = np.exp(vol * np.sqrt(dt))
    downfactor = 1 / upfactor
    upprob = (np.exp(rfr * dt) - downfactor) / (upfactor - downfactor)
    tree = {}

    for m in range(steps + 1):
        currnode = price * (upfactor ** (2 * m - steps))
        tree[(steps, m)] = max(strike - currnode, 0)

    for k in range(steps - 1, -1, -1):
        for m in range(k + 1):
            val = np.exp(-rfr * dt) * (upprob * tree[(k + 1, m + 1)] +
                                                         (1 - upprob) * tree[(k + 1, m)])
            node = price * (upfactor ** (2 * m - k))
            tree[(k, m)] = max(val, max((strike - node), 0))

    return tree[0, 0]

def binomialcall(price, strike, exp, rfr, vol):
    steps = 1000
    exp = exp/365
    dt = exp / steps
    upfactor = np.exp(vol * np.sqrt(dt))
    downfactor = 1 / upfactor
    upprob = (np.exp(rfr * dt) - downfactor) / (upfactor - downfactor)
    tree = np.zeros(steps+1)
    tree[0] = price * downfactor ** steps
    for i in range(1, steps + 1):
        tree[i] = tree[i-1] * (upfactor / downfactor)

    optiontree = np.zeros(steps+1)
    for i in range(0, steps + 1):
        optiontree[i] = max(0, tree[i] - strike)

    for i in range(steps, 0, -1):
        for j in range(0, i):
            optiontree[j] = ((np.exp(-rfr * dt)) *
                             (upprob * optiontree[j+1] + (1-upprob) * optiontree[j]))

    return optiontree[0]

price = float(input("Price of Underlying: "))
strike = float(input("Option Strike Price: "))
exp = int(input("Days to expiration: "))
rfr = float(input("Risk Free Rate: "))
vol = float(input("Underlying Volatility (Sigma): "))

bscall = round(blackscholes(price, strike, exp, rfr, vol)[0], 2)
bsput = round(blackscholes(price, strike, exp, rfr, vol)[1], 2)
mccall = round(montecarlo(price, strike, exp, rfr, vol)[0], 2)
mcput = round(montecarlo(price, strike, exp, rfr, vol)[1], 2)
bccall = round(binomialcall(price, strike, exp, rfr, vol), 2)
bcput = round(binomialput(price, strike, exp, rfr, vol), 2)

print("Black Scholes Call/Put: " + str(bscall) + "/" + str(bsput))
print("Monte Carlo Call/Put: " + str(mccall) + "/" + str(mcput))
print("Binomial Call/Put: " + str(bccall) + "/" + str(bcput))

print("Average Call/Put: " + str(round((bscall + mccall + bccall) / 3, 2)) + "/" + str(round((bsput + mcput + bcput) / 3, 2)))


