Install the yfinance package, which is the downloader connected with the Yahoo Fiannce API.


```python
pip install yfinance
```

    Requirement already satisfied: yfinance in ./opt/anaconda3/lib/python3.9/site-packages (0.1.87)
    Requirement already satisfied: numpy>=1.15 in ./opt/anaconda3/lib/python3.9/site-packages (from yfinance) (1.21.5)
    Requirement already satisfied: lxml>=4.5.1 in ./opt/anaconda3/lib/python3.9/site-packages (from yfinance) (4.9.1)
    Requirement already satisfied: multitasking>=0.0.7 in ./opt/anaconda3/lib/python3.9/site-packages (from yfinance) (0.0.11)
    Requirement already satisfied: pandas>=0.24.0 in ./opt/anaconda3/lib/python3.9/site-packages (from yfinance) (1.4.4)
    Requirement already satisfied: requests>=2.26 in ./opt/anaconda3/lib/python3.9/site-packages (from yfinance) (2.28.1)
    Requirement already satisfied: appdirs>=1.4.4 in ./opt/anaconda3/lib/python3.9/site-packages (from yfinance) (1.4.4)
    Requirement already satisfied: python-dateutil>=2.8.1 in ./opt/anaconda3/lib/python3.9/site-packages (from pandas>=0.24.0->yfinance) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in ./opt/anaconda3/lib/python3.9/site-packages (from pandas>=0.24.0->yfinance) (2022.1)
    Requirement already satisfied: charset-normalizer<3,>=2 in ./opt/anaconda3/lib/python3.9/site-packages (from requests>=2.26->yfinance) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in ./opt/anaconda3/lib/python3.9/site-packages (from requests>=2.26->yfinance) (3.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./opt/anaconda3/lib/python3.9/site-packages (from requests>=2.26->yfinance) (1.26.11)
    Requirement already satisfied: certifi>=2017.4.17 in ./opt/anaconda3/lib/python3.9/site-packages (from requests>=2.26->yfinance) (2022.9.24)
    Requirement already satisfied: six>=1.5 in ./opt/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.8.1->pandas>=0.24.0->yfinance) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.



```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mpl
```

##   1a. Major crypto currency chosen:Bitcoin-usd

   1b. Download btc-usd daily historical two years data and save it as btc-usd pandas data frame


```python
initial_data = yf.download("BTC-USD", start="2020-12-04", end="2022-12-04")
```

    [*********************100%***********************]  1 of 1 completed



```python
initial_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-12-04</th>
      <td>19446.966797</td>
      <td>19511.404297</td>
      <td>18697.193359</td>
      <td>18699.765625</td>
      <td>18699.765625</td>
      <td>33872388058</td>
    </tr>
    <tr>
      <th>2020-12-05</th>
      <td>18698.384766</td>
      <td>19160.449219</td>
      <td>18590.193359</td>
      <td>19154.230469</td>
      <td>19154.230469</td>
      <td>27242455064</td>
    </tr>
    <tr>
      <th>2020-12-06</th>
      <td>19154.179688</td>
      <td>19390.500000</td>
      <td>18897.894531</td>
      <td>19345.121094</td>
      <td>19345.121094</td>
      <td>25293775714</td>
    </tr>
    <tr>
      <th>2020-12-07</th>
      <td>19343.128906</td>
      <td>19411.828125</td>
      <td>18931.142578</td>
      <td>19191.630859</td>
      <td>19191.630859</td>
      <td>26896357742</td>
    </tr>
    <tr>
      <th>2020-12-08</th>
      <td>19191.529297</td>
      <td>19283.478516</td>
      <td>18269.945312</td>
      <td>18321.144531</td>
      <td>18321.144531</td>
      <td>31692288756</td>
    </tr>
  </tbody>
</table>
</div>




```python
initial_data['Adj Close'].head()
```




    Date
    2020-12-04    18699.765625
    2020-12-05    19154.230469
    2020-12-06    19345.121094
    2020-12-07    19191.630859
    2020-12-08    18321.144531
    Name: Adj Close, dtype: float64



adjusted close price


```python
data =  initial_data['Adj Close']
data.info()
```

    <class 'pandas.core.series.Series'>
    DatetimeIndex: 730 entries, 2020-12-04 to 2022-12-03
    Series name: Adj Close
    Non-Null Count  Dtype  
    --------------  -----  
    730 non-null    float64
    dtypes: float64(1)
    memory usage: 11.4 KB


plotting the movement/shape of BTC-USD from November 2020 to November 2022 and subsequently calculating its 
annualised volatility

fig 1.0


```python
data.plot(figsize=(8, 6), subplots=True)
```




    array([<AxesSubplot:xlabel='Date'>], dtype=object)




    
![png](output_12_1.png)
    


from the above, BTC-USD had its peak volatility in November/December of 2021 and fell sharply in November/December of 2022;theres bound to be an increse in volatility after this period

calculating the annualized return


```python
data = yf.download("BTC-USD", start="2022-06-04", end="2022-09-04")
```

    [*********************100%***********************]  1 of 1 completed



```python
log_return = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
```


```python
vol = np.sqrt(365) * log_return.std()
print('The annualised volatility is', round(vol*100,2), '%')
```

    The annualised volatility is 70.21 %


designing the derivative

a. Design a derivative and its associated pricing value using at least two methods which must be different.

Designing the vanilla derivative using the first model:
# Black Scholes method


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
import yfinance as yf
```


```python
def euro_option_bs(S, K, T, r, vol, payoff):
    
    
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        option_value = - S * si.norm.cdf(-d1, 0.0, 1.0) + K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return option_value
```

where S= spot price, k=strike price, T= time to maturity, r=interest rate, vol= volatility


```python
s = 18699
K = 19154
T = 0.25   # 3 months june - sep
r = 0.04
q = 0
v = 0.7
```


```python
euro_option_bs(s, K, T, r, vol, 'call')
```




    2758.259424607415




```python
Designing the vanilla derivative using the first model:
# monte Carlo Simulation
```


```python
def mcs_simulation_np(p):
    M = p
    I = p
    dt = T / M 
    S = np.zeros((M + 1, I))
    S[0] = S0 
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1): 
        S[t] = S[t-1] * np.exp((r - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * rn[t]) 
    return S
```


```python
T = 0.25
r = 0.04
sigma = 0.7
S0 =18699
K = 19154
```


```python
S = mcs_simulation_np(100)
```


```python
S = np.transpose(S)
S

```




    array([[18699.        , 18276.14241655, 19172.02638447, ...,
            28680.73710063, 29336.96954849, 28225.9368332 ],
           [18699.        , 19058.88550151, 20072.24874851, ...,
            35532.84426583, 34504.28131114, 35626.50328507],
           [18699.        , 19950.99357142, 19540.91154997, ...,
            23378.48641344, 22440.73234563, 22860.38518268],
           ...,
           [18699.        , 19122.89115157, 17870.49149624, ...,
            22731.04300266, 23542.43064908, 24156.69123001],
           [18699.        , 18584.98115954, 17947.37720241, ...,
            33209.07335762, 34812.47494455, 36189.72464132],
           [18699.        , 18074.04737494, 17659.42861689, ...,
            20877.37603099, 19916.24902845, 19025.67327989]])




```python
c = np.mean(np.maximum(S[:,-1] - K,0))
print('European call', str(c))
```

    European call 2496.5548527209858


# Greeks

1.DELTA
Delta, ∆, measures the rate of change of the theoretical option value with respect to changes in the underlying asset's price. Delta is the first derivative of the value V of the option with respect to the underlying instrument's price S.


```python
def delta(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        delta = np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0)
    elif payoff == "put":
        delta =  - np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0)
    
    return delta


```


```python
delta(18699, 19154, 0.25, 0.04, 0, 0.7, 'call')


```




    0.55364716060398



if the underlying asset price decrease from 18,699 to 18698: call option value will reduce by $0.55 (dollars

fig 1.1


```python
S = np.linspace(50,150,11)
Delta_Call = np.zeros((len(S),1))
Delta_Put = np.zeros((len(S),1))
for i in range(len(S)):
    Delta_Call [i] = delta(S[i], 18699, 0.25, 0.04, 0, 0.7, 'call')
```


```python
fig = plt.figure()
plt.plot(S, Delta_Call, '-')
plt.grid()
plt.xlabel('crypto currency')
plt.ylabel('Delta')
plt.title('Delta')
plt.legend(['Delta for Call'])

```




    <matplotlib.legend.Legend at 0x7f80026cb040>




    
![png](output_39_1.png)
    


## 2.GAMMA

Gamma, Γ, measures the rate of change in the delta with respect to changes in the underlying price. Gamma is the second derivative of the value function with respect to the underlying price.




```python
def gamma(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    gamma = np.exp(- q * T) * si.norm.pdf(d1, 0.0, 1.0) / (vol * S * np.sqrt(T))
    
    return gamma



```


```python
gamma(18699, 19154, 0.25, 0.04, 0, 0.7, 'call')
```




    6.0405027206724167e-05



if the underlying asset price increase from 18699 to 18700 the call option for Delta will increase by 6.04

fig 1.2


```python
S = np.linspace(50,150,11)
Gamma = np.zeros((len(S),1))
for i in range(len(S)):
    Gamma [i] = gamma(S[i], 18699, 0.25, 0.04, 0, 0.7, 'call')

```


```python
fig = plt.figure()
plt.plot(S, Gamma, '-')
plt.grid()
plt.xlabel('crypto currency')
plt.ylabel('Gamma')
plt.title('Gamma')
plt.legend(['Gamma for Call' ])

```




    <matplotlib.legend.Legend at 0x7f8002fd2f70>




    
![png](output_46_1.png)
    


## 3.SPEED
Speed measures the rate of change in Gamma with respect to changes in the underlying price


```python
def speed(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    speed = - np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0) / ((vol **2) * (S**2) * np.sqrt(T)) * (d1 + vol * np.sqrt(T))
    
    return speed
```


```python
speed(18699, 19154, 0.25, 0.04, 0, 0.7, 'call')

```




    -2.2376499613033007e-09



when the underlying asset price change from 18699 to 1868 then the gamma will decrease by 2.237

fig 1.3


```python
S = np.linspace(50,150,11)
Speed = np.zeros((len(S),1))
for i in range(len(S)):
    Speed [i] = speed(S[i], 18699, 19154, 0.25, 0.04, 0.7, 'call')

```


```python
fig = plt.figure()
plt.plot(S, Speed, '-')
plt.grid()
plt.xlabel('crypto currency')
plt.ylabel('Speed')
plt.title('Speed')
plt.legend(['Speed for Call'])
```




    <matplotlib.legend.Legend at 0x7f80030ab520>




    
![png](output_53_1.png)
    


## 4.theta


```python

```

Theta, 𝜃, measures the sensitivity of the value of the derivative to the passage of time (see Option time value): 
the "time decay"


```python
def theta(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        theta = vol * S * np.exp(-q * T) * si.norm.pdf(d1, 0.0, 1.0) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) + r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        theta = vol * S * np.exp(-q * T) * si.norm.pdf(-d1, 0.0, 1.0) / (2 * np.sqrt(T)) + q * S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0) - r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return theta

```


```python
theta(18699, 19154, 0.25, 0.04, 0, 0.7, 'call')
```




    5489.25919524196



when there is an increase in time then call option value will increase by $1372.

fig 1.4


```python
T = np.linspace(0.25,3,12)
Theta_Call = np.zeros((len(T),1))
Theta_Put = np.zeros((len(T),1))
for i in range(len(T)):
    Theta_Call [i] = theta(18699, 19154, 0.25, 0.04, 0, 0.7, 'call')
```


```python
fig = plt.figure()
plt.plot(T, Theta_Call, '-')
plt.grid()
plt.xlabel('Time to Expiry')
plt.ylabel('Theta')
plt.title('Theta')
plt.legend(['Theta for Call' ])
```




    <matplotlib.legend.Legend at 0x7f80030ff9a0>




    
![png](output_62_1.png)
    


RHO


fig 1.5


```python
def rho(S, K, T, r, q, vol, payoff):
    
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    if payoff == "call":
        rho =  K * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif payoff == "put":
        rho = - K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return rho
```


```python
rho(18699, 19154, 0.25, 0.04, 0, 0.7, 'call')


```




    1966.6834409173605




```python
if the interest rate increase by 1% then call price will increase by $19.67
```


```python
r = np.linspace(0,0.1,11)
Rho_Call = np.zeros((len(r),1))
Rho_Put = np.zeros((len(r),1))
for i in range(len(r)):
    Rho_Call [i] = rho(18699, 19154, 0.25, 0.04, 0, 0.7, 'call')
```


```python
fig = plt.figure()
plt.plot(r, Rho_Call, '-')

plt.grid()
plt.xlabel('Interest Rate')
plt.ylabel('Rho')
plt.title('Rho')
plt.legend(['Rho for Call', ])
```




    <matplotlib.legend.Legend at 0x7f8003198ac0>




    
![png](output_69_1.png)
    



