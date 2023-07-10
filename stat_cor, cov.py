# %%
import matplotlib.pyplot as plt
import numpy as np

N = 100

x = np.array([np.random.randint(50,100) for i in range(N)])
y = np.array([i+np.random.randint(80,120)-np.random.randint(-20,20) for i in x])

plt.scatter(x, y)
b1, b0 = np.polyfit(x, y, 1) #  b0 - intercept, b1 - slope
plt.plot(x, b0 + b1*x, color='red')
plt.show()
# %%
from scipy import stats
import matplotlib.pyplot as plt

def cortest(corr, lenX):
    #t value
    t = corr[0] * ((len(X) - 2) / (1 - corr[0]**2))**0.5
    print("data: x and y")
    print("t = {}, df = {}, p-value = {}".format(round(t,5), len(X)-2, round(stats.t.sf(np.abs(t), 48)*2,5)))

    # Use the Fisher transformation to get z
    z = np.arctanh(corr[0])
    #print("z value: {}".format(z))

    # And, the sigma value i.e standard error
    sigma = (1/((len(X)-3)**0.5))
    #print("sigma value: {}".format(sigma))

    # Get normal 95% interval probability density function for normal continuous random variable apply two-sided conditional formula
    cint = z + np.array([-1, 1]) * sigma * stats.norm.ppf((1+0.95)/2)
    
    #Finally take hyperbolic tangent to get interval values for 95%
    interval = np.tanh(cint)
    
    if corr != 0:
        print("alternative hypothesis: true correlation is not equal to 0")
    print("95 percent confidence interval:")
    print(interval)
    print("sample estimates:")
    print("cor")
    print(corr[0])

# так вычисляем корреляцию
X = stats.norm.rvs(loc=0, scale=1, size=50)
Y = stats.norm.rvs(loc=0, scale=1, size=50)
corr = stats.pearsonr(X, Y)

# так строим scatter plot
plt.scatter(X,Y)
plt.grid()
plt.show()

#print("correlation: {}".format(corr))

#т.к. из лекции мы не знаем выборки, то сразу подставляем корреляцию из лекции
cortest((0.2858888,), 50)
# %%
import pandas as pd
df = pd.DataFrame({'x': [4, 5, 2 , 3, 1], 'y': [2, 1, 4, 3, 5]})
df.corr()
# %%
X = [4,5,2,3,1]
Y = [2,1,4,3,5]
N = 5

ay = sum(X)/N # среднее X
ax = sum(Y)/N # среднее Y

cov = sum((x-ax)*(y-ay) for x,y in zip(X,Y))/(N-1) # Ковариация

Dx = sum((x-ax)**2 for x in X)/(N-1) # Дисперсия X
Dy = sum((y-ay)**2 for y in Y)/(N-1) # Дисперсия Y

sdx = Dx**.5 # Стандртное отклонение X
sdy = Dy**.5 # Стандртное отклонение Y

rxy = cov/(sdx*sdy) # Коэффициент корреляции

rxy
# %%

def cov(x, y):
    # x or y, no sense
    m = len(x)
    return np.sum((x - x.mean()) * (y - y.mean())) / (m - 1)

def std(sample):
    return np.sqrt(np.sum((sample - sample.mean())**2) / (len(sample) - 1))

def corr(x, y):
    return cov(x, y) / (std(x) * std(y))

x = np.array([4, 5, 2, 3, 1])
y = np.array([2, 1, 4, 3 ,5])

covariation = cov(x, y)
correlation = corr(x, y)

correlation
covariation
# %%
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


x = np.array([4,5,2,3,1])
y = np.array([2,1,4,3,5])

print(scipy.stats.pearsonr(x, y))

plt.plot(x, y)
# %%
