#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r'C:\Users\vitaly.flerin\Desktop\states.csv')

Y = np.array(data['poverty'])
X = np.array(data['hs_grad'])

N = len(Y)

sd_y = (sum((i - Y.mean()) ** 2 for i in Y) / (N - 1)) ** 0.5
sd_x = (sum((i - X.mean()) ** 2 for i in X) / (N - 1)) ** 0.5

r_cov = sum(map(lambda x, y: (x - X.mean()) * (y - Y.mean()), X, Y)) / (N - 1)
r_xy = r_cov / (sd_y * sd_x)
R_1 = r_xy ** 2

b_1 = sd_y / sd_x * r_xy
b_0 = Y.mean() - b_1 * X.mean()
f = lambda x: b_0 + b_1 * x

SSres = sum(map(lambda y, x: (y - f(x)) ** 2, Y, X))
SStotal = sum((i - Y.mean()) ** 2 for i in Y)

R_2 = 1 - SSres / SStotal

print(r_xy, b_1)
print(SSres, SStotal)
print(R_2)


plt.scatter(list(X), list(Y), s=15)
plt.xlabel('hs_glad')
plt.ylabel('poverty')
plt.title('Linear Regression')
x = np.array(X)
plt.plot(x, f(x), c="orange")

plt.show()
# %%
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

df = pd.read_csv('http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv')
b0 = scipy.stats.linregress(df['hs_grad'], df['poverty']).intercept
b1 = scipy.stats.linregress(df['hs_grad'], df['poverty']).slope
y = b0 + b1*df['hs_grad']

plt.scatter(df['hs_grad'], df['poverty'])               # scatter plot
plt.text(x=87,y=17,s='R = -0.75', fontsize=20) # text of R value
plt.plot(df['hs_grad'], y, color='green')               # regression line
plt.grid()                                                                # grid lines
plt.xlabel('graduation (%)')                                 # xlabel
plt.ylabel('poverty (%)')                                       # ylabel
plt.title('poverty \ grad')                                      # title
plt.show()                                                              # show everything
# %%
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
URL='http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv'
df=pd.read_csv(URL)
x=df["hs_grad"]
y=df["poverty"]
gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
mn=np.min(x)
mx=np.max(x)
x1=np.linspace(mn,mx,500)
y1=gradient*x1+intercept
plt.plot(x,y,'ob') #Рассталяем точки
plt.plot(x1,y1,'-r') #Рисуем линию
plt.show() #Рисуем график

stats.linregress (x, y)
# %%

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

data=pd.read_csv(r'C:\Users\vitaly.flerin\Desktop\states.csv', sep=',')

# рассчет ковариационной матрицы двух векторов, на выходе ковариация и дисперсии каждого из массивов, 
# или можно рассчитать вручную по формулам из уроков cov - урок 3_1 третье видео; D - урок 1_5 третье видео
X = data['hs_grad']
Y = data['poverty']
ssxm, ssxym, ssyxm, ssym = np.cov(X,Y).flat

# рассчет корреляции по формуле из урока 3_1 третье видео: r = cov/stdx*stdy, 
# т. к. в ковариационной матрице у нас дисперсии, то в знаменателе нужно взять квадратный корень 
if ssxm == 0 or ssym == 0:
    r = 0
else:
    r = ssxym / (ssxm * ssym)**0.5
    
# расчет slope и intercept из урока 3_3 второе видео: slope = (stdy/stdx) * r
slope = r * (ssym/ssxm)**0.5
intercept = np.mean(Y) - slope * np.mean(X)

df = len(X) - 2
# рассчет t - значения, вероятности p и стандартной ошибки для X параметра подсмотренно тут: 
# https://github.com/scipy/scipy/blob/v1.4.1/scipy/stats/_stats_mstats_common.py#L15-L144
tx = r * np.sqrt(df / ((1.0 - r)*(1.0 + r)))
sterrestx = np.sqrt((1 - r**2) * ssym / ssxm / df)
px = 2 * stats.t.sf(np.abs(tx), df)

# Находим сумму квадратов по параметру X
s = [i**2 for i in X]
sterresty = (sterrestx**2/51*sum(s))**0.5

# рассчет t - значения, вероятности p и стандартной ошибки для Y параметра подсмотренно тут: 
# https://en.wikipedia.org/wiki/Simple_linear_regression
ty = intercept/sterresty
py = 2 * stats.t.sf(np.abs(ty), df)

# рассчет F-статистики подсмотренно тут:
# https://www.chem-astu.ru/science/reference/fischer.html
F = r**2/(1-r**2)*df
p_val = stats.f.sf(F, 1, df)

# создаем таблицу как в лекции
ttlinear = pd.DataFrame(data = {
    'Estimate':[round(num, 4) for num in [intercept,slope]], 'Std.Error':[round(num, 4) for num in [sterresty, sterrestx]], 
    't value':[round(num, 2) for num in [ty,tx]], 'Pr(>|t|)':[py,px]}, 
                        index = ['(Intercept)','hs_grand'])

# вычисляем остатки
#residuals = Y - intercept - slope*X

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

# точечный график
ax1.scatter(X,Y)

# регрессионная прямая
x1=np.linspace(min(X),max(X),51)
y1=intercept + slope*x1
plt.plot(x1,y1,'-r',label='b0 = {}, b1 = {}'.format(round(intercept,2), round(slope,2)))

plt.grid()
plt.title('Связь бедности и уровня образования')
plt.xlabel('Среднее образование (%)')
plt.ylabel('Бедность (%)')
plt.xticks(np.arange(76,93, step=4))
plt.yticks(np.arange(5,18, step=5))
plt.legend(loc='upper right')
plt.show()


print(ttlinear)
print('')
print('Multiple R-squared: {},'.format(round(r**2,4)))
print('F-statistic(1,{}) = {}, p-value = {}'.format(df, round(F,2), p_val))




# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma

# получаем данные и вычисляем коэффициэнты регрессионной модели
data=pd.read_csv(r'C:\Users\vitaly.flerin\Desktop\states.csv', sep=',')
intercept, slope = sm.ols(formula="poverty ~ hs_grad", data=data).fit().params
X = data['hs_grad']
Y = data['poverty']

# вычисляем остатки - на сколько Y далека от теоретической модели
residuals = Y - intercept - slope*X
x1=np.linspace(min(X),max(X),len(X))

# создаем графики
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 9))
# scatter plot
ax1.scatter(X,residuals)
ax1.plot(x1,[0]*len(x1),'--r')
# Q-Q plot
sma.qqplot(residuals, line='s', ax = ax2)

ax1.grid()
ax1.set_title('Анализ остатков')
ax1.set_xlabel('Среднее образование (%)')
ax1.set_ylabel('Остатки')
ax1.set_xticks(np.arange(76,93, step=4))
ax1.set_yticks(np.arange(-5,6, step=2.5))

ax2.set_title('QQ-Plot')

fig.tight_layout(pad=3.0)
plt.show()
# %%
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

def calc_ols_statmodels(x, y):
    x_for_ols = sm.add_constant(x)
    model = sm.OLS(y, x_for_ols)
    results = model.fit()
    print('statsmodels: ', results.summary())
    
    # regression formula --------------------------------
    if results.params[1]>0:
        sign="+"
    else:
        sign="-"
    formula = f"y = {results.params[0]:.2f} {sign} {np.abs(results.params[1]):.2f}*x"
    print("-"*80)
    print("OLS Formula: ", formula)

    # Graphics ================================
    fig = plt.figure(figsize=(16,9), constrained_layout=True)
    gs = fig.add_gridspec(ncols=3, nrows=2)
    ax_main = fig.add_subplot(gs[0,:])
    ax_resid = fig.add_subplot(gs[1,0])
    ax_hist = fig.add_subplot(gs[1,1])
    ax_qqplot = fig.add_subplot(gs[1,2])

    # Scatterplot -----------------------------
    sns.scatterplot(x=x, y=y, ax=ax_main, label="fact")
    ax_main.plot(x, results.predict(), color='red', alpha=0.5, label = formula)
    ax_main.set_title(f"Regression scatterplot. R2={results.rsquared:.2f}", fontsize=14)
    ax_main.legend()

    # Residuals -------------------------------
    sns.scatterplot(x=x, y=results.resid, ax=ax_resid, label="residuals")
    ax_resid.hlines(0, x.min(), x.max(), linestyle='--', colors='red', alpha=0.5)
    ax_resid.set_title(f"Residuals vs Fitted values", fontsize=14)

    # Hist -------------------------------------
    sns.histplot(results.resid, ax=ax_hist,)
    ax_hist.set_title("Histogram of Residuals", fontsize=14)
    # ax_hist.legend()

    # QQ Plot ----------------------------------
    # sm.qqplot(results.resid, ax=ax_qqplot) # не такой секси
    stats.probplot(results.resid, dist="norm", plot=ax_qqplot)
    ax_qqplot.set_title("Normal QQ-Plot of Residuals", fontsize=14)

    plt.show()

# ==================================================    
# пример вызова процедуры
calc_ols_statmodels(data["hs_grad"],data["poverty"])    
# %%
# ячейка 1
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ячейка 2

df = pd.read_csv(r'C:\Users\vitaly.flerin\Desktop\states.csv', sep=',')

# ячейка 3
model = smf.ols(formula="poverty ~ hs_grad", data=df).fit()
model.summary()

# ячейка 4
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(model, 'hs_grad', fig=fig)

# ячейка 5
residuals = model.resid
fig = sm.qqplot(residuals, line='s')
plt.show()
#процент бедности при уровне среднего образвания 62%
# ячейка 6
p = [62]
p = pd.DataFrame({'hs_grad': p})
model.predict(p)
# %%




import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = pd.read_csv('http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv')

data_crop = data[['white', 'hs_grad', 'poverty']]
data_crop.head()
white, hs_grad, poverty = [column for column in data_crop.values.T]

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=white, ys=poverty, zs=hs_grad)

ax.set_xlabel('White(%)')
ax.set_ylabel('Poverty(%)')
ax.set_zlabel('Higher education(%)')

plt.show()
# %%


from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv')

X = data[['white', 'hs_grad']]
y = data['poverty']

reg = LinearRegression().fit(X, y)

d1, d2 = list(), list()
for x in np.linspace(min(data['white']), max(data['white']), 100):
    for y in np.linspace(min(data['hs_grad']), max(data['hs_grad']), 100):
        d1.append(x)
        d2.append(y)
d1 = np.array(d1).reshape(-1, 1)
d2 = np.array(d2).reshape(-1, 1)
p = reg.predict(np.concatenate([d1, d2], axis=1))


fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

ax.scatter(data['hs_grad'], data['white'], data['poverty'], s=50)

ax.plot_trisurf(d2.ravel(), d1.ravel(), p.ravel(), alpha=0.2)

ax.set_xlabel('Higher education(%)')
ax.set_ylabel('White(%)')
ax.set_zlabel('Poverty(%)')

ax.elev = 10
ax.azim = -60

plt.show()
# %%


import statsmodels.formula.api as smf
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

%matplotlib notebook
%matplotlib inline
data = pd.read_csv('http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv')

data.head()

lm = smf.ols(formula='poverty ~ metro_res + hs_grad', data=data).fit()

lm.params

def f(x, y):
    return lm.params.Intercept + lm.params.hs_grad * x  + lm.params.metro_res * y

x = data.hs_grad.sort_values()
y = data.metro_res.sort_values()

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.3)
data_below_serf = data[data['poverty'] < f(data['hs_grad'], data['metro_res'])]
data_above_serf = data[data['poverty'] >= f(data['hs_grad'], data['metro_res'])]
ax.scatter(data_below_serf.hs_grad, data_below_serf.metro_res, data_below_serf.poverty, c='r', marker='o')
ax.scatter(data_above_serf.hs_grad, data_above_serf.metro_res, data_above_serf.poverty, c='g', marker='o')
ax.set_xlabel('hs_grad')
ax.set_ylabel('metro_res')
ax.set_zlabel('poverty')
# %%
# libraries
import pandas as pd
import statsmodels.api as sm

# независимая переменная
X = data['poverty']
# зависимые переменные
Y = sm.add_constant(data[['metro_res', 'white', 'hs_grad', 'female_house']])

# обучение модели
model = sm.OLS(X, Y).fit()
# вывод результатов (так же можно print(model.summary()))
model.summary()
# %%
