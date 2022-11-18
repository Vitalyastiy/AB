
# %%
##############################################################
#корреляция
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv')

axes = pd.plotting.scatter_matrix(data, figsize=(10,10), diagonal='kde', grid=True)
corr = data.corr().values

for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
plt.show()
# %%
#correl
corr = data.corr()
data.corr()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv')

grid = sns.PairGrid(data)
grid.map(plt.scatter)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv')
# Function to calculate correlation coefficient between two arrays
def corr(x, y, **kwargs):
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.45), size = 20, xycoords = ax.transAxes)

g = sns.PairGrid(df, diag_sharey=False)
# g.map_upper(sns.scatterplot)
g.map_upper(corr)
g.map_lower(sns.scatterplot)
g.map_diag(sns.kdeplot)
# %%
import statsmodels.formula.api as smf
df = pd.read_csv('http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv')

# Найдём такой набор фичей, что R-squared максимальна
feats = {'metro_res', 'white', 'hs_grad', 'female_house'}

# Рекурсивно переберём все сочетания переменных
def brute_force(params: set, formula: str, result: set) -> str:
    if not params:
        result.add(formula.lstrip('+ '))
        return 
    else:
        new_params = params.copy()
        for el in params:
            new_params.remove(el)

            brute_force(new_params, formula + ' + ' + el, result)
            brute_force(new_params, formula, result)

res = set()          
brute_force(feats, '', res)
res.remove('')

# Теперь посчитаем Adjusted R-Square для каждого сочетания и выведем на экран
results = {}

# Теперь посчитаем Adjusted R-Square для каждого сочетания
for formula in res:
    lm = smf.ols(formula='poverty ~ ' + formula, data=df).fit()
    results[formula] = f'{lm.rsquared_adj:.2}'
    
# выведем отсортированные результаты на экран
d = sorted(results.items(), key=lambda x: x[1], reverse=True)
for i in d:
    print(i[1],'\t',i[0])

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import itertools

feature_list = ['metro_res', 'white', 'hs_grad', 'female_house']
result = {'combinations_features':[], 'coef':[],'r2_score':[], 'MSE':[]}
y = data[['poverty']]

for count_features in range(len(feature_list),0,-1):
  for feature in itertools.combinations(feature_list, count_features):
    result['combinations_features'].append(list(feature))
    X = data[list(feature)]
    model_lr = LinearRegression().fit(X, y)
    result['coef'].append(model_lr.coef_[0])
    result['r2_score'].append(r2_score(y, model_lr.predict(X)))
    result['MSE'].append(mean_squared_error(y, model_lr.predict(X)))

pd.DataFrame(result).sort_values('r2_score', ascending=False)
# %%
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

states = pd.read_csv('http://d396qusza40orc.cloudfront.net/statistics/lec_resources/states.csv') # предварительно скачайте. датасет предоставлен вначале курса


sns.heatmap(states.corr(), annot=True) # построение карты корреляции. Замечаем, что female_house тут явно лишняя. Наблюдается мультиколлинеарность

# сначала поработаем со статистикой:
X, y = states.drop(['state', 'female_house', 'poverty'], axis=1), states.poverty
X_first_var = sm.add_constant(X)
model_OLS = sm.OLS(y, X_first_var).fit()
model_OLS.summary()
# Второй вариант
model2 = smf.ols(formula='poverty ~ metro_res + white + hs_grad', data=states).fit()
model2.summary()
# Создадим саму модель
linear_model = LinearRegression()
linear_model.fit(X,y)
linear_model.score(X,y) # посмотрим на \(R^2\)

def model(x):
    return np.asscalar(linear_model.predict([x]))

arr = [80,50,90]
model(arr)
# %%
