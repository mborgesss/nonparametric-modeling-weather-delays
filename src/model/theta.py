"""Modelos que calculam a probabilidade de realização de cada tarefa."""

# %% libs
from re import M
import pandas as pd
from scipy.optimize import differential_evolution
import warnings
import numpy as np
from scipy import optimize


# %% Sine wave
def opt_sine_func(x, K, A, phi):
    """Função a ser otimizada."""
    return K + A * np.cos(2*np.pi*1*(x/365 - phi))

def convert_date(df, column_date='date'):
    df = df.copy()
    df['year'] = pd.to_datetime(df[column_date]).dt.year
    df['month-day'] = pd.to_datetime(df[column_date]).dt.strftime('%m-%d')
    return df

# %% KNN
class KNN:
    """Calcula a probabilidade de realização de cada tarefa."""

    def __init__(self, window=1, alpha=1, fi='average_window', fy='exponencial'):
        self.alpha = alpha
        self.window = window
        if fy == 'exponencial':
            self.fy = (lambda df: self.fy_exponencial(alpha=alpha, df=df))
        else:
            raise ValueError('Invalid value to fy arg')
        if fi == 'average_window':
            self.fi = (lambda theta: self.fi_average_window(window=window, theta=theta))
        elif fi == 'cossine':
            self.fi = (lambda theta: self.fi_cossine(theta=theta))
        else:
            raise ValueError('Invalid value to fi arg')

    def fit(self, data, column_date='date'):
        """Ajusta o modelo."""
        self.fitted = {}
        self.data = data.copy()
        self.column_date = column_date
        df = convert_date(data, column_date=column_date)
        self.tasks_columns = data.drop(columns=[column_date]).columns.tolist()
        theta = {}
        for task in self.tasks_columns:
            theta[task] = self.fi(theta=self.fy(
                df=(
                    df
                    .rename(columns={task: 'task'})
                    .loc[:, ['year', 'month-day', 'task']]
                )))
            theta[task] = [min(t, 1) for t in theta[task]]
        self.theta = theta
        return theta

    def fy_exponencial(self, alpha, df):
        """Group by month, day and shifts. Apply exponential decay."""
        df = df.copy()
        df = df.sort_values(['year', 'month-day'])
        # Extract year
        years = df['year'].values
        maxyear = years.max()
        # calculating exponential decay weights
        df['weight'] = (alpha**(maxyear - years))
        # apply exponential decay
        df['task'] = df['task']*df['weight']
        # group by
        df = (
                df.loc[
                    lambda df: df['task'].notna(),
                    ['task', 'month-day', 'weight']
                    ]
                .groupby(['month-day'])
                .agg('sum')
                .reset_index()
                .sort_values('month-day')
            )
        df['task'] = df['task']/df['weight']
        df = df[['task', 'month-day']]
        theta = df['task'].tolist()
        return theta

    def fi_cossine(self, theta):
        """Calcula os parametros para cada task."""
        theta = np.array(theta)
        if len(theta) == 366: # exist leap year
            leap_year = [59.5]
        else:
            leap_year = []
        X = np.concatenate([
            np.arange(1, 60),
            leap_year,
            np.arange(60, 366)
            ])
        # chute inicial dos parametros
        def mse(parameterTuple):
            warnings.filterwarnings("ignore")
            val = opt_sine_func(X, *parameterTuple)
            return np.sum((theta - val) ** 2.0)

        def generate_Initial_Parameters():
            parameterBounds = [[0, 1], [0, 1], [0, 1]]
            result = differential_evolution(mse, parameterBounds, seed=0)
            return result.x

        generic_parameters = generate_Initial_Parameters()
        params, _ = optimize.curve_fit(
            opt_sine_func, X, theta, p0=generic_parameters)
        adj_values = np.array([
            max(min(
                opt_sine_func(x=x, K=params[0], A=params[1], phi=params[2]),
                1), 0)
            for x in X
            ])
        error = (theta-np.array(adj_values))
        if 'cossine' not in self.fitted:
            self.fitted['cossine'] = {}
        self.fitted['cossine']['task'] = {
            'mse': (error**2).mean(),
            'K': params[0],
            'A': params[1],
            'phi': params[2]
            }
        return adj_values.tolist()

    def fi_average_window(self, window, theta):
        """
        Apply moving average.

        Parameters
        ----------
        window : int
            length of moving window.
        """
        n = len(theta)
        if window > 0:
            adj_values = theta[(-window):] + theta + theta[:window]
            adj_values = [
                np.mean(adj_values[(i-window):(i+window+1)])
                for i  in range(window, n+window)
                ]
        else:
            adj_values = theta
        error = (np.array(theta)-np.array(adj_values))
        if 'average_window' not in self.fitted:
            self.fitted['average_window'] = {}
        self.fitted['average_window']['task'] = {
            'mse': (error**2).mean()
            }
        return adj_values
