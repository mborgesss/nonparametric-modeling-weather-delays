"""Modelos que calculam a probabilidade de realização de cada tarefa."""


# %% libs
from re import M
import pandas as pd
from scipy.optimize import differential_evolution
import warnings
import numpy as np
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

# %% libs
import numpy as nps
import datetime

# Inicio e fim do ano
start_date = datetime.date(year=2001, month=1, day=1)
end_date = datetime.date(year=2001, month=12, day=31)

# Lista para armazenar os dias
month_days = []
delta = datetime.timedelta(days=1)
while start_date <= end_date:
    month_days.append(start_date.strftime("%m-%d"))
    start_date += delta
md_to_doy = {month_day: i for i, month_day in enumerate(month_days)}
month_days = pd.DataFrame({'month-day': list(md_to_doy.keys()), 'doy': list(md_to_doy.values())}).sort_values('month-day')
doy_range = np.array(list(range(0, 365))*3)


# %% Sine wave
def opt_sine_func(x, K, A, phi):
    """Função a ser otimizada."""
    return K + A * np.cos(2*np.pi*1*(x/365 - phi))

# %% KNN
class WeatherPert:
    """Calcula a probabilidade de realização de cada tarefa."""

    def __init__(self, smoothing=1, alpha=1, column_date='date', mc=False):
        if not isinstance(self.alpha, float):
            raise ValueError('Invalid value to alpha arg')
        if not (isinstance(self.smoothing, int) or self.smoothing == 'cossine'):
            raise ValueError('Invalid value to smoothing arg')
        if mc and self.smoothing == 'cossine':
            raise ValueError('Invalid value to mc arg')
        self.column_date = column_date
        self.mc = mc
        self.alpha = alpha
        self.smoothing = smoothing

    def fit(self, data, tasks_conditions):
        """Ajusta o modelo."""
        self.fitted = {}
        self.data = data[[self.column_date]]
        for task, condition_func in tasks_conditions.items():
            condition_func = tasks_conditions[task]
            self.data[task] = data.apply(lambda row: condition_func(**row)*1, axis=1)
        self.data['year'] = pd.to_datetime(self.data[self.column_date]).dt.year
        self.data['month-day'] = pd.to_datetime(self.data[self.column_date]).dt.strftime('%m-%d')
        self.data = self.data.loc[lambda df: df['month-day'] != '02-29']
        self.data['j'] = pd.to_datetime(self.data[self.column_date]).dt.strftime('%j').astype(int)
        self.tasks = list(tasks_conditions.keys())

        self.theta = {task: [] for task in self.tasks}
        self._fit_exponencial_smoothing()
        self._fit_year_smoothing()
        self.smoothing





        theta = {}
        for task in self.tasks:
            if self.mc:
                theta[task] = {}
                theta[task][-1] = self.fi(theta=self.fy(
                    df=(
                        self.data
                        .rename(columns={task: 'task'})
                        .loc[:, ['year', 'month-day', 'task']]
                    ), fillna=0))
                theta[task][-1] = [min(t, 1) for t in theta[task][-1]]
                for cond in [0, 1]:
                    theta[task][cond] = self.fi(theta=self.fy(
                        df=(
                            self.data
                            .rename(columns={task: 'task'})
                            .loc[lambda df: (df['task'].shift(1) == cond)&(((df['i'].shift(1)-df['i']).abs() == 1)|((df['month-day'] == '01-01')&(df['month-day'] == '12-31'))), ['year', 'month-day', 'task']]
                        ), fillna=theta[task][-1]))
                    theta[task][cond] = [min(t, 1) for t in theta[task][cond]]
            else:
                theta[task] = self.fi(theta=self.fy(
                    df=(
                        self.data
                        .rename(columns={task: 'task'})
                        .loc[:, ['year', 'month-day', 'task']]
                    ), fillna=0))
                theta[task] = [min(t, 1) for t in theta[task]]
        self.theta = theta
        
        return theta
   


    def predict(self, date_start, project_schedule, B=1000):
        """
        Simulates the day each task was completed,B times,using parallel computing.
        Parameters
        ----------
        B : int
            number of simulations.
        threads : int
            number of threads.
        njobs : int
            number of jobs.
        task_control : create_task_config class
        format_result : string, optional
            result format. The default is 'DataFrame'.
        Returns
        -------
        results : list or dataframe pandas
        """
        mc = self.mc
        theta = self.theta

        month_day_start = datetime.datetime.strptime(date_start, "%Y-%m-%d").strftime('%m-%d')
        doy_start = month_day_i[month_day_start]

        total_duration = sum(project_schedule.values())
        n_days =  total_duration*5
        probs = np.random.random(size=(B, int(n_days)))

        def simulator(probs):
            nonlocal n_days
            i = 0
            doy = doy_start
            for task, duration in project_schedule.items():
                first = True
                while duration > 0:
                    if i == n_days:
                        n_days = total_duration
                        probs = np.random.random(size=n_days)
                        i = 0
                    if mc:
                        if first:
                            t = theta[task][-1][(doy % 365)]
                        else:
                            t = theta[task][is_working][(doy % 365)]
                    else:
                        t = theta[task][(doy % 365)]
                    p = probs[i]
                    is_working = (p <= t)*1
                    duration -= is_working
                    doy += 1
                    i += 1
                    first = False
            return doy - doy_start
        durations = [simulator(probs=probs[b]) for b in range(B)]
        return durations
    

    

    def _fit_exponencial_smoothing(self):
        for task in self.tasks:
            _df = self.data.loc[self.data[task].notna(), ['year', 'month-day', task]]
            _df = _df.sort_values(self.column_date)

            years = _df['year'].values
            work = df[task].values

            maxyear = years.max()
            weights = (self.alpha**(maxyear - years))
            work_weights = work*weights

            _df = (
                _df
                .assign(work_weights = work_weights, weights = weights)
                .groupby(['month-day']).agg('sum').reset_index()
                .assign(theta = lambda df: df['work_weights']/df['weights']))
            theta = (
                month_days
                .merge(df, on='month-day', how='left') .fillna(0)
                .sort_values('month-day')['theta'].tolist())
            
            if self.mc:
                self.theta[task][-1] = theta
                # pré filtro de dias seguinte
                pre_filter = (
                    ((_df['j'].shift(1)-_df['j']).abs() == 1)|
                    ((_df['month-day'] == '01-01')&(_df['month-day'].shift(1) == '12-31'))
                    ).values
                doys = _df['j'].values

                for is_working in [0, 1]:
                    filter_is_working = (np.roll(work, 1) == is_working)*pre_filter
                    weights_ = weights[filter_is_working]
                    work_weights_ = work_weights[filter_is_working]
                    doys_ = doys[filter_is_working]

                    theta_ = []
                    for doy in month_days.values():
                        range_size = self.smoothing
                        is_neighbor = [False]
                        while not np.max(is_neighbor) and range_size <= 182:
                            is_neighbor = np.in1d(doys_, doy_range[(365 + doy - range_size):(365 + 1 + doy + range_size)])
                            range_size += 1
                        if not np.max(is_neighbor):  
                            theta_.append(theta[doy])
                        else:
                            theta_.append(np.sum(work_weights_[filter])/np.sum(weights_[filter]))
                    self.theta[task][is_working] = theta_
            else:
                self.theta[task] = theta
            
        

        
            if self.mc:
                theta[task] = {}
                theta[task][-1] = self.fi(theta=self.fy(
                    df=(
                        self.data
                        .rename(columns={task: 'task'})
                        .loc[:, ['year', 'month-day', 'task']]
                    ), fillna=0))
                theta[task][-1] = [min(t, 1) for t in theta[task][-1]]
                for cond in [0, 1]:
                    theta[task][cond] = self.fi(theta=self.fy(
                        df=(
                            self.data
                            .rename(columns={task: 'task'})
                            .loc[lambda df: (df['task'].shift(1) == cond)&(((df['i'].shift(1)-df['i']).abs() == 1)|((df['month-day'] == '01-01')&(df['month-day'] == '12-31'))), ['year', 'month-day', 'task']]
                        ), fillna=theta[task][-1]))
                    theta[task][cond] = [min(t, 1) for t in theta[task][cond]]
            else:
                theta[task] = self.fi(theta=self.fy(
                    df=(
                        self.data
                        .rename(columns={task: 'task'})
                        .loc[:, ['year', 'month-day', 'task']]
                    ), fillna=0))
                theta[task] = [min(t, 1) for t in theta[task]]
        self.theta = theta
    


    def fy_exponencial(self, df, fillna=0):
        """Group by month, day and shifts. Apply exponential decay."""
        alpha = self.alpha
        if self.mc and self.average_window:
            theta = []
            doy_range = np.array(list(range(0, 365))*3)


            df = df[df['task'].notna()].merge(date_list, on='month-day', how='left')
            years = df['year'].values
            doys = df['i'].astype(int).values
            task = df['task'].values


            maxyear = years.max()
            weights = (alpha**(maxyear - years))
            task_weights = task*weights
            
            for month_day, i in month_day_i.items():
                range_size = self.window
                filter = [False]
                while not np.max(filter) and range_size <= 182:
                    filter = np.in1d(doys, doy_range[(365 + i - range_size):(365 + 1 + i + range_size)])
                    range_size += 1
                if not np.max(filter):  
                    theta.append(fillna[i])
                else:
                    theta.append(np.sum(task_weights[filter])/np.sum(weights[filter]))
            return theta
        else:
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
            df = date_list.merge(df, on='month-day', how='left').fillna(fillna).sort_values('month-day')
            theta = df['task'].tolist()
            return theta

    def _fit_year_smoothing(self):
        if self.smoothing == 'cossine':
            for task in self.tasks:
                """Calcula os parametros para cada task."""
                theta = np.array(self.task[task])
                X = np.arange(1, 366)
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
                self.fitted['cossine'][task] = {
                    'mse': (error**2).mean(),
                    'K': params[0],
                    'A': params[1],
                    'phi': params[2]
                    }
                self.theta[task] = adj_values.tolist()
        elif not self.mc and self.smoothing > 0:
            """
            Apply moving average.

            Parameters
            ----------
            window : int
                length of moving window.
            """
            for task in self.tasks:
                theta = self.theta[task]
                n = len(theta)
                adj_values = theta[(-self.smoothing):] + theta + theta[:self.smoothing]
                adj_values = [
                    np.mean(adj_values[(i-self.smoothing):(i+self.smoothing+1)])
                    for i  in range(self.smoothing, n+self.smoothing)
                    ]
                error = (np.array(theta)-np.array(adj_values))
                if 'average_window' not in self.fitted:
                    self.fitted['average_window'] = {}
                self.fitted['average_window'][task] = {
                    'mse': (error**2).mean()
                    }
                self.theta[task] = adj_values
