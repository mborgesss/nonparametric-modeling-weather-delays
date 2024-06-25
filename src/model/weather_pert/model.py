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

    def __init__(self, smoothing=1, alpha=1, column_date='date', mc=False, cossine=False):
        self.column_date = column_date
        self.mc = mc
        self.alpha = alpha
        self.smoothing = smoothing
        self.cossine = cossine
    @classmethod
    def by_task(cls, params, column_date='date'):
        params = {
            arg: {
                task: task_args[arg]
                for task, task_args in params.items()}
            for arg in ['alpha', 'smoothing', 'mc', 'cossine']
            }
        return cls(**params, column_date=column_date)

    def fit(self, data, tasks_conditions=None):
        """Ajusta o modelo."""
        self.fitted = {}
        if tasks_conditions:
            self.data = data[[self.column_date]]
            for task, condition_func in tasks_conditions.items():
                condition_func = tasks_conditions[task]
                self.data[task] = data.apply(lambda row: condition_func(**row)*1, axis=1)
            self.tasks = list(tasks_conditions.keys())
        else:
            self.data = data
            self.tasks = [task for task in self.data.columns if task != self.column_date]
        self.data['year'] = pd.to_datetime(self.data[self.column_date]).dt.year
        self.data['month-day'] = pd.to_datetime(self.data[self.column_date]).dt.strftime('%m-%d')
        self.data = self.data.loc[lambda df: df['month-day'] != '02-29']
        self.data['j'] = pd.to_datetime(self.data[self.column_date]).dt.strftime('%j').astype(int)
        

        self.theta = {task: [] for task in self.tasks}
        self._fit_exponencial_smoothing()
        self._fit_year_smoothing()
        self._fit_cossine()

    def predict(self, date_start, project_schedule, project_cost={}, B=1000):
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
        
        theta = self.theta

        month_day_start = datetime.datetime.strptime(date_start, "%Y-%m-%d").strftime('%m-%d')
        doy_start = md_to_doy[month_day_start]

        total_duration = sum(project_schedule.values())
        n_days =  total_duration*5
        probs = np.random.random(size=(B, int(n_days)))

        def simulator(probs):
            nonlocal n_days
            i = 0
            doy = doy_start
            cost = 0
            for task, planned_duration in project_schedule.items():
                doy_task_start = int(doy)
                duration = int(planned_duration)
                mc = self.mc[task] if isinstance(self.mc, dict) else self.mc
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
                if project_cost:
                    estimate_duration = doy - doy_task_start
                    cost += project_cost[task](estimate_duration=estimate_duration, planned_duration=planned_duration)
            return {'duration': doy - doy_start, 'cost': cost}
        durations = []
        costs = []
        for b in range(B):
            simulation = simulator(probs=probs[b])
            durations.append(simulation['duration'])
            costs.append(simulation['cost'])
        if project_cost:
            return {'durations': durations, 'costs': costs}
        else:
            return durations
    

    def _fit_exponencial_smoothing(self):
        self.data = self.data.sort_values(self.column_date)
        for task in self.tasks:
            smoothing = self.smoothing[task] if isinstance(self.smoothing, dict) else self.smoothing
            mc = self.mc[task] if isinstance(self.mc, dict) else self.mc
            alpha = self.alpha[task] if isinstance(self.alpha, dict) else self.alpha

            _df = self.data.loc[self.data[task].notna(), ['year', 'month-day', 'j', task]]

            years = _df['year'].values
            work = _df[task].values

            maxyear = years.max()
            weights = (alpha**(maxyear - years))
            work_weights = work*weights

            _df_agg = (
                _df
                .assign(work_weights = work_weights, weights = weights)
                .groupby(['month-day']).agg('sum').reset_index()
                .assign(theta = lambda df: df['work_weights']/df['weights']))
            theta = (
                month_days
                .merge(_df_agg, on='month-day', how='left') .fillna(0)
                .sort_values('month-day')['theta'].tolist())
            
            if mc:
                self.theta[task] = {-1: theta}
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
                    for doy in md_to_doy.values():
                        range_size = smoothing
                        is_neighbor = [False]
                        while not np.max(is_neighbor) and range_size <= 182:
                            is_neighbor = np.in1d(doys_, doy_range[(365 + doy - range_size):(365 + 1 + doy + range_size)])
                            range_size += 1
                        if not np.max(is_neighbor):  
                            theta_.append(theta[doy])
                        else:
                            theta_.append(np.sum(work_weights_[is_neighbor])/np.sum(weights_[is_neighbor]))
                    self.theta[task][is_working] = theta_
            else:
                self.theta[task] = theta   

    def _fit_cossine(self):
        for task in self.tasks:
            cossine = self.cossine[task] if isinstance(self.cossine, dict) else self.cossine
            mc = self.mc[task] if isinstance(self.mc, dict) else self.mc
            if cossine:
                """Calcula os parametros para cada task."""
                theta_task = self.theta[task]
                if not mc:
                    theta_task = [(0, theta_task)]
                else:
                    theta_task = theta_task.items()
                    
                for i, theta in theta_task:
                    theta = np.array(theta)

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
                    theta = adj_values.tolist()
                    if not mc:
                        self.theta[task] = theta
                    else:
                        self.theta[task][i] = theta

    def _fit_year_smoothing(self):
        for task in self.tasks:
            smoothing = self.smoothing[task] if isinstance(self.smoothing, dict) else self.smoothing
            mc = self.mc[task] if isinstance(self.mc, dict) else self.mc
            if not mc and smoothing > 0:
                theta = self.theta[task]
                n = len(theta)
                adj_values = theta[(-smoothing):] + theta + theta[:smoothing]
                adj_values = [
                    np.mean(adj_values[(i-smoothing):(i+smoothing+1)])
                    for i  in range(smoothing, n+smoothing)
                    ]
                error = (np.array(theta)-np.array(adj_values))
                if 'average_window' not in self.fitted:
                    self.fitted['average_window'] = {}
                self.fitted['average_window'][task] = {
                    'mse': (error**2).mean()
                    }
                self.theta[task] = adj_values
