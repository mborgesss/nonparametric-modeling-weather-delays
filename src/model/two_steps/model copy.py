from .weather_generator import WeatherGenerator
from joblib import Parallel, delayed 
import numpy as np
from copy import deepcopy

def simulate(model):
    model = deepcopy(model)
    for task, duration in model.project_schedule.items():
        while duration > 0:
            weather = model.weather_generator.simulate_weather()
            is_work_day = model.tasks_conditions[task](**weather)
            duration -= is_work_day
    project_duration = (model.weather_generator.date - model.initial_date).days
    return project_duration


def map_prob(value, probs, cats):
    cumsum = 0
    for prob, cat in zip(probs, cats):
        cumsum += prob
        if value < cumsum:
            return cat
    return cats[-1]


class KNN:

    def __init__(
            self, precipitation_var, month_var, doy_var, weather_vars,
            wet_threshold=.3, wet_extreme_quantile_threshold=.8,
            window=7, distance_weights=None):
        self.weather_generator_args = {
            'precipitation_var': precipitation_var,
            'month_var': month_var,
            'doy_var': doy_var,
            'weather_vars': weather_vars,
            'wet_threshold': wet_threshold,
            'wet_extreme_quantile_threshold': wet_extreme_quantile_threshold,
            'window': window,
            'distance_weights': distance_weights,
        }
        self.probs = []
    
    def fit(self, data, project_schedule, tasks_conditions):
        self.data = data
        self.project_schedule = project_schedule
        self.tasks_conditions = tasks_conditions
        self.weather_generator = WeatherGenerator(**self.weather_generator_args)

    def _simulate(self):
        return simulate(model=self)

    def _get_prob(self):
        if len(self.probs):
            prob = self.probs[-1]
            self.probs = self.probs[:-1]
            return prob
        else:
            self.probs = list(np.random.rand(365))
            return self._get_prob()

    def predict(
            self, initial_weather, initial_precipitation, initial_date,
            B=100, parallel=True):
        self.initial_date = initial_date
        self.weather_generator.fit(
            data=self.data,
            initial_weather=initial_weather,
            initial_precipitation=initial_precipitation,
            initial_date=initial_date)
        if parallel:
            durations = Parallel(n_jobs=-1, prefer="processes")(delayed(simulate)(
                model=self) for _ in range(B))
            durations = durations[:B]
        else:
            durations = [self._simulate() for _ in range(B)]
        return durations
