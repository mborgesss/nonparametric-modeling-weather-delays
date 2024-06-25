from .weather_generator import WeatherGenerator
from joblib import Parallel, delayed 
import numpy as np
from copy import deepcopy

def simulate(date, state, weather, generate_weather, project_schedule, tasks_conditions):
    initial_date = deepcopy(date)
    for task, duration in project_schedule.items():
        while duration > 0:
            (date, state, weather) = generate_weather(date, state, weather)
            is_work_day = tasks_conditions[task](**weather)
            duration -= is_work_day
    project_duration = (date - initial_date).days
    return project_duration


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
    
    def fit(self, data, project_schedule, tasks_conditions):
        self.data = data
        self.project_schedule = project_schedule
        self.tasks_conditions = tasks_conditions
        self.weather_generator = WeatherGenerator(**self.weather_generator_args)

    def _simulate(self):
        return simulate(model=self)


    def predict(
            self, initial_weather, initial_precipitation, initial_date,
            B=100, njobs=2, threads=4):
        self.initial_date = initial_date
        ((date, state, weather), generate_weather) = self.weather_generator.fit(
            data=self.data,
            initial_weather=initial_weather,
            initial_precipitation=initial_precipitation,
            initial_date=initial_date)
        if njobs and threads:
            durations = Parallel(n_jobs=-1, prefer="threads")(delayed(simulate)(
                date=date, state=state, weather=weather,
                generate_weather=generate_weather,
                project_schedule=self.project_schedule,
                tasks_conditions=self.tasks_conditions
                ) for _ in range(B))
            durations = durations[:B]
        else:
            durations = [self._simulate() for _ in range(B)]
        return durations
