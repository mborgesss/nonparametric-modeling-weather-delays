import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np  
import os
import pickle
import sys
from numpy import nan
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
from joblib import Parallel, delayed 
import numpy as np
from copy import deepcopy
import time



class WeatherGenerator:

    def __init__(
            self,
            precipitation_var, month_var, doy_var, weather_vars,
            wet_threshold=.3, wet_extreme_quantile_threshold=.8, window=7,
            distance_weights=None):
        self.mc_states = ['dry', 'wet', 'extreme']
        self.wet_threshold = wet_threshold
        self.wet_extreme_threshold = None
        self.wet_extreme_quantile_threshold = wet_extreme_quantile_threshold
        self.window = window
        self.precipitation_var = precipitation_var
        self.month_var = month_var
        self.doy_var = doy_var
        self.weather_vars = weather_vars
        if distance_weights and isinstance(distance_weights, dict):
            self.distance_weights = {var: distance_weights.get(var, 0) for var in weather_vars}
        else:
            self.distance_weights = {var: 1 for var in weather_vars}
        self.month_thresholds = {}
        self.data = None

    #==========================================================================
    # Markov
    #==========================================================================
    def create_state_generator(self, data):
        months = data[self.month_var].values
        precipitations = data[self.precipitation_var].values
        # calculando pesos
        self._get_mc_state_threshold(months=months, precipitations=precipitations)
        states = self._assign_mc_states(months=months, precipitations=precipitations)
        self._fit_mc_transitions(months=months, states=states)
        data['state'] = states
        # criando gerador
        month_transitions = self.month_transitions
        mc_states = self.mc_states
        def state_generator(month, state, prob=None):
            return map_prob(cats=mc_states, probs=month_transitions[month][mc_states.index(state)], value=prob)
        self.state_generator = state_generator
        return data

    def _get_mc_state_threshold(self, months, precipitations):
        self.month_thresholds = {}

        for month in range(1, 13):
            wet_threshold = self.wet_threshold
            month_precipitations = precipitations[months == month]
            if len(month_precipitations):
                month_precipitations = month_precipitations[month_precipitations > wet_threshold]
                if len(month_precipitations):
                    wet_extreme_threshold = np.quantile(month_precipitations, self.wet_extreme_quantile_threshold)
                else:
                    wet_extreme_threshold = None
            else:
                wet_threshold = None
                wet_extreme_threshold = None
            self.month_thresholds[month] = {'wet': wet_threshold, 'wet_extreme': wet_extreme_threshold}

    def _assign_mc_state(self, precipitation, month):
        thresholds = self.month_thresholds[month]
        if precipitation < thresholds['wet']:
            return 'dry'
        elif precipitation < thresholds['wet_extreme']:
            return 'wet'
        else:
            return 'extreme'

    def _assign_mc_states(self, precipitations, months):
        states = []
        for month, precipitation in zip(months, precipitations):
            state = self._assign_mc_state(precipitation=precipitation, month=month)
            states.append(state)
        return np.array(states)

    def _fit_mc_transitions(self, states, months):
        next_states = np.roll(states, -1)
        next_states[-1] = np.nan
        
        n_states = len(self.mc_states)
        states = np.array(states, dtype=str)
        
        self.month_transitions = {}

        for month in range(1, 13):
            month_state = states[months == month]
            month_next_states = next_states[months == month]

            transition = np.zeros((n_states, n_states))
            for i, state in enumerate(self.mc_states):
                for j, next_state in enumerate(self.mc_states):
                    transition[i, j] = ((month_state == state)&(month_next_states == next_state)).sum()

            transition /= np.sum(transition, axis=1, keepdims=True)
            transition[np.isnan(transition)] = 0
            self.month_transitions[month] = transition

    def simulate_state(self, month, state, prob=None):
        transition = self.month_transitions[month]
        state_idx = self.mc_states.index(state)
        probs = transition[state_idx]
        state = map_prob(cats=self.mc_states, probs=probs, value=prob)
        return state
    

    #==========================================================================
    # KNN
    #==========================================================================
    def _distance(self, weather, neighbor):
        return sum(
            self.distance_weights[var]*((weather[var] -  neighbor[var])**2)
            for var in self.weather_vars)
    
    
    def create_weather_generator(self, data):
        # salvando informações de clima
        _index_weather = data[self.weather_vars].values
        # criando cache de index proximos
        doys = data[self.doy_var].values
        states = data['state'].values
        next_states =  data['state'].shift(-1).values
        indexes = data['_index'].values
        _cache_neighbor_indexes = {
            state: {next_state: {} for next_state in self.mc_states}
            for state in self.mc_states}
        doy_range = np.array(list(range(1, 366))*3)
        for state in self.mc_states:
            for next_state in self.mc_states:
                _filter = (states == state) & (next_states == next_state)
                _doys = doys[_filter]
                for next_doy in range(1, 367):
                    # set initial values
                    range_size = self.window
                    filter = [False]
                    # iterando até ter dias selecionados
                    while not max(filter):
                        # buscando vizinhos
                        filter = np.in1d(_doys, doy_range[(365 - 1 + next_doy - range_size):(365 + next_doy + range_size)]) # next day
                        range_size += 1
                    _cache_neighbor_indexes[state][next_state][next_doy] = indexes[_filter][filter]
        #_distance = self._distance
        def weather_generator(weather, state, next_doy, next_state, prob=None):
            neighbor_indexes = _cache_neighbor_indexes[state][next_state][next_doy]
            ## calculando numero de vizinhos mais proximos
            q = len(neighbor_indexes)
            k = int(np.ceil(np.sqrt(q)))
            ## calculando distancias euclidiana
            ## selecionando k vizinhos mais proximos
            distances = np.sum((_index_weather[neighbor_indexes] - weather)**2, axis=1)
            k_neighbors = [neighbor_indexes[i] for i in sorted(range(q), key=lambda i: distances[i])][:k]
            ## sorteando vizinho
            weights = [1/i for i in range(1, k + 1)]
            probs = [w/sum(weights) for w in weights]
            selected_neighbor = map_prob(cats=k_neighbors, probs=probs, value=prob)
            ## selecionando clima do proximo dia
            successive_day_index = selected_neighbor + 1
            new_weather = _index_weather[successive_day_index]
            return new_weather
        self.weather_generator = weather_generator


    def simulate_weather(self, weather, state, next_doy, next_state, prob=None):
        self.weather_generator(
            weather=weather, state=state, next_doy=next_doy,
            next_state=next_state, prob=prob)


    def fit(self, data):
        # removendo linhas com algum valor ausente
        data = (data[
            ~data[self.weather_vars + [self.precipitation_var, self.month_var, self.doy_var]]
            .isna().any(axis=1)
            ])
        data['_index'] = range(len(data))
        # calculando pesos
        data = self.create_state_generator(data=data)
        # criando cache
        self.create_weather_generator(data=data)
        start = time.time()
        # salvando estados iniciais

    def get_weather_generator(self):
        return self.weather_generator


    def get_state_generator(self):
        return self.state_generator

    def check_state(self, precipitation, month):
        return self._assign_mc_state(
            precipitation=precipitation, month=month)


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
    
    def fit(self, data, tasks_conditions):
        self.data = data
        self.tasks_conditions = tasks_conditions
        self._weather_generator = WeatherGenerator(**self.weather_generator_args)
        self._weather_generator.fit(data=data)
        self.state_generator = self._weather_generator.get_state_generator()
        self.weather_generator = self._weather_generator.get_weather_generator()

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
            project_schedule, B=100):
        self.initial_date = initial_date
        self.initial_weather = initial_weather
        self.initial_precipitation = initial_precipitation
        self.initial_state = self._weather_generator.check_state(
            precipitation=self.initial_precipitation, month=initial_date.month)

        initial_state = self.initial_state
        initial_weather = np.array(list(self.initial_weather.values()))
        initial_date = self.initial_date
        tasks_conditions = self.tasks_conditions
        weather_generator = self.weather_generator
        state_generator = self.state_generator
        weather_vars = self.weather_generator_args['weather_vars']

        total_duration = sum(project_schedule.values())
        n_days =  total_duration*5
        probs = np.random.random(size=(B, n_days, 2))

        def simulator(probs):
            nonlocal n_days
            weather = initial_weather
            state = initial_state
            date = initial_date
            i = 0
            for task, duration in project_schedule.items():
                condition = tasks_conditions[task]
                while duration > 0:
                    month = date.month
                    next_date = date + datetime.timedelta(days=1)
                    next_doy = next_date.timetuple().tm_yday
                    new_state = state_generator(month=month, state=state, prob=probs[i][0])
                    new_weather = weather_generator(weather=weather, state=state, next_doy=next_doy, next_state=new_state, prob=probs[i][1])
                    is_work_day = condition(**{var: new_weather[i] for i, var in enumerate(weather_vars)})
                    duration -= is_work_day
                    i += 1
                    state = new_state
                    weather = new_weather
                    date = next_date
                    if i == n_days:
                        n_days = total_duration
                        probs = np.random.random(size=(n_days, 2))
                        i = 0
            project_duration = (date - initial_date).days
            return project_duration

        durations = [simulator(probs=probs[b]) for b in range(B)]
        return durations
