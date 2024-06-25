import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

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

    def _get_mc_state_threshold(self):
        precipitations = self.data[self.precipitation_var].values
        months = self.data[self.month_var].values
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

    def _assign_mc_states(self):
        states = []
        for row in self.data[[self.precipitation_var, self.month_var]].to_dict('records'):
            precipitation = row[self.precipitation_var]
            month = row[self.month_var]
            state = self._assign_mc_state(precipitation=precipitation, month=month)
            states.append(state)
        
        self.data['state'] = states

    def _fit_mc_transitions(self):
        states = self.data['state']
        next_states = states.shift(-1)
        months = self.data[self.month_var]
        
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


    def _mc_simulate(self, month, state):
        transition = self.month_transitions[month]
        state_idx = self.mc_states.index(state)
        probs = transition[state_idx]
        state = np.random.choice(self.mc_states, size=1, p=probs)[0]
        return state


    def _simulate_knn_weather(self, weather, state, next_doy, next_state):
        range_size = self.window
        filter = [False]
        # iterando até ter dias selecionados
        while not max(filter):
            range_size += 1
            doy_range = np.arange(next_doy - range_size, next_doy + range_size + 1)
            doy_range[doy_range < 0] += 365
            # buscando vizinhos
            filter = (
                self.data[self.doy_var].isin(doy_range) &
                (self.data['state'] == state) &
                (self.data['state'].shift(-1) == next_state)) # next day
        neighbors = self.data[filter].to_dict('records')
        ## calculando numero de vizinhos mais proximos
        q = len(neighbors)
        k = int(np.ceil(np.sqrt(q)))
        ## calculando distancias euclidiana
        distances = [
            np.sqrt(
                sum(self.distance_weights[var]*((weather[var] -  neighbor[var])**2)
                for var in self.weather_vars))
            for neighbor in neighbors]
        ## selecionando k vizinhos mais proximos
        idx_order = sorted(range(len(distances)), key=lambda i: distances[i])
        k_neighbors = [neighbors[i] for i in idx_order][:k]
        ## sorteando vizinho
        weights = [1/i for i in range(1, k + 1)]
        probs = [w/sum(weights) for w in weights]
        selected_neighbor = np.random.choice(k_neighbors, size=1, p=probs)[0]
        ## selecionando clima do proximo dia
        successive_day_index = selected_neighbor['_index'] + 1
        successive_day = self.data[self.data['_index'] == successive_day_index].to_dict('records')[0]
        new_weather = {var: successive_day[var] for var in self.weather_vars}
        return new_weather


    def simulate_weather(self):
        ## verificando informações do dia atual
        month = self.date.month
        state=self.state
        ## verificando informações do próximo dia
        nex_date = self.date + datetime.timedelta(days=1)
        next_doy = nex_date.timetuple().tm_yday
        ## simulando estado da precipitação
        new_state = self._mc_simulate(month=month, state=state)
        ## selecionando proximo clima
        new_weather = self._simulate_knn_weather(
            weather=self.weather, state=state,
            next_doy=next_doy, next_state=new_state)
        ## atulizando estados da cadeia
        self.date = nex_date
        self.state = new_state
        self.weather = new_weather
        return new_weather


    def fit(self, data, initial_precipitation, initial_weather, initial_date):
        # removendo linhas com algum valor ausente
        self.data = (data[
            ~data[self.weather_vars + [self.precipitation_var, self.month_var, self.doy_var]]
            .isna().any(axis=1)
            ])
        self.data['_index'] = range(len(self.data))
        # calculando pesos
        self._get_mc_state_threshold()
        self._assign_mc_states()
        self._fit_mc_transitions()
        # salvando estados iniciais
        self.date = initial_date
        self.state = self._assign_mc_state(
            precipitation=initial_precipitation, month=initial_date.month)
        self.weather = initial_weather