"""
Para cada estação:
1. Abrir dados climáticos
2. Verificar os dias trabalháveis a partir dos limites climáticos
3. Verificar a data real do projeto para diferentes datas de inicio
4. Se houver pelo menos 75% dos dias (30 dias) salvar os resultdos
"""

########################################################################
# %% libs
########################################################################
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np  
import time
import os
import pickle
import sys
import time
from itertools import product
from numpy import nan

sys.path.append('../..')
from src.model.true import check_duration
from experiments.real.config import *
from tqdm import tqdm
tasks = list(project_schedule.keys())

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import os
import sys
import time
sys.path.append('../..')
from src.model.weather_pert import WeatherPert
from src.model.two_steps.model import KNN, WeatherGenerator
from src.validation.validation import *
from experiments.real.config import * 
import pickle
from experiments.real.config import * 
from tqdm import tqdm

script_path = os.path.dirname(os.path.abspath(__file__))
print(script_path)


months = list(range(1, 13))

date_train_start = datetime.date(2013, 1, 1)
date_train_end = datetime.date(2016, 12, 31)
date_train = [(date_train_start+datetime.timedelta(days=x)).isoformat() for x in range(0, (date_train_end-date_train_start).days+1, 7)]
print('date_train', date_train)

date_test_start = datetime.date(2017, 1, 1)
date_test_end = datetime.date(2020, 12, 31)
date_test = [(date_test_start+datetime.timedelta(days=x)).isoformat() for x in range(0, (date_test_end-date_test_start).days+1, 7)]
print('date_test', date_test)
month_var = 'month'
doy_var = 'doy'


two_steps_params = {
    'precipitation_var': 'prcp_amt',
    'month_var': 'month',
    'doy_var':  'doy',
    'weather_vars':  ['prcp_amt', 'max_temp', 'min_temp', 'mean_air_temp', 'max_gust_speed'],
    'wet_threshold':  .3,
    'wet_extreme_quantile_threshold':  .8,
    'window':  7,
}

weather_pert_hpt = []

for alpha in [0.85, 0.9, 0.95, 1.]:
    for smoothing in [0, 3, 7, 14]:
        if smoothing >= 3:
            mcs = [True, False]
        else:
            mcs = [False]
        for mc in mcs:
            #if not mc:
            for cossine in [True, False]:
                weather_pert_hpt.append({'alpha': alpha, 'smoothing': smoothing, 'mc': mc, 'cossine': cossine})
            #else:
            #    weather_pert_hpt.append({'alpha': alpha, 'smoothing': smoothing, 'mc': mc, 'cossine': False})

def sim_hpt(args, data, date_start, tasks_conditions, project_schedule, B, date):
    data_train = data[data[date_column] < date]
    model = WeatherPert(**args, column_date='date')
    model.fit(
        data=data_train, tasks_conditions=tasks_conditions)
    durations = model.predict(
        B=B, date_start=date_start, project_schedule=project_schedule)
    return {'date': date, 'args': args, 'estimate': durations}


B = 20_000



########################################################################
# %% variaveis
########################################################################
data_dir = os.path.join('..', '..', 'data','cleaned')
results_dir = os.path.join('..', '..', 'data', 'real', 'results_4years')

if not os.path.exists(results_dir): os.mkdir(results_dir)

src_ids = [
    file_name.replace('.csv', '') for file_name in sorted(os.listdir(data_dir))
    if file_name.endswith('.csv') and file_name.replace('.csv', '').isdigit()
    ]
src_ids.sort()
print(f'{len(src_ids)} stations...')



def simulate_KNN(date, data):
    data_train = data[data[date_column] < date]
    last_day = (
        data_train
        .sort_values(date_column, ascending=False)
        .to_dict('records')[0])
    initial_date = datetime.date.fromisoformat(date)
    initial_precipitation = last_day[two_steps_params['precipitation_var']]
    initial_weather = {var: last_day[var] for var in two_steps_params['weather_vars']}
    model = KNN(
        precipitation_var=two_steps_params['precipitation_var'], month_var=month_var,
        doy_var=doy_var, weather_vars=two_steps_params['weather_vars'])
    model.fit(
        data=data_train,
        tasks_conditions=tasks_conditions)
    durations = model.predict(
        initial_weather=initial_weather, project_schedule=project_schedule,
        initial_precipitation=initial_precipitation,
        initial_date=initial_date,
        B=B)
    return durations

def simulate_PERT(date, data, weather_pert_params):
    data_train = data[data[date_column] < date]
    model = WeatherPert(**weather_pert_params, column_date=date_column)
    model.fit(
        data=data_train,
        tasks_conditions=tasks_conditions)
    durations = model.predict(
        B=B, date_start=date, project_schedule=project_schedule,)
    return durations


def parallel_observed(data, date):
    try:
        return check_duration(
            data=data, date_start=date,
            tasks_conditions=tasks_conditions, project_schedule=project_schedule)
    except Exception as e:
        print(date, e)
        return 0




import time

if os.path.exists(os.path.join(results_dir, 'exclude_src_id.pickle')):
    with open(os.path.join(results_dir, 'exclude_src_id.pickle'), 'rb') as file:
        exclude_src_id = pickle.load(file)
else:
    exclude_src_id = []
print(exclude_src_id)

########################################################################
# %% executando
########################################################################
for src_id in tqdm(src_ids):
    #if int(src_id) not in [409, 440, 556, 643, 726, 9]:#[150, 212, 386, 409, 440, 556, 643, 726, 9, 1319, 17314]:
    #    continue

    src_id = int(src_id)
    print('src_id', src_id)
    if src_id in exclude_src_id:
        continue
    src_dir = os.path.join(results_dir, str(src_id))
    # 1. Abrir dados climáticos
    data = pd.read_csv(os.path.join(data_dir, f"{src_id}.csv")).fillna(nan).replace([None], [nan])
    data[month_var] = data[date_column].apply(lambda v: int(v[5:7]))
    data[doy_var] = data[date_column].apply(lambda v: datetime.date.fromisoformat(v).timetuple().tm_yday)
    
    if not os.path.exists(os.path.join(src_dir, 'observed.pickle')):
        #==========================================================================
        # Verificando durações na datas de treino e teste
        #==========================================================================
        observed = {}
        dates = date_test + date_train
        durations = (
            Parallel(n_jobs=-1, prefer="processes")
            (delayed(parallel_observed)(
                data=data, date=date)
            for date in dates
            ))
        observed = {date: duration for date, duration in zip(dates, durations) if duration > 0}
        # 4. Se houver pelo menos 75% dos dias (30 dias) salvar os resultdos
        observed_test = {k: v for k, v in observed.items() if k in date_test}
        observed_train = {k: v for k, v in observed.items() if k in date_train}
        print('observed_test', len(observed_test))
        print('observed_train', len(observed_test))
        if len(observed_test.keys()) < .7*len(date_test) or len(observed_train.keys()) < .5*len(date_train):
            exclude_src_id.append(src_id)
            with open(os.path.join(results_dir, 'exclude_src_id.pickle'), 'wb') as file:
                pickle.dump(exclude_src_id, file, protocol=pickle.HIGHEST_PROTOCOL)
            continue
        if not os.path.exists(src_dir): os.mkdir(src_dir)
        with open(os.path.join(src_dir, 'observed.pickle'), 'wb') as file:
            pickle.dump(observed, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(src_dir, 'observed.pickle'), 'rb') as file:
            observed = pickle.load(file)
        observed_test = {k: v for k, v in observed.items() if k in date_test}
        observed_train = {k: v for k, v in observed.items() if k in date_train}

            
    #==========================================================================
    # Verificando Weather PERT HPT
    #==========================================================================
    
    #if os.path.exists(os.path.join(src_dir, 'weather_pert_hpt_exps.pickle')):
    #    os.remove(os.path.join(src_dir, 'weather_pert_hpt_exps.pickle'))
    #for src_id in tqdm(src_ids):
    #
    #    with open(os.path.join(results_dir, src_id, 'weather_pert_hpt_exps_old.pickle'), 'rb') as file:
    #        weather_pert_hpt_exps = pickle.load(file)
    #    new_weather_pert_hpt_exps = []
    #    for i, exp in enumerate(weather_pert_hpt_exps):
    #        exp = deepcopy(exp)
    #        args = exp['args']
    #        smoothing = args['smoothing']
    #        if smoothing == 'cossine':
    #            args['smoothing'] = 0
    #            args['cossine'] = True
    #        else:
    #            args['cossine'] = False
    #        exp['args'] = args
    #        new_weather_pert_hpt_exps.append(exp)
    #    with open(os.path.join(results_dir, src_id, 'weather_pert_hpt_exps.pickle'), 'wb') as file:
    #        pickle.dump(new_weather_pert_hpt_exps, file, protocol=pickle.HIGHEST_PROTOCOL)
    #elif os.path.exists(os.path.join(src_dir, 'weather_pert_hpt_exps_backup.pickle')):
    #    with open(os.path.join(src_dir, 'weather_pert_hpt_exps_backup.pickle'), 'rb') as file:
    #        weather_pert_hpt_exps_backup = pickle.load(file)
    #    #with open(os.path.join(src_dir, 'weather_pert_hpt_exps.pickle'), 'rb') as file:
    #    #    weather_pert_hpt_exps = pickle.load(file)
    #    #retry = len(weather_pert_hpt_exps) == len(weather_pert_hpt_exps_backup)
    #    #if not retry:
    #    #    weather_pert_hpt_exps = weather_pert_hpt_exps_backup
    #    retry = True
    #else:
    #    weather_pert_hpt_exps_backup = []
    

    if not os.path.exists(
            os.path.join(src_dir, 'weather_pert_hpt_exps.pickle')):
        start = time.time()
        jobs = ((date, args) for date in observed_train for args in weather_pert_hpt)
        
        simulations = (
            Parallel(n_jobs=-1, prefer="processes")
            (delayed(sim_hpt)(
                args=args, data=data, date_start=date, B=B,
                tasks_conditions=tasks_conditions,
                project_schedule=project_schedule,
                date=date
                )
            for date, args in tqdm(jobs)
            ))

        simulations_metrics = (
            Parallel(n_jobs=-1, prefer="processes")
            (delayed(calculate_metrics)(observed=observed_train[simulation['date']], estimate=simulation['estimate'])
            for simulation in simulations
            ))
        print('HPT TIME ', round(start-time.time(), 2))
    

        weather_pert_hpt_exps = []
        for simulation, metrics in zip(simulations, simulations_metrics):
            estimate = simulation['estimate']
            date = simulation['date']
            args = simulation['args']
            real_duration  = observed_train[date]

            metrics['date'] = date
            metrics['observed'] = real_duration
            metrics['estimate'] = estimate
            metrics['args'] = args
            weather_pert_hpt_exps.append(metrics)

        with open(os.path.join(src_dir, 'weather_pert_hpt_exps.pickle'), 'wb') as file:
            pickle.dump(weather_pert_hpt_exps, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(src_dir, 'weather_pert_hpt_exps.pickle'), 'rb') as file:
            weather_pert_hpt_exps = pickle.load(file)
    
    weather_pert_hpt_best = {}
    for loss_type in losses:
        weather_pert_hpt_best[loss_type] = sorted(
            weather_pert_hpt, key=lambda args: sum([
                exp[loss_type]
                for exp in weather_pert_hpt_exps
                if exp['args'] == args
            ]))[0]
    with open(os.path.join(src_dir, 'weather_pert_hpt_best.pickle'), 'wb') as file:
        pickle.dump(weather_pert_hpt_best, file, protocol=pickle.HIGHEST_PROTOCOL)
    loss_type = 'right-tail'#'pinball'
    weather_pert_params = weather_pert_hpt_best[loss_type]
    print('weather_pert_params', weather_pert_params)

    #==========================================================================
    # Verificando Estimativas
    #==========================================================================
    if os.path.exists(os.path.join(src_dir, 'weather_pert_estimate.pickle')):
        os.remove(os.path.join(src_dir, 'weather_pert_estimate.pickle'))
    if os.path.exists(os.path.join(src_dir, 'metrics_test.pickle')):
        os.remove(os.path.join(src_dir, 'metrics_test.pickle'))

    if (
            not os.path.exists(os.path.join(src_dir, 'two_steps_estimate.pickle')) or
            not os.path.exists(os.path.join(src_dir, 'weather_pert_estimate.pickle')) or
            not os.path.exists(os.path.join(src_dir, 'metrics_testpickle'))
            ):
        
        print('iniciando teste...')
        simulations_test = {'two_steps': {}, 'weather_pert': {}}
        metrics_test = []
        
        
        if not os.path.exists(os.path.join(src_dir, 'two_steps_estimate.pickle')):
            start = time.time()
            two_steps = Parallel(n_jobs=-1, prefer="processes")(delayed(simulate_KNN)(date=date, data=data) for date in observed_test)
            print(f'simulate_KNN exec time: {round(time.time()-start, 2)} s')
            for i, (date, real_duration) in  tqdm(enumerate(observed_test.items())):
                simulations_test['two_steps'][date] = two_steps[i]
            with open(os.path.join(src_dir, 'two_steps_estimate.pickle'), 'wb') as file:
                pickle.dump(simulations_test['two_steps'], file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(src_dir, 'two_steps_estimate.pickle'), 'rb') as file:
                simulations_test['two_steps'] = pickle.load(file)

        if not os.path.exists(os.path.join(src_dir, 'weather_pert_estimate.pickle')):
            start = time.time()
            weather_pert = Parallel(n_jobs=-1, prefer="processes")(delayed(simulate_PERT)(date=date, data=data, weather_pert_params=weather_pert_params) for date in observed_test)
            print(f'simulate_PERT exec time: {round(time.time()-start, 2)} s')
            for i, (date, real_duration) in  tqdm(enumerate(observed_test.items())):
                simulations_test['weather_pert'][date] = weather_pert[i]
            with open(os.path.join(src_dir, 'weather_pert_estimate.pickle'), 'wb') as file:
                pickle.dump(simulations_test['weather_pert'], file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(src_dir, 'weather_pert_estimate.pickle'), 'rb') as file:
                simulations_test['weather_pert'] = pickle.load(file)
        
        
        two_steps_metrics = (
            Parallel(n_jobs=-1, prefer="processes")
            (delayed(calculate_metrics)(observed=real_duration, estimate=simulations_test['two_steps'][date])
            for i, (date, real_duration) in enumerate(observed_test.items())
            ))
            
        weather_pert_metrics = (
            Parallel(n_jobs=-1, prefer="processes")
            (delayed(calculate_metrics)(observed=real_duration, estimate=simulations_test['weather_pert'][date])
            for i, (date, real_duration) in enumerate(observed_test.items())
            ))
            


        for i, (date, real_duration) in  tqdm(enumerate(observed_test.items())):
            #==========================================================================
            # Verificando two_steps
            #==========================================================================
            metrics = two_steps_metrics[i]
            metrics['date'] = date
            metrics['model'] = 'two_steps'
            metrics_test.append(metrics)
            #==========================================================================
            # Verificando PERT
            #==========================================================================
            start = time.time()
            metrics = weather_pert_metrics[i]
            metrics['date'] = date
            metrics['model'] = 'weather_pert'
            metrics_test.append(metrics)

        with open(os.path.join(src_dir, 'metrics_test.pickle'), 'wb') as file:
            pickle.dump(metrics_test, file, protocol=pickle.HIGHEST_PROTOCOL)
