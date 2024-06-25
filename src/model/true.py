"""
Modelos implementados.

- Monte Carlo
- Real
"""

import numpy as np

import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('..')
from .utils import *
import datetime

def check_duration(data, date_start, project_schedule, tasks_conditions=None, replace_na=False):
    """
    True duration task.

    Parameters
    ----------
    p_sucess : numpy.ndarray
        probability of success.
    index_begin : int
        index that the impact starts.
    period : int
        task duration.
    n_shifts : int
        number of shifts
    task_name : string, optional
        task name. The default is ''.
    calendar : dict
        calendar dict, where keys is years.The default is [].
    impact : list, optional
        Each element is an impact. The default is [].

    Returns
    -------
    number_days : int
        task duration.
    """
    date_start = datetime.datetime.strptime(date_start, "%Y-%m-%d").date()
    data = data.loc[lambda df: df.date >= date_start.isoformat()]

    tasks = list(project_schedule.keys())

    if tasks_conditions:
        data_tasks = data[['date']]

        for task, condition_func in tasks_conditions.items():
            data_tasks[task] = data.apply(lambda row: condition_func(**row)*1, axis=1)
    else:
        data_tasks = data

    if data_tasks[tasks].mean().isna().any():
        raise ValueError('NA')
    
    data = data_tasks.sort_values('date')


    total = 0
    # while exist next_tasks
    for task  in list(project_schedule.keys()):
        result = data.loc[lambda df: df.date >= date_start.isoformat(), task]
        if replace_na:
            result = result.fillna(1)
        dates = data.loc[lambda df: df.date >= date_start.isoformat(), 'date']
        cum_sucess = np.cumsum(result)
        # selecting index where value is bigger than success
        days_sucess = np.where(cum_sucess >= project_schedule[task])[0]
        # update
        if len(days_sucess) > 0:  # if finished
            duration = int(days_sucess[0]+ 1)
            if (not replace_na) and result[:duration].isna().any():
                raise ValueError(f"NA in serie! {list(dates[:duration][result[:duration].isna()].values)}",)
        else:
            raise ValueError('End data')
        total += duration
        date_start += datetime.timedelta(days = duration)
    return total


def check_task_duration(data, tasks_conditions, date_start, project_schedule, replace_na=False):
    date_start = datetime.datetime.strptime(date_start, "%Y-%m-%d").date()
    data = data.loc[lambda df: df.date >= date_start.isoformat()]

    data_tasks = data[['date']]
    tasks = list(project_schedule.keys())

    for task, condition_func in tasks_conditions.items():
        data_tasks[task] = data.apply(lambda row: condition_func(**row)*1, axis=1)

    if data_tasks[tasks].mean().isna().any():
        raise ValueError('NA')
    
    data = data_tasks.sort_values('date')

    total = 0
    # while exist next_tasks
    task_duration = {}
    for task  in list(project_schedule.keys()):
        result = data.loc[lambda df: df.date >= date_start.isoformat(), task]
        if replace_na:
            result = result.fillna(1)
        dates = data.loc[lambda df: df.date >= date_start.isoformat(), 'date']
        cum_sucess = np.cumsum(result)
        # selecting index where value is bigger than success
        days_sucess = np.where(cum_sucess >= project_schedule[task])[0]
        # update
        if len(days_sucess) > 0:  # if finished
            duration = int(days_sucess[0]+ 1)
            if (not replace_na) and result[:duration].isna().any():
                raise ValueError(f"NA in serie! {list(dates[:duration][result[:duration].isna()].values)}",)
        else:
            raise ValueError('End data')
        task_duration[task] = duration
        total += duration
        date_start += datetime.timedelta(days = duration)
    return total, task_duration
