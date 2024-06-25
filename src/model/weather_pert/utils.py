"""Funções auxiliares utilizadas no módulo model."""

# %% libs

import pandas as pd
import numpy as np
import time
import logging as log
import os

# %% functions


def return_calendar(calendar, index_begin, year_simulation): 
    """
    Return calendar list.

    Parameters
    ----------
    calendar : dict
        calendar dict, where keys is years.
    index_begin : int
        index that the impact starts.
    year_simulation : int
        year of simulation.

    Returns
    -------
    results  list
        days where the task can run.
    """
    # select year of simulation
    results = calendar[str(year_simulation)]
    # start of task
    results = results[index_begin:]
    return results


def transform_result(data, key = 'dt_end'):
    """
    Transform monte_carlo_parallel result in dataframe pandas.
    
    - Input Example:
        data = [
            {"key":{"task0": value0, "task1": value1}},
            {"key":{"task0": value2, "task1": value3}}
            ]
    
    Parameters
    ----------
    data : list
        monte_carlo_parallel result
    key : string
        dict key.
    
    Returns
    -------
    results : pandas.core.frame.DataFrame
        dataframe pandas where index is the key parameter.
    """
    # transforming into dataframe
    df = pd.DataFrame([{key: value, task: 1} for i in data for task,value in i[key].items()])
    # select cols
    tasks_name = list(df.columns)
    tasks_name.remove(key) # remove key col
    # create func arg
    tasks_agg = {
        i: 'sum' # sum resuts
        for i in tasks_name
        }
    # group by key
    results  = (
        df
        .groupby(key)
        .agg(tasks_agg)
        .sort_index()
        )
    return results


def processing_result(data, keys = ['dt_begin', 'dt_end', 'duration']):
    """
    Create dict where each value is transform_result function.
    
    - Input Example:
        data = [
            {
                "key":{"task0": value00, "task1": value01},
                "key1":{"task0": value10, "task1": value11}
            },
            {
                "key":{"task0": value02, "task1": value03},
                "key1":{"task0": value12, "task1": value13}
            }
            ]
        
    Parameters
    ----------
    data : list
        monte_carlo_parallel result
    keys : list, optional
        dict keys. The default is ['dt_begin', 'dt_end', 'duration'].

    Returns
    -------
    result : dict
        Each value is dataframe pandas where index is the key parameter.
    """
    # create result variable
    result = {}
    # itering by the keys
    for key in keys:
        # applying transform_result in key
        result[key] = transform_result(
            data = data,
            key = key
            )
    return result

def transform_result(data, key = 'dt_end'):
    """
    Transform monte_carlo_parallel result in dataframe pandas.
    
    - Input Example:
        data = [
            {"key":{"task0": value0, "task1": value1}},
            {"key":{"task0": value2, "task1": value3}}
            ]
    
    Parameters
    ----------
    data : list
        monte_carlo_parallel result
    key : string
        dict key.
    
    Returns
    -------
    results : pandas.core.frame.DataFrame
        dataframe pandas where index is the key parameter.
    """
    # transforming into dataframe
    df = pd.DataFrame([{key: value, task: 1} for i in data for task,value in i[key].items()])
    # select cols
    tasks_name = list(df.columns)
    tasks_name.remove(key) # remove key col
    # create func arg
    tasks_agg = {
        i: 'sum' # sum resuts
        for i in tasks_name
        }
    # group by key
    results  = (
        df
        .groupby(key)
        .agg(tasks_agg)
        .sort_index()
        )
    return results

def processing_result(data, keys = ['dt_begin', 'dt_end', 'duration']):
    """
    Create dict where each value is transform_result function.
    
    - Input Example:
        data = [
            {
                "key":{"task0": value00, "task1": value01},
                "key1":{"task0": value10, "task1": value11}
            },
            {
                "key":{"task0": value02, "task1": value03},
                "key1":{"task0": value12, "task1": value13}
            }
            ]
        
    Parameters
    ----------
    data : list
        monte_carlo_parallel result
    keys : list, optional
        dict keys. The default is ['dt_begin', 'dt_end', 'duration'].

    Returns
    -------
    result : dict
        Each value is dataframe pandas where index is the key parameter.
    """
    # create result variable
    result = {}
    # itering by the keys
    for key in keys:
        # applying transform_result in key
        result[key] = transform_result(
            data = data,
            key = key
            )
    return result




class ResultSimulation:
    """
    save result   
    """
    def add_config(self, tasks_dict):
        """
        Save configs.
        
        Parameters
        ----------
        tasks_dict : dict
            tasks configs.
        """
        # save tasks names      
        self.tasks_name = list(tasks_dict.keys())
        # save tasks period
        self.period = {
            task_name: tasks_dict[task_name]['period'] for task_name in self.tasks_name
            }
        # monitoring of each task
        self.monitoring_names = {
            task: 
                []
                if 'monitoring' not in config['conditions'].keys()
                else 
                [i['name'] for i in config['conditions']['monitoring']]
            for task,config in tasks_dict.items()
            }

    def add_requirements(self, requirements):
        """
        Save requiments.
        """
        self.requiments = requirements
        
    def add_probability(self, data_probability):
        """
        Save tasks probability.
        
        Parameters
        ----------
        data_probability : pandas.core.frame.DataFrame
        """
        self.probability = data_probability
        
    def add_date_finish(self, date_finish):
        """
        Save date_finish for all tasks.
        
        Parameters
        ----------
        date_finish : pandas.core.frame.DataFrame
        """
        self.date_finish = date_finish
            
    def add_date_end(self, df):
        """
        Save date end of tasks.
        
        Parameters
        ----------
        df : pandas.core.frame.DataFrame
        """
        self.date_end = df

    def add_date_begin(self, df):
        """
        Save date begin of tasks.
        
        Parameters
        ----------
        df : pandas.core.frame.DataFrame
        """
        self.date_begin = df

    def add_duration(self, df):
        """
        Save tasks duration.
        
        Parameters
        ----------
        df : pandas.core.frame.DataFrame
        """
        # Salvando a quantidade de vezes em cada cada tarefa foi finalizada em cada dia.
        self.duration = df
        
    def add_uninitiated(self, uninitiated):
        """
        Save uninitiated tasks.
        
        Parameters
        ----------
        df : pandas.core.frame.DataFrame
        """
        # Salvando a quantidade de vezes em cada cada tarefa foi finalizada em cada dia.
        self.uninitiated = uninitiated
        
    def add_unfinished(self, unfinished):
        """
        Save unfinished tasks.
        
        Parameters
        ----------
        df : pandas.core.frame.DataFrame
        """
        # Salvando a quantidade de vezes em cada cada tarefa foi finalizada em cada dia.
        self.unfinished = unfinished
        
    def get_delay_distribution(self, task_name):
        """
        Get delay distribution for delay day.
        
        Parameters
        ----------
        task_name : string
            task_name
        
        Returns
        ---------
        dfdelay : dataframe
        """        
        dfdelay = self.duration[task_name]
        index_name = dfdelay.index.name
        dfdelay = dfdelay.reset_index()
        dfdelay.rename({index_name : 'dtend'}, inplace = True, axis = 1)
        dfdelay['shifts'] = dfdelay['dtend'] - self.tasks_period[task_name]+1
        dfdelay['prob'] = dfdelay[task_name] / dfdelay[task_name].sum()
        dfdelay.sort_values('shifts', inplace = True)
        dfdelay = dfdelay[dfdelay.prob > 0]
        dfdelay.sort_values('shifts', inplace = True)
        dfdelay['cumprob'] = dfdelay.prob.cumsum()
        dfdelay.drop(['dtend', 'prob', task_name], axis = 1, inplace = True)
        return(dfdelay)

    def get_delay_quantile(self, task_name, p, method = 'smaller'):
        """
        p quantil for delay days.
        Parameters
        ----------
        task_name : string
            nome da tarefa
        p : float
            Value between 0 and 1.
        method : string
        """
        dfdelay = self.get_delay_distribution(task_name)
        if method == 'larger':
            return(dfdelay[dfdelay.cumprob >= p].shifts.min())
        elif method == 'smaller':
            return(dfdelay[dfdelay.cumprob <= p].shifts.max())
        else:
            raise ValueError("Method must be either 'smaller' or 'larger'")
        return None
    
    
    def get_violation_freq(self, task_name):
        problems = self.monitoring[task_name].copy()
        cols = self.monitoring_names[task_name]
        depend = [
            col
            for row in problems[cols].to_dict('records')
            for col, value in row.items()
                if value 
            ]
        freq = pd.Series(depend).value_counts() 
        freq = freq / sum(freq)
        return freq
    
    def check_uninitiated(self):
        result  = [
            col 
            for col in self.date_begin.columns 
            if col not in self.date_end.columns
            ]
        return result
    