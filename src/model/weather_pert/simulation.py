# %% libs
import numpy as nps
import datetime
from .utils import *

# %% Monte Carlo
def _get_duration(probs, T):
    """simulação de monte carlo."""
    total = 0
    for i, p in enumerate(probs):
        total += p
        if total == T:
            break
    # update
    return i+1

def monte_carlo(B, data, date_start, project_schedule):
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
    try:
        date_start = datetime.datetime.strptime(date_start, "%Y-%m-%d").date()
        day_of_year = int(date_start.strftime('%j')) -1
        year = int(date_start.year)
    except:
        date_start = datetime.datetime.strptime(date_start, "%m-%d").date()
        day_of_year = int(date_start.strftime('%j')) -1
        year = -1
    #crc = data.to_dict('list')
    crc = data
    #print(round(time.time()-start, 2))
    #start = time.time()
    p = []
    if year == -1:
        for task in project_schedule:
            p.extend(crc[task] + crc[task] + crc[task] + crc[task])
            days = 365*4
    elif year % 4 == 0:
        for task in project_schedule:
            leap_p =  list(np.delete(crc[task], 59))
            p.extend(crc[task] + leap_p + leap_p + leap_p)
            days = 1461
    elif year % 4 == 1:
        for task in project_schedule:
            leap_p =  list(np.delete(crc[task], 59))
            p.extend(leap_p + leap_p + leap_p + crc[task])
            days = 1461
    elif year % 4 == 2:
        for task in project_schedule:
            leap_p =  list(np.delete(crc[task], 59))
            p.extend(leap_p + leap_p + crc[task]+ leap_p)
            days = 1461
    elif year % 4 == 3:
        for task in project_schedule:
            leap_p =  list(np.delete(crc[task], 59))
            p.extend(leap_p+ crc[task] + leap_p + leap_p)
            days = 1461
    p = np.random.binomial(n=1, p=p, size=(B, int(days*len(project_schedule)))).reshape(B, len(project_schedule), days)
    durations = [day_of_year for i in range(B)]
    for (j, task)  in enumerate(list(project_schedule.keys())):
        durations = [d + _get_duration(p[b][j][d:], project_schedule[task]) for b, d in enumerate(durations)]
    durations = [d-day_of_year for d in durations]
    return durations
