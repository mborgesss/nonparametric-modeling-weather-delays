from numpy import nan, isnan
from itertools import product


def check_earthworks(mean_air_temp, prcp_amt, **kwargs):
    if isnan(mean_air_temp) or isnan(prcp_amt):
        return nan
    return (mean_air_temp > 0)&(prcp_amt < 10)
def check_concrete(min_temp, max_temp, prcp_amt, max_gust_speed,**kwargs):
    if isnan(min_temp) or isnan(max_temp) or isnan(prcp_amt) or isnan(max_gust_speed):
        return nan
    return (min_temp > 0)&(max_temp < 40)&(prcp_amt < 10)&(max_gust_speed < 30)
def check_formworks(max_gust_speed,**kwargs):
    if isnan(max_gust_speed):
        return nan
    return (max_gust_speed < 30)
def check_steelworks(mean_air_temp, max_temp, prcp_amt, max_gust_speed, **kwargs):
    if isnan(mean_air_temp) or isnan(max_temp) or isnan(prcp_amt) or isnan(max_gust_speed):
        return nan
    return (mean_air_temp > 0)&(max_temp < 40)&(prcp_amt < 30)&(max_gust_speed < 30)
def check_outdoor(mean_air_temp, prcp_amt, max_gust_speed, **kwargs):
    if isnan(mean_air_temp) or isnan(prcp_amt) or isnan(max_gust_speed):
        return nan
    return (mean_air_temp > 0)&(prcp_amt < 1)&(max_gust_speed < 30)
def check_pavements(min_temp, max_temp, prcp_amt, **kwargs):
    if isnan(min_temp) or isnan(max_temp) or isnan(prcp_amt):
        return nan
    return (min_temp > 0)&(max_temp < 40)&(prcp_amt < 1)

tasks_conditions = {
    'earthworks_1': check_earthworks,
    'concrete_1': check_concrete,
    'formworks_1': check_formworks,
    'steelworks_1': check_steelworks,
    'outdoor_1': check_outdoor,
    'pavements_1': check_pavements,
    'earthworks_2': check_earthworks,
    'concrete_2': check_concrete,
    'formworks_2': check_formworks,
    'steelworks_2': check_steelworks,
    'outdoor_2': check_outdoor,
    'pavements_2': check_pavements,
}

project_schedule = {
    'earthworks_1': 15,
    'concrete_1': 15,
    'steelworks_1': 15,
    'formworks_1': 15,
    'pavements_1': 10,
    'outdoor_1': 10,
    'earthworks_2': 15,
    'concrete_2': 15,
    'steelworks_2': 15,
    'formworks_2': 15,
    'pavements_2': 10,
    'outdoor_2': 10,
}

# experiment
seasons_start = {
    'spring': '03-21',
    'summer': '06-21',
    'autumn': '09-21',
    'winter': '12-21'
}
losses = ['density', 'two-tailed', 'left-tail', 'right-tail', 'centered', 'pinball', 'mse']

date_column = 'date'