"""Verifica a condição para cada tarefa."""

# %% libs

import pandas as pd
import re
import numpy as np

# %% Auxiliary Functions
def _rule_correction(rule, cols):
    """
    Corret rules.

    Parameters
    ----------
    rule : string
    cols : list
        list of columns.

    Returns
    -------
    rule : string
    """
    rule = rule.replace(" ", "")
    rule = '('+rule+')'

    common_r = ["\)", "\.", "\,"]
    common_l = ["\(", "\)", "\,"]
    regex_r = "(("+')|('.join(common_r)+"))"
    regex_l = "(("+')|('.join(common_l)+"))"


    for col in sorted(cols, key=len, reverse=True):
        while True:
            try:
                find = re.search(f"{regex_l}({col}){regex_r}", rule).group()
                if col in find:
                    replace = find.replace(f"{col}", f"  df['{col}']  ")
                    rule = rule.replace(find, replace)
                else:
                    break
            except AttributeError:
                break
    rule = rule.replace(" ", "")
    return rule


def _find_cols(rule, cols):
    """
    Find used cols in rule.

    Parameters
    ----------
    rule : string
    cols : list
        list of columns.

    Returns
    -------
    rule : string
    """
    rule = rule.replace(" ", "")
    rule = '('+rule+')'

    common_r = ["\)", "\.", "\,"]
    common_l = ["\(", "\)", "\,"]
    regex_r = "(("+')|('.join(common_r)+"))"
    regex_l = "(("+')|('.join(common_l)+"))"

    result = []
    for col in sorted(cols, key=len, reverse=True):
        while True:
            try:
                find = re.search(f"{regex_l}({col}){regex_r}", rule).group()
                if col in find:
                    result.append(col)
                    replace = find.replace(f"{col}", f"  df['{col}']  ")
                    rule = rule.replace(find, replace)
                else:
                    break
            except AttributeError:
                break
    result = list(set(result))
    return result

class TasksVerification:
    """Apply data manipulations for calculation daily conclusion tasks."""

    def __init__(self, data, column_date='date'):
        """
        Save data and configs.

        Parameters
        ----------
        data : pandas.core.frame.DataFrame
            data, aggregate by date and shifts.
        column_date : str, optional
            name of date column. The default is 'date'.tasks_dict
        """
        self.data = data
        self.column_date = column_date
        # number of shifts
        #self.n_shifts =  len(self.data['shift'].unique())
        # number of digits
        #self.n_digits = len(str(self.n_shifts))
        #transform shifts
        #self.data['shift'] = self.data['shift'].apply(lambda x : str(x).zfill(
        #    self.n_digits))

    def rule_verification(self, rule):
        """Apply rule verification."""
        # variable using in eval function
        df = self.data.copy()
        cols = df.columns
        used_cols = _find_cols(rule=rule, cols=cols)
        sucess = _rule_correction(rule=rule, cols=cols)
        return np.where(df[used_cols].isna().any(axis=1), np.nan, eval(sucess)*1)

    def verification(self, task_cond):
        """
        Check requirements.

        Parameters
        ----------
        task_cond : dict
            tasks config.
        """
        pd.options.mode.chained_assignment = None
        result = self.data[[self.column_date]]
        # itering by task
        for task_name, conditions in task_cond.items():
            # check sucess
            result[task_name] = self.rule_verification(conditions)
        return result