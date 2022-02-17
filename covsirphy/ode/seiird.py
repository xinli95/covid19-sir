# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 21:21:07 2022

@author: bitsi
"""
import numpy as np
import pandas as pd
from covsirphy.util.error import deprecate
from covsirphy.ode.mbase import ModelBase

class SEIIRD(ModelBase):
    """
    SEIIRD model.
    """
    # Model name
    NAME = "SEIIRD"
    # names of parameters
    PARAMETERS = ["beta_1", "beta_2", "alpha", "gamma", "lamda"]
    DAY_PARAMETERS = [
        "1/beta_1 [day]", "1/beta_2 [day]", "1/alpha [day]", "1/gamma [day]", "1/lamda [day]"
    ]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "S": "Susceptible",
        "E": "Exposed",
        "I": "Infectious",
        "Iso": "Isolated",
        "R": "Recovered",
        "D": "Dead"
    }
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array([0, 1, 1, 1, 1, 1])
    # Variables that increases monotonically
    VARS_INCLEASE = ["Recovered", "Dead"]
    DSIFR_COLUMNS = ['Date', 'Susceptible', 'Exposed', "Infectious", "Isolated", "Recovered", "Dead"]
    # Example set of parameters and initial values
    EXAMPLE = {
        ModelBase.STEP_N: 180,
        ModelBase.N.lower(): 1_000_000,
        ModelBase.PARAM_DICT: {
            "beta_1": 0.2, "beta_2": 1/4.5, "alpha": 0.005, "gamma": 0.075,
            "lamda": 0.005, "delta":0.2
        },
        ModelBase.Y0_DICT: {
            "Susceptible": 999_000, "Exposed": 10, "Infectious": 10, "Isolated": 0,
            "Recovered": 0, "Dead": 0
            
        },
    }

    def __init__(self, population, beta_1, beta_2, alpha, gamma, lamda):
        # Total population
        self.population = self._ensure_population(population)
        # Non-dim parameters
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.non_param_dict = {
            "beta_1": beta_1, "beta_2": beta_2, "alpha": alpha, "gamma": gamma,
            "lamda": lamda}
        
    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.

        Args:
            t (int): time steps
            X (numpy.array): values of th model variables

        Returns:
            (np.array)
        """
        n = self.population
        s, e, i, iso, r, d = X
        dsdt = 0 - self.beta_1 * s * i / n
        dedt = -dsdt - self.beta_2 * e
        disodt = -(self.alpha + self.gamma) * iso + self.lamda * i
        drdt = self.gamma * (i + iso)
        dddt = self.alpha * (i + iso)
        didt = 0 - dsdt - dedt - disodt - drdt - dddt
        return np.array([dsdt, dedt, didt, disodt, drdt, dddt])
    
    def calc_r0(self):
        """
        Calculate (basic) reproduction number.

        Returns:
            float
        """
        try:
            rt = self.alpha1 / ((1 - self.delta) * self.alpha + self.gamma)
        except ZeroDivisionError:
            return None
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.

        Args:
            param tau (int): tau value [min]

        Returns:
            dict[str, int]
        """
        try:
            return {
                "1/beta_1 [day]": int(tau / 24 / 60 / self.beta_1),
                "1/beta_2 [day]": int(tau / 24 / 60 / self.beta_2),
                "1/alpha [day]": int(tau / 24 / 60 / self.alpha),
                "1/gamma [day]": int(tau / 24 / 60 / self.gamma),
                "1/lamba [day]": int(tau / 24 / 60 / self.lamda)
            }
        except ZeroDivisionError:
            return {p: None for p in self.DAY_PARAMETERS}
    @classmethod
    def _convert(cls, data, tau):
        """
        Divide dates by tau value [min].

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int): tau value [min] or None (skip division by tau values)

        Returns:
            pandas.DataFrame:
                Index
                    - Date (pd.Timestamp): Observation date (available when @tau is None)
                    - t (int): time steps (available when @tau is not None)
                Columns
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        df = cls._ensure_dataframe(data, name="data", columns=cls.DSIFR_COLUMNS)
        if tau is None:
            return df.set_index(cls.DATE)
        # Convert to tau-free
        tau = cls._ensure_tau(tau, accept_none=False)
        time_series = (df[cls.DATE] - df[cls.DATE].min()).dt.total_seconds() // 60
        df.index = (time_series / tau).astype(np.int64)
        df.index.name = cls.TS
        return df.drop(cls.DATE, axis=1)
    
    @classmethod
    def convert(cls, data, tau):
        """
        Divide dates by tau value [min] and convert variables to model-specialized variables.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int): tau value [min] or None (skip division by tau values)

        Returns:
            pandas.DataFrame:
                Index
                    - Date (pd.Timestamp): Observation date (available when @tau is None)
                    - t (int): time steps (available when @tau is not None)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
        """
        # Convert to tau-free if tau was specified
        df = cls._convert(data, tau)
        # Conversion of variables: un-necessary for SIR-F model
        return df.loc[:, cls.DSIFR_COLUMNS[1:]]

    @classmethod
    def convert_reverse(cls, converted_df, start, tau):
        """
        Calculate date with tau and start date, and restore Susceptible/Infected/Fatal/Recovered.

        Args:
            converted_df (pandas.DataFrame):
                Index
                    t: Dates divided by tau value (time steps)
                Columns
                    - Susceptible (int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Recovered (int): the number of recovered cases
                    - Fatal (int): the number of fatal cases
            start (pd.Timestamp): start date of simulation, like 14Apr2021
            tau (int): tau value [min]

        Returns:
            pandas.DataFrame:
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
        """
        # Calculate date with tau and start date
        df = cls._convert_reverse(converted_df, start, tau)
        # Conversion of variables: un-necessary for SIR-F model
        return df.loc[:,  cls.DSIFR_COLUMNS]  #[cls.DATE, cls.S, cls.CI, cls.F, cls.R]

    @classmethod
    def guess(cls, data, tau, q=0.5):
        """
        With (X, dX/dt) for X=S, I, R, F, guess parameter values.

        Args:
            data (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Susceptible(int): the number of susceptible cases
                    - Infected (int): the number of currently infected cases
                    - Fatal(int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
            tau (int): tau value [min]
            q (float or tuple(float,)): the quantile(s) to compute, value(s) between (0, 1)

        Returns:
            dict(str, float or pandas.Series): guessed parameter values with the quantile(s)

        Note:
            We can guess parameter values with difference equations as follows.
            - theta -> +0 (i.e. around 0 and not negative)
            - kappa -> (dF/dt) / I when theta -> +0
            - rho = - n * (dS/dt) / S / I
            - sigma = (dR/dt) / I
        """
        # Convert to tau-free and model-specialized dataset
        df = cls.convert(data=data, tau=tau)
        # Remove negative values and set variables
        df = df.loc[(df[cls.S] > 0) & (df["Infectious"] > 0) & (df["Exposed"] > 0)]
        n = cls.population
        # Calculate parameter values with difference equation and tau-free data
        beta_1_series = 0 - n * df[cls.S].diff() / df[cls.S] / df["Infectious"]
        beta_2_series = -(df["Exposed"].diff() + df[cls.S].diff()) / df["Exposed"]
        alpha_series = df["Dead"].diff() / (df["Infectious"] + df["Isolated"])
        gamma_series = df["Recovered"].diff() / (df["Infectious"] + df["Isolated"])
        lamda_series = (df["Isolated"].diff() + (gamma_series + alpha_series) * df["Isolated"]) / df["Infectious"]
        # Guess representative values
        res = {
            "beta_1": cls._clip(beta_1_series.quantile(q=q), 0, 1),
            "beta_2": cls._clip(beta_2_series.quantile(q=q), 0.01, 1),
            # "beta_2": pd.Series(data=[0, 0.2], index=[0.1, 0.9]),
            # "beta_2": cls._clip(beta_2_series.quantile(q=q), 0, 1),
            "alpha": cls._clip(alpha_series.quantile(q=q), 0, 1),
            "gamma": cls._clip(gamma_series.quantile(q=q), 0, 1),
            "lamda": cls._clip(lamda_series.quantile(q=q), 0, 1),
        }
        return res
    