# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:37:17 2022

@author: bitsi
"""

import numpy as np
import pandas as pd
from covsirphy.util.error import SubsetNotFoundError, deprecate
from covsirphy.cleaning.cbase import CleaningBase
from covsirphy.cleaning.jhu_data import JHUData
from covsirphy.cleaning.country_data import CountryData
from covsirphy.cleaning.jhu_complement import JHUDataComplementHandler



class JHUData_extend(CleaningBase):
    """
    Data cleaning of user_supplied data, subclassed from CleaningBase
    """
    
    _RAW_COLS_DEFAULT = [
        CleaningBase.DATE, CleaningBase.ISO3, CleaningBase.COUNTRY, CleaningBase.PROVINCE,
        CleaningBase.C, CleaningBase.CI, CleaningBase.F, CleaningBase.R, CleaningBase.N
    ]
    additional_cols = ["Isolated"]
    _RAW_COLS_DEFAULT += additional_cols
    
    def __init__(self, filename=None, data=None, citation=None):
        variables = ["Susceptible", "Exposed", "Infectious", "Isolated", "Recovered", "Dead", "Population"]
        super().__init__(filename=filename, data=data, citation=citation, variables=variables)
        # Recovery period
        self._recovery_period = None
        self.variables = variables

    def subset(self, country, province=None, start_date=None, end_date=None,
               population=None, recovered_min=1):
        """
        Return the subset of dataset.

        Args:
            country (str): country name or ISO3 code
            province (str or None): province name
            start_date (str or None): start date, like 22Jan2020
            end_date (str or None): end date, like 01Feb2020
            population (int or None): population value
            recovered_min (int): minimum number of recovered cases records must have

        Returns:
            pandas.DataFrame
                Index
                    reset index
                Columns
                    - Date (pd.Timestamp): Observation date
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases (> 0)
                    - Susceptible (int): the number of susceptible cases, if calculated

        Note:
            If @population (high priority) is not None or population values are registered in subset,
            the number of susceptible cases will be calculated.
        """
        # country_alias = self.ensure_country_name(country)
        # Subset with area, start/end date
        # try:
        #     subset_df = super().subset(
        #         country=country, province=province, start_date=start_date, end_date=end_date)
        # except SubsetNotFoundError:
        #     raise SubsetNotFoundError(
        #         country=country, country_alias=country_alias, province=province,
        #         start_date=start_date, end_date=end_date) from None
        df = self._subset_by_area(country=country, province=province)
        
        # Calculate Susceptible
        # df = self._calculate_susceptible(subset_df, population)
        # Select records where Recovered >= recovered_min
        recovered_min = self._ensure_natural_int(recovered_min, name="recovered_min", include_zero=True)
        df = df.loc[df[self.R] >= recovered_min, :].reset_index(drop=True)
        if df.empty:
            raise SubsetNotFoundError(
                country=country, country_alias=country, province=province,
                start_date=start_date, end_date=end_date,
                message=f"with 'Recovered >= {recovered_min}'") from None
        return df
    
    def _subset_by_area(self, country, province=None):
        """
        Return subset for the country/province.

        Args:
            country (str): country name
            province (str or None): province name or None (country level data)

        Returns:
            pandas.DataFrame: subset for the country/province, columns are not changed

        Raises:
            SubsetNotFoundError: no records were found for the condition
        """
        # Country level
        if province is None or province == self.UNKNOWN:
            df = self._cleaned_df.copy()
            country_alias = country
            df = df.loc[df[self.COUNTRY] == country_alias]
            return df.reset_index(drop=True)
        # # Province level
        # df = self.layer(country=country)
        # df = df.loc[df[self.PROVINCE] == province]
        if df.empty:
            raise SubsetNotFoundError(country=country)
        return df.reset_index(drop=True)
    
    def from_dataframe(cls, dataframe):
        """
        Create JHUData instance using a pandas dataframe.

        Args:
            dataframe (pd.DataFrame): cleaned dataset
                Index
                    reset index
                Columns
                    - Date: Observation date
                    - ISO3: ISO3 code (optional)
                    - Country: country/region name
                    - Province: province/prefecture/state name
                    - Confirmed: the number of confirmed cases
                    - Infected: the number of currently infected cases
                    - Fatal: the number of fatal cases
                    - Recovered: the number of recovered cases
                    - Popupation: population values (optional)
            directory (str): directory to save geometry information (for .map() method)

        Returns:
            covsirphy.JHUData: JHU-style dataset
        """
        df = cls._ensure_dataframe(dataframe, name="dataframe")
        df[cls.ISO3] = df[cls.ISO3] if cls.ISO3 in df else cls.UNKNOWN
        instance = cls()
        instance._cleaned_df = cls._ensure_dataframe(df, name="dataframe", columns=cls._RAW_COLS_DEFAULT)
        return instance
        