# IU - International University of Applied Science
# Time Series Analysis
# Course Code: DLBDSTSA01

# Utiliy function to prepare sample data

import numpy as np
import pandas as pd

def prepare_gdp_data(df_gdp, country):

    # extract subdata
    df_gdp = df_gdp.iloc[:,150:]
    
    # extract country
    gdp_country = df_gdp.loc[country]
    
    # drop missing values
    gdp_country = gdp_country.dropna()

    # iterate over all data points
    for i in range(len(gdp_country)):
        
        # convert text abbreviations to numbers 
        last_digit = gdp_country[i][-1:]
        last_two_digit = gdp_country[i][-2:]
        if last_digit == 'M':
            num_str = gdp_country[i][:-1]
            num_flt = float(num_str)
            gdp_country[i] = np.log(int(num_flt * 1000))
        elif last_digit == 'B':
            num_str = gdp_country[i][:-1]
            num_flt = float(num_str)
            gdp_country[i] = np.log(int(num_flt * 1000000))
        elif last_two_digit == 'TR':
            num_str = gdp_country[i][:-2]
            num_flt = float(num_str)
            gdp_country[i] = np.log(int(num_flt * 1000000000))
        else: 
            gdp_country[i] = np.log(gdp_country[i])

    # adapt index
    gdp_country.index = gdp_country.index.astype(int)

    return gdp_country