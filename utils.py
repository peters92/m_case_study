import pandas as pd
import numpy as np

from econml.dml import LinearDML
from xgboost import XGBRegressor
from matplotlib import pyplot as plt

def estimate_elasticity(dataframe):
    # This function creates a dataframe with estimated price elasticities for 
    # each region in the input dataframe
    estimate_dict = {}

    # Run for each region
    for region in dataframe['region'].unique():
        df = dataframe[dataframe['region'] == region]
        # Create inputs to causal model
        y, T, X, W = _prepare_inputs(df)
        # Get estimate of elasticity (E) and the intercept (K) in
        # ln(Q) = K + E * ln(P)  
        E, intercept = _estimate_elasticity(y, T, X, W)
        estimate_dict[region] = {'price elasticity estimate': E,
                                 'intercept': intercept}
    
    # convert to dataframe
    estimate_pd = (
        pd.DataFrame
            .from_dict(estimate_dict, orient='index')
            .reset_index(names='region'))
    return estimate_pd

def _prepare_inputs(df):
    # Create the inputs fitting the causal model
    y = np.log(df['volume_lb'])
    T = np.log(df['avg_price_lb'])
    X = df[['year_month']].astype(int)
    W = df[['gas_index']].astype(float)
    
    return (y, T, X, W)

def _estimate_elasticity(y, T, X, W):
    # Create estimator and fit inputs
    est = (
        LinearDML(
            model_y=XGBRegressor(),
            model_t=XGBRegressor(),
            random_state=42
                    ))
    est.fit(y, T, X=X, W=W)

    # Retrieve the estimated elasticity and intercept
    E = est.const_marginal_effect(X).mean()
    intercept = est.intercept_

    return (E, intercept)

def visualize_profit_curves(estimate_pd, WHOLESALE_COST):
    # This function visualizes the optimal price curve for each region
    regions = estimate_pd['region'].unique()
    
    for region in regions:
        region_pd = estimate_pd[estimate_pd['region'] == region]
        E = region_pd['price elasticity estimate'].values[0]
        intercept = region_pd['intercept'].values[0]
        opt_price = WHOLESALE_COST / (1 + (1 / E))

        # Calculate price/demand vectors and the profit
        price = np.arange(WHOLESALE_COST, 6, 0.1)
        demand = np.exp(intercept + np.log(price) * E)
        profit = (price - WHOLESALE_COST) * demand
        
        # plot of profit curve
        plt.plot(price, profit)
        plt.title('Net profit as function of price')
        plt.xlabel('Price / lb [USD]')
        plt.ylim([0, 0.2])