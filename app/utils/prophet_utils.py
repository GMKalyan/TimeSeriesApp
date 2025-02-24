import pandas as pd
import numpy as np
import logging
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json, model_from_json
import holidays

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fit_prophet_model(df, params):
    """
    Fit a Prophet model with the given parameters
    
    :param df: Dataframe with ds and y columns
    :param params: Dictionary of model parameters
    :return: Fitted Prophet model or None if error
    """
    try:
        # Extract parameters
        seasonality_mode = params.get('seasonality_mode', 'additive')
        daily = params.get('daily_seasonality', False)
        weekly = params.get('weekly_seasonality', True)
        yearly = params.get('yearly_seasonality', True)
        growth = params.get('growth', 'linear')
        changepoint_prior_scale = params.get('changepoint_prior_scale', 0.05)
        seasonality_prior_scale = params.get('seasonality_prior_scale', 10.0)
        country_holidays = params.get('holidays_country', None)
        monthly = params.get('monthly_seasonality', False)
        
        # Initialize model
        m = Prophet(
            seasonality_mode=seasonality_mode,
            daily_seasonality=daily,
            weekly_seasonality=weekly,
            yearly_seasonality=yearly,
            growth=growth,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale
        )
        
        # Add country holidays if specified
        if country_holidays:
            m.add_country_holidays(country_name=country_holidays)
        
        # Add monthly seasonality if requested
        if monthly:
            m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)
        
        # Fit model
        logger.info("Fitting Prophet model...")
        m.fit(df)
        logger.info("Model fitting completed")
        
        return m
    except Exception as e:
        logger.error(f"Error fitting Prophet model: {e}")
        return None

def make_forecast(model, periods, freq='D', include_history=True):
    """
    Generate a forecast using a fitted Prophet model
    
    :param model: Fitted Prophet model
    :param periods: Number of periods to forecast
    :param freq: Frequency of forecast (D=daily, W=weekly, etc)
    :param include_history: Whether to include historical data in the forecast
    :return: Forecast dataframe or None if error
    """
    try:
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        
        # Check if model was fit with logistic growth
        if model.growth == 'logistic':
            # Try to get cap and floor from model params
            cap = 1
            floor = 0
            
            # Set cap and floor in future dataframe
            future['cap'] = cap
            future['floor'] = floor
        
        # Make prediction
        forecast = model.predict(future)
        
        return forecast
    except Exception as e:
        logger.error(f"Error making forecast: {e}")
        return None

def cross_validate_model(model, initial, period, horizon, parallel='processes'):
    """
    Perform cross-validation on a Prophet model
    
    :param model: Fitted Prophet model
    :param initial: Initial training period
    :param period: Spacing between cutoff dates
    :param horizon: Forecast horizon
    :param parallel: Type of parallelism ('processes' or 'threads')
    :return: Cross-validation results dataframe or None if error
    """
    try:
        logger.info(f"Cross-validating with initial={initial}, period={period}, horizon={horizon}")
        df_cv = cross_validation(
            model, 
            initial=initial,
            period=period, 
            horizon=horizon,
            parallel=parallel
        )
        
        return df_cv
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        try:
            # Try again with different parallelism
            alt_parallel = 'threads' if parallel == 'processes' else 'processes'
            logger.info(f"Retrying with {alt_parallel} parallelism")
            
            df_cv = cross_validation(
                model, 
                initial=initial,
                period=period, 
                horizon=horizon,
                parallel=alt_parallel
            )
            
            return df_cv
        except Exception as e2:
            logger.error(f"Error in retry cross-validation: {e2}")
            return None

def get_performance_metrics(cv_results, rolling_window=0.1):
    """
    Calculate performance metrics from cross-validation results
    
    :param cv_results: Cross-validation results dataframe
    :param rolling_window: Size of rolling window for metrics calculation
    :return: Performance metrics dataframe or None if error
    """
    try:
        df_p = performance_metrics(cv_results, rolling_window=rolling_window)
        return df_p
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return None

def optimize_hyperparameters(df, param_grid, initial, period, horizon):
    """
    Find the best hyperparameters for a Prophet model
    
    :param df: Training dataframe with ds and y columns
    :param param_grid: Dictionary of parameter grids
    :param initial: Initial training period
    :param period: Spacing between cutoff dates
    :param horizon: Forecast horizon
    :return: Dictionary with best parameters and all results
    """
    try:
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []
        
        logger.info(f"Running hyperparameter optimization with {len(all_params)} combinations")
        
        # Use cross validation to evaluate all parameters
        for params in all_params:
            logger.info(f"Testing parameters: {params}")
            m = Prophet(**params).fit(df)
            
            try:
                df_cv = cross_validation(
                    m, 
                    initial=initial,
                    period=period,
                    horizon=horizon,
                    parallel="processes"
                )
            except:
                # Try with threads if processes fails
                df_cv = cross_validation(
                    m, 
                    initial=initial,
                    period=period,
                    horizon=horizon,
                    parallel="threads"
                )
                
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
        
        # Find the best parameters
        best_params = all_params[np.argmin(rmses)]
        
        # Create results dataframe
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        
        return {
            'best_params': best_params,
            'all_results': tuning_results
        }
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {e}")
        return None

def save_model_to_json(model):
    """
    Serialize a Prophet model to JSON
    
    :param model: Fitted Prophet model
    :return: JSON string representation of the model
    """
    try:
        return model_to_json(model)
    except Exception as e:
        logger.error(f"Error serializing model: {e}")
        return None

def load_model_from_json(json_str):
    """
    Load a Prophet model from a JSON string
    
    :param json_str: JSON string representation of a Prophet model
    :return: Loaded Prophet model
    """
    try:
        return model_from_json(json_str)
    except Exception as e:
        logger.error(f"Error deserializing model: {e}")
        return None