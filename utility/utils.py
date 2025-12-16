import os
import pandas as pd
import logging
import numpy as np

def get_logger(log_dir, name='train'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def process_time_data(df, q=4, eps=1e-6):
    """
    Process the data to get the discretized time labels.
    """

    df = df.copy()
    # Ensure censor is int (0 or 1)
    # Note: Adjust logic if your data uses 0 for event or 1 for event.
    uncensored_df = df[df.censor == 0]
    
    if len(uncensored_df) == 0:
        print("Warning: No uncensored data found for discretization.")
        return df

    disc_labels, q_bins = pd.qcut(uncensored_df.survival_months, q=q, retbins=True, labels=False)
    q_bins[-1] = df['survival_months'].max() + eps
    q_bins[0] = df['survival_months'].min() - eps
    
    disc_labels, q_bins = pd.cut(df['survival_months'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
    
    # Insert new column
    if 'time_interval_label' in df.columns:
        df['time_interval_label'] = disc_labels.values.astype(float)
    else:
        df.insert(2, 'time_interval_label', disc_labels.values.astype(float))
        
    return df