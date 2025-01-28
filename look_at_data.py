import pandas as pd
from loguru import logger

"""
This is just supposed to be a small script for looking at the data for reverse engineering
"""


# Velocity file
logger.info("Looking at the .vel file")
data_riv_df = pd.read_csv("data/20250122130314.vel")
print(data_riv_df.head())

# Sum file
# Uses special encoding: ISO-8859-1
logger.info("Looking at the .sum file")
data_sum_df = pd.read_csv("data/20250122130314.sum",encoding='ISO-8859-1')
print(data_sum_df.head())

# Signal to noise ratio file
logger.info("Looking at the .snr file")
data_wsp_df = pd.read_csv("data/20250122130314.snr")
print(data_wsp_df.head())

# Signal to noise ratio file
logger.warning("The .wsp file is a binary file")
#data_wsp_df = pd.read_csv("data/20250122130314.wsp")
#print(data_wsp_df.head())

# River file
logger.warning("The .riv file is a binary file; Doing strings data/file.riv | head -n 100 reveals some fun stuff")
#data_riv_df = pd.read_csv("data/20250122130314.riv")
#print(data_riv_df.head())