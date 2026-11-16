#loading data
from os import times

import pandas as pd
import numpy as np
#1
df=pd.read_csv('data/sensor_data.csv')
print(df)
print(df.dtypes)

df=df.astype({'timestamp':'datetime64[ns]','sensor_id':'category'})
print(df.dtypes)

df.set_index(['sensor_id', 'timestamp'], inplace=True)

start_len=len(df)
#2
temp_low=df['temperature'].quantile(0.20)
temp_high=df['temperature'].quantile(0.80)

df_mid_temp=df[(df['temperature']>=temp_low) & (df['temperature']<=temp_high)]
df_mid_temp_clean=df_mid_temp.dropna()
outlier_count=len(df)-len(df_mid_temp_clean)
print("#rows removed (na or outliers):", outlier_count)

clean_len=len(df_mid_temp_clean)
print("#clean rows:", clean_len)
#3
#
print("High temp low hum count:", len(df_mid_temp_clean[(df_mid_temp_clean['temperature']>25) & (df_mid_temp_clean['humidity']<40)]))
#
df_mid_temp_clean['temp_class']=pd.cut(df_mid_temp_clean['temperature'], bins=[-np.inf, 20, 30, np.inf], labels=['low', 'medium', 'high'])
#

df_clean_wos5=df_mid_temp_clean[~((df_mid_temp_clean.index.get_level_values('sensor_id')=='S5'))]
stdev_temp=df_clean_wos5['temperature'].std()
df_clean_wos5['zscore_temp'] = (df_clean_wos5['temperature'] - df_clean_wos5['temperature'].mean()) / stdev_temp
df1=df_clean_wos5[(df_clean_wos5['zscore_temp']>=-1) & (df_clean_wos5['zscore_temp']<=1)]

#4

#
first_measurement=df1.index.get_level_values('timestamp').min()
df1['time_since_first_measurement'] = (df1.index.get_level_values('timestamp') - first_measurement).total_seconds() / 3600
df1['is_daytime'] = (df1.index.get_level_values('timestamp').hour >= 6) & (df1.index.get_level_values('timestamp').hour < 18)
df1['after_12h'] = df1['time_since_first_measurement'] >= 12

#4.5 dayofweek: Monday=0 ... Saturday=5, Sunday=6
df1['is_weekend'] = df1.index.get_level_values('timestamp').dayofweek >= 5


df2=df1.loc['S2']
df2['rolling_temp_30m'] = df2['temperature'].rolling('30min').mean()
df2['hum_change']=df2['humidity']-df2['humidity'].shift(-1)
df2['temp_change']=df2['temperature']-df2['temperature'].shift(-1)
df2['alert'] = (df2['hum_change'].abs() > 10) & (df2['temp_change'].abs() > 5)

df2.to_csv('data/sensor_data_S2.csv')