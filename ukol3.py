import pandas as pd
import numpy as np

# 1. Load the Data
df=pd.read_csv('data/sensor_data.csv')
print(df.describe(include='all'))
print(df.head())

df['timestamp']=pd.to_datetime(df['timestamp'])
df['sensor_id']=df['sensor_id'].astype('category')

df.set_index(['sensor_id', 'timestamp'], inplace=True)
start_len=len(df)

# 2. Data Cleaning
#urcim kvantily
p20=df['temperature'].quantile(0.20)
p80=df['temperature'].quantile(0.80)

#urcim outliers a missing values
outliers = ((df['temperature'] < p20) | (df['temperature'] > p80))
chybejici = df.isna().any(axis=1)
to_remove = outliers & chybejici

remove_len=to_remove.sum()

print("Rows to remove:", remove_len)





# 3. Filtering and Feature Engineering

# 3.1
obs_count=len(df[(df['temperature']>25) & (df['humidity']<40)])
print("High temp low hum count:", obs_count)

# 3.2
cond=[
    df['temperature']<20,
    (df['temperature']>=20) & (df['temperature']<=30),
    df['temperature']>30
]
choic=['low','medium','high']
df['temp_class']=np.select(cond, choic, default='unknown')


# 3.3
mean_t=df['temperature'].mean()
std_t=df['temperature'].std()

drop_cond = (df.index.get_level_values('sensor_id') == 'S5') & \
            ((df['temperature'] < mean_t - std_t) | (df['temperature'] > mean_t + std_t))

df=df[~drop_cond]


# 4. Time and Date Analysis
times=df.index.get_level_values('timestamp')

# 4.1
first_t = times.min()
df['hours_from_start']=(times - first_t).total_seconds() / 3600.0


# 4.2
df['is_daytime'] = (times.hour >= 6) & (times.hour < 18)

# 4.3
df['after_12h']=df['hours_from_start'] >= 12

# 4.5 (is_weekend udelam pred 4.4 at to je v hlavnim df)
df['is_weekend'] = times.dayofweek >= 5

# 4.4
df_s2 = df.xs('S2', level='sensor_id').copy()
df_s2=df_s2.sort_index()

df_s2['rolling_temp_30m'] = df_s2['temperature'].rolling('30min').mean()

df_s2['hum_diff'] = df_s2['humidity'] - df_s2['humidity'].shift(-1)

temp_up = df_s2['temperature'] - df_s2['temperature'].shift(1) > 5
hum_down = df_s2['humidity'].shift(1) - df_s2['humidity'] > 20

df_s2['alert'] = temp_up & hum_down

df_s2.to_csv('data/sensor_data_S2.csv')