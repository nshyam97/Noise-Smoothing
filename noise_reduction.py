import pandas
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

mobility_data = pandas.read_csv("applemobilitytrends-2020-04-13.csv")


# Function to transpose the data. Currently the date values are columns and so we need to make these rows
# to make plotting the data easier
def transpose_df(region_name, transportation):
    # First choose the region and transportation type to create the usable dataframe
    df = mobility_data[(mobility_data['region'] == region_name) &
                       (mobility_data['transportation_type'] == transportation)]
    # Drop the geo_type column as it isn't useful anymore
    df = df.drop(['geo_type'], axis=1)
    # Pivots the dataframe from a wide to a tall format. Move the Date and Values as separate rows and corresponding
    # columns.
    df_t = df.melt(['region', 'transportation_type'], var_name='Date', value_name='Value')
    # Convert date column to datetime column
    df_t.Date = pandas.to_datetime(df_t.Date, format='%Y-%m-%d')
    # Make date column the index column to allow for easier plotting
    df_t.set_index('Date', inplace=True)
    # Values are currently percentages with the first value being the baseline. To make it a change in
    # baseline mobility, minus all values by the first value.
    df_t.Value = df_t.Value - df_t.Value.iloc[0]
    # Round all the values to 2 decimal places
    df_t.Value = df_t.Value.round(2)
    # Return the finished dataframe, ready to plot
    return df_t


UK_walking = transpose_df('UK', 'walking')
UK_driving = transpose_df('UK', 'driving')
UK_transit = transpose_df('UK', 'transit')

US_walking = transpose_df('United States', 'walking')
US_driving = transpose_df('United States', 'driving')
US_transit = transpose_df('United States', 'transit')

SK_walking = transpose_df('Republic of Korea', 'walking')
SK_driving = transpose_df('Republic of Korea', 'driving')

NZ_walking = transpose_df('New Zealand', 'walking')
NZ_driving = transpose_df('New Zealand', 'driving')
NZ_transit = transpose_df('New Zealand', 'transit')

# Savgol filter for smoothing
UK_walking = UK_walking.assign(savgol=savgol_filter(UK_walking['Value'], 7, 1))
US_walking = US_walking.assign(savgol=savgol_filter(US_walking['Value'], 7, 1))
SK_walking = SK_walking.assign(savgol=savgol_filter(SK_walking['Value'], 7, 1))
NZ_walking = NZ_walking.assign(savgol=savgol_filter(NZ_walking['Value'], 7, 1))

fig = go.Figure()
fig.add_trace(go.Scatter(x=UK_walking.index, y=UK_walking.savgol, name='UK', mode='lines'))
fig.add_trace(go.Scatter(x=US_walking.index, y=US_walking.savgol, name='US', mode='lines'))
fig.add_trace(go.Scatter(x=SK_walking.index, y=SK_walking.savgol, name='SK', mode='lines'))
fig.add_trace(go.Scatter(x=NZ_walking.index, y=NZ_walking.savgol, name='NZ', mode='lines'))
fig.show()

# Gaussian Filtering for smoothing

UK_walking = UK_walking.assign(gaussian=gaussian_filter(UK_walking['Value'], sigma=2.5))
US_walking = US_walking.assign(gaussian=gaussian_filter(US_walking['Value'], sigma=2.5))
SK_walking = SK_walking.assign(gaussian=gaussian_filter(SK_walking['Value'], sigma=2.5))
NZ_walking = NZ_walking.assign(gaussian=gaussian_filter(NZ_walking['Value'], sigma=2.5))

fig = go.Figure()
fig.add_trace(go.Scatter(x=UK_walking.index, y=UK_walking.gaussian, name='UK', mode='lines'))
fig.add_trace(go.Scatter(x=US_walking.index, y=US_walking.gaussian, name='US', mode='lines'))
fig.add_trace(go.Scatter(x=SK_walking.index, y=SK_walking.gaussian, name='SK', mode='lines'))
fig.add_trace(go.Scatter(x=NZ_walking.index, y=NZ_walking.gaussian, name='NZ', mode='lines'))
fig.show()

# Seasonal decompose
UK_walking = UK_walking.assign(seasonal=seasonal_decompose(UK_walking['Value'], model='additive', period=7).trend)
US_walking = US_walking.assign(seasonal=seasonal_decompose(US_walking['Value'], model='additive', period=7).trend)
SK_walking = SK_walking.assign(seasonal=seasonal_decompose(SK_walking['Value'], model='additive', period=7).trend)
NZ_walking = NZ_walking.assign(seasonal=seasonal_decompose(NZ_walking['Value'], model='additive', period=7).trend)

fig = go.Figure()
fig.add_trace(go.Scatter(x=UK_walking.index, y=UK_walking.seasonal, name='UK', mode='lines'))
fig.add_trace(go.Scatter(x=US_walking.index, y=US_walking.seasonal, name='US', mode='lines'))
fig.add_trace(go.Scatter(x=SK_walking.index, y=SK_walking.seasonal, name='SK', mode='lines'))
fig.add_trace(go.Scatter(x=NZ_walking.index, y=NZ_walking.seasonal, name='NZ', mode='lines'))
fig.show()

