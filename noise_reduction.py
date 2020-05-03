import pandas
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

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
w = savgol_filter(UK_walking['Value'], 7, 1)
w2 = savgol_filter(US_walking['Value'], 7, 1)
w3 = savgol_filter(SK_walking['Value'], 7, 1)
w4 = savgol_filter(NZ_walking['Value'], 7, 1)

# plt.plot(UK_walking.index, w, label='UK')
# plt.plot(US_walking.index, w2, label='US')
# plt.plot(SK_walking.index, w3, label='SK')
# plt.plot(NZ_walking.index, w4, label='NZ')
# plt.legend()
# plt.show()

# Gaussian Filtering for smoothing

g = gaussian_filter(UK_walking['Value'], sigma=2.5)
g2 = gaussian_filter(US_walking['Value'], sigma=2.5)
g3 = gaussian_filter(SK_walking['Value'], sigma=2.5)
g4 = gaussian_filter(NZ_walking['Value'], sigma=2.5)

plt.plot(UK_walking.index, g, label='UK')
plt.plot(US_walking.index, g2, label='US')
plt.plot(SK_walking.index, g3, label='SK')
plt.plot(NZ_walking.index, g4, label='NZ')
plt.legend()
plt.show()
