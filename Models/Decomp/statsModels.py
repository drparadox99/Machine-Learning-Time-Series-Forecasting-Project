
# #time series data with explicit dates

# import numpy as np
# import pandas as pd
# from statsmodels.tsa.seasonal import seasonal_decompose
# import matplotlib.pyplot as plt

# # Generate random time series data
# np.random.seed(42)
# dates = pd.date_range(start='2023-01-01', periods=365)
# random_data = np.random.randn(len(dates))
# ts = pd.Series(random_data, index=dates)

# # Perform time series decomposition
# decomposition = seasonal_decompose(ts, model='additive')

# # Get the trend, seasonal, and residual components
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid

# # Plotting the original time series and its components
# plt.figure(figsize=(10, 8))

# plt.subplot(411)
# plt.plot(ts, label='Original Time Series')
# plt.legend()

# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend()

# plt.subplot(413)
# plt.plot(seasonal, label='Seasonality')
# plt.legend()

# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend()

# plt.tight_layout()
# plt.show()



######
#time series data without explicit dates

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Generate random time series data
np.random.seed(42)
random_data = np.random.randn(365)  # Generating 365 random data points

# Convert the data into a Pandas Series without specifying dates
ts = pd.Series(random_data)

print(ts.shape)


#In this example, period=12 indicates that the function should look for a seasonal pattern that repeats every 12 data points (in this case, 12 months if it's monthly data)
# # Perform time series decomposition
# decomposition = seasonal_decompose(ts, model='additive', period=12)  # Assuming a seasonal period of 12

# # Get the trend, seasonal, and residual components
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid



def extract_components(ts,composition,period):
    decomp_object = seasonal_decompose(ts, model=composition, period=period)
    trend = decomp_object.trend
    seasonal = decomp_object.seasonal
    residual = decomp_object.resid
    return trend.to_numpy(),seasonal.to_numpy(),residual.to_numpy()


trend, seasonal, residual = extract_components(ts,"additive",12)

print(type(trend))
print(trend.shape)
print(seasonal.shape)
print(residual.shape)

# Plotting the original time series and its components
plt.figure(figsize=(10, 8))

plt.subplot(411)
plt.plot(ts, label='Original Time Series')
plt.legend()

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend()

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend()

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend()

plt.tight_layout()
plt.show()





#Using fourrier decompose 
#from scipy.fft import fft, ifft

# Perform Fourier Transform
#fft_values = fft(random_data)

# Filter based on frequencies (representing seasonality)
# ...

# Perform Inverse Fourier Transform to reconstruct components
#seasonal_component = ifft(filtered_values)
