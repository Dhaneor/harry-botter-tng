import numpy as np
import pandas as pd

# Create sample market data
timestamps = pd.date_range(start='2023-01-01', periods=5, freq='D')
assets = ['AAPL', 'GOOGL', 'MSFT']
market_data = np.random.rand(5, 3)  # 5 timestamps, 3 assets

# Create sample signals for different parameter combinations
num_param_combinations = 4
signals = np.random.choice([0, 1], size=(5, 3, num_param_combinations))

# Define the structure of our array
dtype = [('price', float)] + [(f'signal_{i}', int) for i in range(num_param_combinations)]

# Create the structured array
structured_data = np.empty((5, 3), dtype=dtype)
structured_data['price'] = market_data
for i in range(num_param_combinations):
    structured_data[f'signal_{i}'] = signals[:,:,i]

print("Structured data shape:", structured_data.shape)
print("\nStructured data:")
print(structured_data)

# Access data
print("\nMarket data for all assets at first timestamp:")
print(structured_data['price'][0])

print("\nAll signals for first asset across all timestamps:")
print(structured_data[['signal_0', 'signal_1', 'signal_2', 'signal_3']][:,0])

print("\nSignal from 3rd parameter combination for all assets and timestamps:")
print(structured_data['signal_2'])

# Create a pandas MultiIndex DataFrame from the structured array
index = pd.MultiIndex.from_product([timestamps, assets], names=['Timestamp', 'Asset'])
df = pd.DataFrame(structured_data.ravel(), index=index)

print("\nMultiIndex DataFrame:")
print(df)

# Example of selecting data using the DataFrame
print("\nAll data for AAPL:")
print(df.loc[(slice(None), 'AAPL'), :])

print("\nAll signals for the first timestamp:")
print(df.loc[(timestamps[0], slice(None)), 'signal_0':])