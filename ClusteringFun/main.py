import numpy as np
header = ["height(cm)", "weight(kg)"] #, "size(t-shirt)"]
X_train = [
    [158, 58], # "M"
    [158, 59], # "M"
    [158, 63], # "M"
    [160, 59], # "M"
    [160, 60], # "M"
    [163, 60], # "M"
    [163, 61], # "M"
    [160, 64], # "L"
    [163, 64], # "L"
    [165, 61], # "L"
    [165, 62], # "L"
    [165, 65], # "L"
    [168, 62], # "L"
    [168, 63], # "L"
    [168, 66], # "L"
    [170, 63], # "L"
    [170, 64], # "L"
    [170, 68] # "L"
]
# TODO: normalize data before calculating distances
y_train = ["M", "M", "M", "M", "M", "M", "M", 
           "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L"]

# Min-Max Normalization (scales to [0, 1])
def normalize_minmax(X):
    X_array = np.array(X)
    X_min = X_array.min(axis=0)
    X_max = X_array.max(axis=0)
    X_normalized = (X_array - X_min) / (X_max - X_min)
    return X_normalized.tolist(), X_min, X_max

# Z-score Normalization
def normalize_zscore(X):
    X_array = np.array(X)
    X_mean = X_array.mean(axis=0)
    X_std = X_array.std(axis=0)
    X_normalized = (X_array - X_mean) / X_std
    return X_normalized.tolist(), X_mean, X_std

X_train_normalized, X_min, X_max = normalize_minmax(X_train)

def normalize_test_data(X_test, X_min, X_max):
    X_test_array = np.array(X_test)
    X_test_normalized = (X_test_array - X_min) / (X_max - X_min)
    return X_test_normalized.tolist()