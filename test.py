import numpy as np
import matplotlib.pyplot as plt

# Simulated data: 50 trials, each with metric values for 100 thresholds
# For demonstration, let's assume the metric is some function of the threshold with added noise
thresholds = np.linspace(0, 10, 100)
data = [np.sin(thresholds) + np.random.normal(0, 0.5, thresholds.shape) for _ in range(50)]

# Convert to a numpy array for easier calculations
data = np.array(data)

# Compute mean and standard deviation for each threshold across trials
mean = np.mean(data, axis=0)
std_dev = np.std(data, axis=0)

# Plotting
plt.plot(thresholds, mean, color='blue', label='Mean')
plt.fill_between(thresholds, mean - std_dev, mean + std_dev, color='blue', alpha=0.2, label='1 Std Dev')
plt.title('Mean and Variance of Metric across 50 Trials')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.legend()
plt.show()
