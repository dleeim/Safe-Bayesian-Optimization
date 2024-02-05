import numpy as np

# Number of trials and iterations
num_trials = 3
num_iterations = 5

# Dictionary to store data for each trial
data = {}

for trial in range(num_trials):
    # List to store data for each iteration in the trial
    trial_data = []

    for iteration in range(num_iterations):
        # Perform your calculations here and store the result
        result = trial * iteration**2+2  # Replace this with your actual calculation

        # Append the result to the trial_data list
        trial_data.append(result)

    # Store the trial_data list in the data dictionary with the trial as the key
    data[f"Trial {trial + 1}"] = trial_data

# Print the stored data
for trial, trial_data in data.items():
    print(f"{trial}: {trial_data}")

# Find average
for i in range(5):
    mean = np.mean([data[f"Trial {trial + 1}"][i] for trial in range(num_trials)])
    max = np.max([data[f"Trial {trial + 1}"][i] for trial in range(num_trials)])
    print(mean, max)
