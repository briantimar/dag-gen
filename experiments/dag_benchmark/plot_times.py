import numpy as np
import matplotlib.pyplot as plt

times = np.load("times.npy")
intermediate_sizes = list(range(50))
times_mean = np.mean(times, axis=1)
times_err = np.std(times, axis=1) / np.sqrt(times.shape[1])

fig, ax = plt.subplots()
plt.errorbar(intermediate_sizes, times_mean, yerr=times_err)
plt.xlabel("Num. intermediate vertices")
plt.ylabel("time")
fig.savefig("forward_pass_times.png")