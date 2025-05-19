import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Path to your TensorBoard log directory
log_dir = "results/runs"

# Load the event file
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# Tags you want to plot
tags = ['train/epoch', 'train/learning-rate', 'train/loss']

# Plotting each tag
for tag in tags:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    plt.figure()
    plt.plot(steps, values, label=tag)
    plt.xlabel('Step')
    plt.ylabel(tag.split('/')[-1].replace('-', ' ').capitalize())
    plt.title(f"{tag} over time")
    plt.legend()
    plt.grid(True)

plt.show()