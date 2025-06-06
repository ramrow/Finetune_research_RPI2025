import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tensorboard import notebook

log_dir = "results/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))