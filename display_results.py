from tensorboard import notebook

log_dir = "results/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))