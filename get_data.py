import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tensorboard import notebook

# class custom_notebook(notebook):
#     def kill(pid):
#         if os.name == "nt":
#             subprocess.check_output(["taskkill", "/pid", str(int(pid)), "/f"])
#             manager.remove_info_file(pid)
#         else:
#             os.kill(pid, signal.SIGTERM)

# Path to your TensorBoard log directory
log_dir = "pre_results/runs"
# notebook.start("kill 16472")
notebook.start("--logdir {} --port 4000".format(log_dir))
