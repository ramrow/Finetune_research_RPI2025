from transformers import pipeline

model_name_or_path = "llama-foam" #path/to/your/model/or/name/on/hub
pipe = pipeline("text-generation", model=model_name_or_path,)

print(pipe("Question: Conduct a RAS simulation for the turbulent flow over " \
"a backward-facing step using the pimpleFoam solver. Set the inlet velocity to" \
" 10 m/s and the outlet pressure to 0. The walls should be no-slip, and the front "
"and back should be empty. Use the k-epsilon turbulence model for this simulation. " \
"The fluid is Newtonian with a kinematic viscosity of 1e-5 m²/s. The control settings " \
"are: endTime = 0.40 and writeInterval = 0.005."))
