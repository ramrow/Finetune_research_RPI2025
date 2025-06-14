from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

model_name_or_path = "finalform/foam-nuTilda-llama-13B"

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"  # Automatically place model on available GPUs
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

output = pipe("<s>[INST] Conduct a RAS simulation for the turbulent flow over a backward-facing step using the pimpleFoam solver. Set the inlet velocity to 10 m/s and the outlet pressure to 0. The walls should be no-slip, and the front and back should be empty. Use the k-epsilon turbulence model for this simulation. The fluid is Newtonian with a kinematic viscosity of 1e-5 m²/s. The control settings are: endTime = 0.40 and writeInterval = 0.005.", do_sample=True)

print(output[0]['generated_text'])

