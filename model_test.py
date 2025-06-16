from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# model_name_or_path = "finalform/foam-nuTilda-sft-llama2-13B"
model_name_or_path = "finalform/foam-nuTilda-codellama2-13b"

md = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"  # Automatically place model on available GPUs
)

tk = AutoTokenizer.from_pretrained(model_name_or_path)

text = "<s>[INST] Conduct a RAS simulation of turbulent flow over a backward-facing step using pimpleFoam. Set the inlet velocity to 10 m/s, maintain the outlet pressure at 0, apply no-slip conditions to the walls, and treat the front and back as empty. Use the k-epsilon turbulence model for this simulation. The fluid is a Newtonian fluid with a kinematic viscosity of 1e-5 m^2/s. Control settings are as follows: endTime = 0.30, writeInterval = 0.005."



pipe = pipeline(task="text-generation", model=md, tokenizer=tk, device_map="auto")

messages = [
    {"role": "user", "content": text},
]

output = pipe(messages, max_new_tokens=512)
print(output[0])

