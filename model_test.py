from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# model_name_or_path = "finalform/foam-nuTilda-sft-llama2-13B"
model_name_or_path = "/home/xup2/environment/main_finetune/model_test.py"

md = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"  # Automatically place model on available GPUs
)

tk = AutoTokenizer.from_pretrained(model_name_or_path)

text = "<s>[INST] Conduct a laminar flow simulation around a cylinder using icoFoam. Apply an inlet velocity boundary condition, maintain a fixed zero pressure at the outlet, enforce a no-slip condition on the cylinder surface, and set zero-gradient conditions on the sides. Utilize a Newtonian fluid model with a kinematic viscosity of nu = 0.01 m^2/s. Initially, the velocity field inside the domain is (0.01, 0, 0) m/s, while the inlet velocity is (1, 0, 0) m/s. Control settings specify an endTime of 500 and a writeInterval of 40."

pipe = pipeline(task="text2text-generation", model=md, tokenizer=tk)
output = pipe(text)
print(output)

