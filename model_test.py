from transformers import (
    pipeline, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    Qwen2_5_VLForConditionalGeneration
)
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "finalform/foamQwen2.5-Coder-7B-Instruct"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
system = "You are an expert in OpenFOAM simulation and numerical modeling.Your task is to generate a complete and functional file named: <file_name>blockMeshDict</file_name> within the <folder_name>system</folder_name> directory. Before finalizing the output, ensure:\n- Ensure units and dimensions are correct** for all physical variables.\n- Ensure case solver settings are consistent with the user's requirements. Available solvers are: ['SRFPimpleFoam', 'shallowWaterFoam', 'solidEquilibriumDisplacementFoam', 'driftFluxFoam', 'solidDisplacementFoam', 'twoLiquidMixingFoam', 'compressibleMultiphaseInterFoam', 'dsmcFoam', 'mhdFoam', 'foamyHexMesh', 'reactingFoam', 'multiphaseInterFoam', 'chtMultiRegionFoam', 'denseParticleFoam', 'rhoCentralFoam', 'icoFoam', 'electrostaticFoam', 'interMixingFoam', 'interFoam', 'chemFoam', 'cavitatingFoam', 'buoyantReactingFoam', 'dnsFoam', 'pimpleFoam', 'multiphaseEulerFoam', 'laplacianFoam', 'financialFoam', 'simpleFoam', 'compressibleInterFoam', 'scalarTransportFoam', 'particleFoam', 'mdEquilibrationFoam', 'refineMesh', 'PDRFoam', 'buoyantFoam', 'SRFSimpleFoam', 'rhoSimpleFoam', 'potentialFoam', 'blockMesh', 'XiFoam', 'rhoPimpleFoam', 'boundaryFoam', 'rhoPorousSimpleFoam', 'snappyHexMesh', 'adjointShapeOptimisationFoam', 'mdFoam', 'foamyQuadMesh', 'pisoFoam', 'porousSimpleFoam', 'moveDynamicMesh', 'potentialFreeSurfaceFoam'].\nProvide only the code-no explanations, comments, or additional text."
# prompt = "User requirement: Perform a 3D Bernard Cell simulation using OpenFOAM buoyantFoam solver. The computational domain spans 9 m x 1 m x 2 m. The simulation begins at t=0 seconds and runs until t=1000 seconds with a time step of 1 second, and results are written at intervals of every 50 seconds. One wall has a temperature of 301 K, while the other has a temperature of 300 K.\nPlease ensure that the generated file is complete, functional, and logically sound.Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."
with open("temp.txt", 'r') as file:
    prompt = file.read()
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    temperature=0.5,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
print( thinking_content)
print( content)

#################################################################################
#################################################################################


# model_name = "finalform/foamQwen3-8B-trl"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# system = "You are an expert in OpenFOAM simulation and numerical modeling.Your task is to generate a complete and functional file named: <file_name>controlDict</file_name> within the <folder_name>system</folder_name> directory. Before finalizing the output, ensure: - Ensure units and dimensions are correct** for all physical variables. - Ensure case solver settings are consistent with the user's requirements. Available solvers are: rhoPorousSimpleFoam. Provide only the code—no explanations, comments, or additional text."
# prompt = "User requirement: Perform a compressible flow simulation through an angled duct with porous media using rhoPorousSimpleFoam solver. The geometry consists of a 45-degree angled duct with three sections: inlet (150 length), porous region (100 length), and outlet (100 length), with a width of 50 units and convertToMeters factor of 0.001. Use k-epsilon RAS turbulence model with standard wall functions. Set inlet with turbulent boundary layer profile and mass flow rate of 0.1 kg/s, outlet with fixed pressure of 1e5 Pa, no-slip condition on walls, and slip condition on porous walls. Initial conditions: temperature 293K, pressure 1e5 Pa, zero velocity field, k=1 m²/s², epsilon=200 m²/s³. Physical properties: air with molecular weight 28.9, Cp=1005 J/kgK, dynamic viscosity 1.82e-5 kg/m·s, Prandtl number 0.71. Mesh consists of 15x20x20 cells in inlet section, 20x20x20 in porous section, and 20x20x20 in outlet section. Porous region parameters: Darcy coefficient d=(5e7 -1000 -1000) and Forchheimer coefficient f=(0 0 0), with coordinate system rotated 45 degrees. Use SIMPLE algorithm with 2 U-correctors, relaxation factors: p=0.3, U=0.7, k/epsilon=0.9. Run steady-state simulation for 100 time units with writeInterval of 10 timesteps. Please ensure that the generated file is complete, functional, and logically sound.Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence.When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."
# messages = [
#     {"role": "system", "content": system},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True,

# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=1024,
#     temperature=0.6,
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)

#################################################################################
#################################################################################
"""
model_name_or_path =  "finalform/foamMistral0.3-7B-Instruct-trl"

md = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

tk = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

prompt = "You are an expert in OpenFOAM simulation and numerical modeling.Your task is to generate a complete and functional file named: <file_name>fvSolution</file_name> within the <folder_name>system</folder_name> directory. Before finalizing the output, ensure: - Ensure units and dimensions are correct** for all physical variables. - Ensure case solver settings are consistent with the user's requirements. Available solvers are: interFoam. Provide only the code—no explanations, comments, or additional text."
user = "User requirement: Perform a multiphase water channel simulation using interFoam solver with k-omega-SST turbulence model. The domain is a 3D channel with dimensions 20x20x6 units (convertToMeters=1). The channel geometry consists of three blocks: a lower section (20x7x6), middle section (20x6x6), and upper section (20x7x6). Initial water level is set at z=2.2 units, with water below and air above. Inlet boundary has a fixed volumetric flow rate of 50 m³/s, walls have no-slip condition, atmosphere patch uses prghTotalPressure with p0=0, and outlet has inletOutlet velocity condition. Use PIMPLE algorithm with 2 correctors and no non-orthogonal corrections. Physical properties include: water (rho=1000 kg/m³, nu=1e-06 m²/s), air (rho=1 kg/m³, nu=1.48e-05 m²/s), and surface tension coefficient of 0.07 N/m. Gravity is set to (0,0,-9.81) m/s². The mesh consists of three blocks with grading (20,5,20), (20,10,20), and (20,5,20) cells respectively. Run simulation from t=0 to t=200s with initial deltaT=0.1s, adjustable timestep with maxCo=6, and write results every 5 seconds. Initial turbulence conditions: k=0.0001 m²/s², omega=0.003 1/s, with 5% intensity at inlet. Just modify the necessary parts to make the file complete and functional.Please ensure that the generated file is complete, functional, and logically sound.Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence.When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": user}
]

text = tk.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

model_inputs = tk([text], return_tensors="pt").to(md.device)
generated_ids = md.generate(
    **model_inputs,
    max_new_tokens=1028,
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tk.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tk.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print(content)

"""
#################################################################################
#################################################################################

# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tk.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)

# output = pipe(messages, )
# result = output[0]['generated_text'][-1]['content']
# print(result)

#################################################################################
#################################################################################

# print(len(output))
# print(result)
# print(len(result))
# names = ['allrun', '0/U', 'constant/transportProperties', 'constant/turbulenceProperties', '0/s', '0/sigma', 'constant/fvOptions', '0/omega', 'constant/MRFProperties', '0/k', 'system/fvSchemes', '0/nut', '0/p', '0/epsilon', 'system/controlDict', 'system/fvSolution', 'constant/dynamicMeshDict', '0/nuTilda', 'system/topoSetDict']
# for n in names:
#     if not (n in result):
#         print(n)
# md = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     # torch_dtype="auto",
#     torch_dtype=torch.float16,
#     device_map={"": 0}  # Automatically place model on available GPUs
# )

# tk = AutoTokenizer.from_pretrained(model_name_or_path)
# # tk.pad_token = "[PAD]"
# # tk.padding_side = "left"

# # prompt = "You are an expert in OpenFOAM simulation and numerical modeling. Your task is to generate a complete and functional file named: <file_name>transportProperties</file_name> within the <folder_name>constant</folder_name> directory. Ensure all required values are present and match with the files content already generated. Before finalizing the output, ensure: All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`). Cross-check field names between different files to avoid mismatches. Ensure units and dimensions are correct for all physical variables. Ensure case solver settings are consistent with the user's requirements. Available solvers are: ['mhdFoam', 'rhoPorousSimpleFoam', 'foamyQuadMesh', 'laplacianFoam', 'rhoSimpleFoam', 'potentialFreeSurfaceFoam', 'simpleFoam', 'icoFoam', 'SRFPimpleFoam', 'compressibleInterFoam', 'XiFoam', 'financialFoam', 'interMixingFoam', 'rhoCentralFoam', 'pisoFoam', 'interFoam', 'foamyHexMesh', 'driftFluxFoam', 'multiphaseInterFoam', 'boundaryFoam', 'potentialFoam', 'compressibleMultiphaseInterFoam', 'moveDynamicMesh', 'cavitatingFoam', 'adjointShapeOptimisationFoam', 'electrostaticFoam', 'dsmcFoam', 'shallowWaterFoam', 'refineMesh', 'chtMultiRegionFoam', 'snappyHexMesh', 'porousSimpleFoam', 'multiphaseEulerFoam', 'chemFoam', 'SRFSimpleFoam', 'rhoPimpleFoam', 'reactingFoam', 'particleFoam', 'blockMesh', 'PDRFoam', 'buoyantReactingFoam', 'buoyantFoam', 'mdEquilibrationFoam', 'mdFoam', 'dnsFoam', 'solidDisplacementFoam', 'solidEquilibriumDisplacementFoam', 'pimpleFoam', 'twoLiquidMixingFoam', 'denseParticleFoam', 'scalarTransportFoam']. Provide only the code—no explanations, comments, or additional text."
# prompt = "You are an expert in OpenFOAM simulation and numerical modeling. Your task is to generate a complete and functional file named: <file_name>epsilon</file_name> within the <folder_name>0</folder_name> directory. Ensure all required values are present and match with the files content already generated. Before finalizing the output, ensure: All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`). Cross-check field names between different files to avoid mismatches. Ensure units and dimensions are correct for all physical variables. Ensure case solver settings are consistent with the user's requirements. Available solvers are: ['mhdFoam', 'rhoPorousSimpleFoam', 'foamyQuadMesh', 'laplacianFoam', 'rhoSimpleFoam', 'potentialFreeSurfaceFoam', 'simpleFoam', 'icoFoam', 'SRFPimpleFoam', 'compressibleInterFoam', 'XiFoam', 'financialFoam', 'interMixingFoam', 'rhoCentralFoam', 'pisoFoam', 'interFoam', 'foamyHexMesh', 'driftFluxFoam', 'multiphaseInterFoam', 'boundaryFoam', 'potentialFoam', 'compressibleMultiphaseInterFoam', 'moveDynamicMesh', 'cavitatingFoam', 'adjointShapeOptimisationFoam', 'electrostaticFoam', 'dsmcFoam', 'shallowWaterFoam', 'refineMesh', 'chtMultiRegionFoam', 'snappyHexMesh', 'porousSimpleFoam', 'multiphaseEulerFoam', 'chemFoam', 'SRFSimpleFoam', 'rhoPimpleFoam', 'reactingFoam', 'particleFoam', 'blockMesh', 'PDRFoam', 'buoyantReactingFoam', 'buoyantFoam', 'mdEquilibrationFoam', 'mdFoam', 'dnsFoam', 'solidDisplacementFoam', 'solidEquilibriumDisplacementFoam', 'pimpleFoam', 'twoLiquidMixingFoam', 'denseParticleFoam', 'scalarTransportFoam']. Provide only the code—no explanations, comments, or additional text."

# # text = "Conduct a RAS simulation for the turbulent flow over a backward-facing step using the pimpleFoam solver. Set the inlet velocity to 10 m/s and the outlet pressure to 0. The walls should be no-slip, and the front and back should be empty. Use the k-epsilon turbulence model for this simulation. The fluid is Newtonian with a kinematic viscosity of 1e-5 m²/s. The control settings are: endTime = 0.40 and writeInterval = 0.005. Please ensure that the generated file is complete, functional, and logically sound. Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."
# text = "Conduct a RAS simulation for the turbulent flow over a backward-facing step using the pimpleFoam solver. Set the inlet velocity to 10 m/s and the outlet pressure to 0. The walls should be no-slip, and the front and back should be empty. Use the k-epsilon turbulence model for this simulation. The fluid is Newtonian with a kinematic viscosity of 1e-5 m²/s. The control settings are: endTime = 0.40 and writeInterval = 0.005. Please ensure that the generated file is complete, functional, and logically sound. Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."

# pipe = pipeline(task="text-generation", model=md, tokenizer=tk, device_map={"":0})

# messages = [
#     {"role": "system", "content": prompt},
#     {"role": "user", "content": text}
# ]

# output = pipe(messages, max_new_tokens=512)
# print(output[0]['generated_text'][2]['content'])

#################################################################################
#################################################################################