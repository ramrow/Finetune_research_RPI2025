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

model_name = "finalform/foamMistral0.3-7B-Instruct-trl"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

system = "You are an expert in OpenFOAM simulation and numerical modeling.Your task is to generate a complete and functional file named: <file_name>T.particles</file_name> within the <folder_name>0</folder_name> directory. Before finalizing the output, ensure: - Ensure units and dimensions are correct** for all physical variables. - Ensure case solver settings are consistent with the user's requirements. Available solvers are: multiphaseEulerFoam. Provide only the code—no explanations, comments, or additional text."
prompt = "User requirement: Perform a multiphase Euler-Euler simulation of particle-laden flow through a bent pipe using multiphaseEulerFoam solver. The domain consists of a pipe with diameter 28.28mm (inner diameter 19.8mm) that runs vertically for 400mm before transitioning through a 90-degree bend, followed by a horizontal section of 460mm length. Use convertToMeters value of 1e-3. The simulation involves water as continuous phase (density 997 kg/m3, viscosity 8.9e-4 m2/s) and particles as dispersed phase (density 1400 kg/m3) with initial volume fractions of 0.999 and 0.001 respectively. At inlet, set fixed velocity of (0 0.5 0) m/s for both phases, with water temperature of 300K. Use RNGkEpsilon turbulence model for water phase and phasePressure model for particles phase. The particle size distribution ranges from 1μm to 250μm divided into 30 groups with specified initial probability distribution. Apply no-slip condition for water and slip condition for particles at walls, with zeroGradient conditions at outlet. Set initial pressure to 1e5 Pa. Use PIMPLE algorithm with 3 outer correctors and 2 inner correctors. The mesh consists of 15 blocks with grading of 0.35 in radial direction. Run simulation from 0 to 2 seconds with adjustable timestep (initial 0.003s, maxCo=1) and write results every 0.1 seconds. Include population balance modeling for particle agglomeration using AdachiStuartFokkink coalescence model and Kusters breakup model. Please ensure that the generated file is complete, functional, and logically sound.Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence.When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
##############################################################################
##############################################################################
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

###########################################

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

