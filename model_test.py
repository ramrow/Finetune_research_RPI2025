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

model_name_or_path =  "finalform/foamqwen2.5"

md = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)


tk = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

prompt = "You are an expert in OpenFOAM simulation and numerical modeling.Your task is to generate a complete and functional file named: <file_name>physicalProperties.water</file_name> within the <folder_name>constant</folder_name> directory. Before finalizing the output, ensure: - Ensure units and dimensions are correct** for all physical variables. - Ensure case solver settings are consistent with the user's requirements. Available solvers are: multiphaseEulerFoam. Provide only the code—no explanations, comments, or additional text."
user = "User requirement: Conduct a four-phase dam break simulation using multiphaseEulerFoam solver for water, oil, mercury, and air interaction. The domain is a rectangular tank with dimensions 4x4x0.1 units (scaled by convertToMeters=0.146), featuring a dam section at x=0.1461 containing water, followed by an oil section from x=0.1461 to x=0.2922, and a mercury section at the bottom from x=0 to x=0.1461 with height 0.1. The domain has no-slip boundary conditions on leftWall, rightWall, and lowerWall, with contact angles of 90 degrees between all fluid pairs. The atmosphere boundary is specified as an inlet-outlet condition with fixed pressure of 1e5 Pa. Initial temperature is set to 300K for all phases. Physical properties include: water (rho=1000 kg/m³, nu=1e-6 m²/s), oil (rho=500 kg/m³, nu=1e-6 m²/s), mercury (rho=13529 kg/m³, nu=1.125e-7 m²/s), and air (rho=1 kg/m³, nu=1.48e-5 m²/s). Surface tension coefficients are specified between all phase pairs (air-water: 0.07, air-oil: 0.032, air-mercury: 0.486, water-oil: 0.03, water-mercury: 0.415, oil-mercury: 0.4 N/m). The mesh consists of five blocks with grading (23x8x1, 19x8x1, 23x42x1, 4x42x1, 19x42x1). Use PIMPLE algorithm with 2 correctors, deltaT=0.0001s, endTime=6s, and writeInterval=0.02s. The simulation is laminar and includes gravity effects (g = 0 -9.81 0 m/s²). Just modify the necessary parts to make the file complete and functional.Please ensure that the generated file is complete, functional, and logically sound.Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence.When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."

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

pipe = pipeline(task="text-generation", model=md, tokenizer=tk, device_map={"":0})
output = pipe(messages, max_new_tokens=512)
print(output[0]['generated_text'][2]['content'])

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

