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

model_name_or_path =  "finalform/foamLlama3.1-8B-Instruct"

md = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)


tk = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

prompt = "You are an expert in OpenFOAM simulation and numerical modeling.Your task is to generate a complete and functional file named: <file_name>momentumTransport</file_name> within the <folder_name>constant</folder_name> directory. Before finalizing the output, ensure: - Ensure units and dimensions are correct** for all physical variables. - Ensure case solver settings are consistent with the user's requirements. Available solvers are: rhoCentralFoam. Provide only the code—no explanations, comments, or additional text."
text = "User requirement: Perform a compressible flow simulation using rhoCentralFoam solver for a forward-facing step geometry. The domain consists of three blocks: an inlet section (0.6x0.2), a vertical section (0.6x0.8), and a main channel (2.4x0.8), with a total depth of 0.1 (convertToMeters=1). Use a structured mesh with 48x16 cells for the inlet section, 48x64 cells for the vertical section, and 192x64 cells for the main channel. Set inlet conditions with a fixed velocity of (3 0 0) m/s, fixed pressure of 1 Pa, and temperature of 1 K. Apply symmetryPlane conditions for top and bottom boundaries, slip condition for the obstacle, and wave transmissive outlet condition. The simulation should run from 0 to 4 seconds with an initial timestep of 0.002s and adjustable timestepping (maxCo=0.2, maxDeltaT=1s), writing results every 0.1 seconds. Use laminar flow conditions with perfectGas equation of state (molWeight=11640.3), constant specific heat capacity Cp=2.5, and Prandtl number Pr=1. Implement the Kurganov flux scheme with vanLeer reconstruction for density and temperature, and vanLeerV for velocity. Just modify the necessary parts to make the file complete and functional.Please ensure that the generated file is complete, functional, and logically sound.Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence.When generating controlDict, do not include anything to preform post processing. Just include the necessary settings to run the simulation."

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": text}
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

