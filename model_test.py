from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

# model_name_or_path = "finalform/foam-nuTilda-sft-llama2-13B"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

model_name_or_path = "finalform/foamllama-7B"

md = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float32,
    device_map={"": 0}  # Automatically place model on available GPUs
)

tk = AutoTokenizer.from_pretrained(model_name_or_path)
# tk.pad_token = "[PAD]"
# tk.padding_side = "left"

prompt = "You are an expert in OpenFOAM simulation and numerical modeling. Your task is to generate a complete and functional file named: <file_name>epsilon</file_name> within the <folder_name>0</folder_name> directory. Ensure all required values are present and match with the files content already generated. Before finalizing the output, ensure: All necessary fields exist (e.g., if `nu` is defined in `constant/transportProperties`, it must be used correctly in `0/U`). Cross-check field names between different files to avoid mismatches. Ensure units and dimensions are correct for all physical variables. Ensure case solver settings are consistent with the user's requirements. Available solvers are: ['mhdFoam', 'rhoPorousSimpleFoam', 'foamyQuadMesh', 'laplacianFoam', 'rhoSimpleFoam', 'potentialFreeSurfaceFoam', 'simpleFoam', 'icoFoam', 'SRFPimpleFoam', 'compressibleInterFoam', 'XiFoam', 'financialFoam', 'interMixingFoam', 'rhoCentralFoam', 'pisoFoam', 'interFoam', 'foamyHexMesh', 'driftFluxFoam', 'multiphaseInterFoam', 'boundaryFoam', 'potentialFoam', 'compressibleMultiphaseInterFoam', 'moveDynamicMesh', 'cavitatingFoam', 'adjointShapeOptimisationFoam', 'electrostaticFoam', 'dsmcFoam', 'shallowWaterFoam', 'refineMesh', 'chtMultiRegionFoam', 'snappyHexMesh', 'porousSimpleFoam', 'multiphaseEulerFoam', 'chemFoam', 'SRFSimpleFoam', 'rhoPimpleFoam', 'reactingFoam', 'particleFoam', 'blockMesh', 'PDRFoam', 'buoyantReactingFoam', 'buoyantFoam', 'mdEquilibrationFoam', 'mdFoam', 'dnsFoam', 'solidDisplacementFoam', 'solidEquilibriumDisplacementFoam', 'pimpleFoam', 'twoLiquidMixingFoam', 'denseParticleFoam', 'scalarTransportFoam']. Provide only the code—no explanations, comments, or additional text."

text = "Conduct a RAS simulation for the turbulent flow over a backward-facing step using the pimpleFoam solver. Set the inlet velocity to 10 m/s and the outlet pressure to 0. The walls should be no-slip, and the front and back should be empty. Use the k-epsilon turbulence model for this simulation. The fluid is Newtonian with a kinematic viscosity of 1e-5 m²/s. The control settings are: endTime = 0.40 and writeInterval = 0.005. Please ensure that the generated file is complete, functional, and logically sound. Additionally, apply your domain expertise to verify that all numerical values are consistent with the user's requirements, maintaining accuracy and coherence."


pipe = pipeline(task="text-generation", model=md, tokenizer=tk, device_map={"":0})

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": text}
]

output = pipe(messages, max_new_tokens=2048)
print(output[0]['generated_text'][2]['content'])

