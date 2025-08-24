# Finetune_research_RPI2025
Rensselaer Polytechnic Institute Research on AI Agent finetuning done under Professor Shaowu Pan, created an agent through finetuning capable of creating foamfiles for Openfoam software, software designed for computational fluid dynamics, to run through software such as Trl and llamafactory.

This project experimented with trl and llamafactory so it is recommended to use different environments in order to run each

The dataset used for this project can be found: [FoamData](https://huggingface.co/datasets/LeoYML/FoamGPT)
## Requirements
### Trl
Before you begin, please take note that all trl finetuning code is written and ran the following library versions(there is no guarantee if the code would still run with newer versions):

- **Python 3.12.7**: Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **Pip**: Python package installer. It usually comes with Python installations. You can check if you have it by running:
    ```bash
    pip --version
    ```
- **transformers 4.52.3**: A library for state-of-the-art natural language processing
- **trl 0.18.0**: A library for reinforcement learning with transformers
- **peft 0.17.0**: library for parameter-efficient fine-tuning
- **jinja2 3.1.4**: A templating engine for Python, used for rendering templates 
- **bitsandbytes 0.45.5**: A library for efficient training of large models
- **tf-keras 2.19.0**: A high-level API for building and training deep learning models with TensorFlow

If dependicies not installed then please utilize the following command:
```bash
    pip install transformers==4.52.3 trl==0.19.0 peft==0.17.0 jinja2==3.1.4 bitsandbytes==0.45.5 tf-keras==2.19.0
```
### Llamafactory
Before you begin, please take note that all llamafactory finetuning code is written and ran the following library versions(there is no guarantee if the code would still run with newer versions):

- **Python 3.12.0**: Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
- **Pip**: Python package installer. It usually comes with Python installations. You can check if you have it by running:
    ```bash
    pip --version
    ```
- **llamafactory 0.9.4.dev0**: library for building and training large language models efficiently(basically any version can work but the earlier ones might not be compatiable with the newer models)
- **torch 2.7.1**: A deep learning framework that provides a flexible platform for building and training neural networks
- **transformers 4.51.3**: A library for state-of-the-art natural language processing

If dependicies not installed then please utilize the following command:
```bash
pip install llamafactory torch==2.7.1 transformers==4.51.3
```

## Results
The research focused on using various models and training them all with both methods that yielded varying results:
### trl table
| Models | Pre-training loss | Post-training accuracy| Post-training Loss | Train runtime |
|------------------|---------------|-----------|---|---|
| **meta-llama/Llama-3.1-8B-Instruct** | 2.4526 | 0.9872  | 0.0438 | 7474.6963 |
| **Qwen/Qwen3-8B**        | 2.3461  | 0.9868 | 0.0434 | 6753.9399 |
| **mistralai/Mistral-7B-Instruct-v0.3** | 2.2221 | 0.9856 | 0.0649 | 6896.2251 |
| **Qwen/Qwen2.5-7B-Instruct** | 2.1476 | 0.9898 | 0.0428 | 9443.4287 |

All finetund trl models can be found here: [trlWriterAgents](https://huggingface.co/collections/finalform/trl-writer-agents-688fdc0b868242fe9c375461)
### llamafactory table
| Models | Pre-training loss | Post-training accuracy| Post-training Loss | Train runtime |
|------------------|---------------|-----------|---|---|
| **meta-llama/Llama-3.1-8B-Instruct** | 1.0263 | 0.9526 | 0.2624 | 2311.8159 |
| **Qwen/Qwen3-8B**        | 1.661 | 0.9448 | 0.2496 | 2751.4148 |
| **mistralai/Mistral-7B-Instruct-v0.3** | 1.4116 | 0.9587 | 0.226 | 2068.3584 |
| **ibm-granite/granite-3.3-8b-instruct** | 1.3811 | 0.9475 | 0.2523 | 2410.313 |

All finetuned llamafactory models can be fonud here: [llamafactoryWriterAgents](https://huggingface.co/collections/finalform/llamafactory-writer-agents-688c219ba46fe6624cdc7e80)

When comparing the results, it is clear as day that the llamafactory results are subpar when compared to the trl results in both the post training loss and pos training accuracy, however, llamafactory seemes to be much faster when it comes to handling the training when looking at the training runtime. 