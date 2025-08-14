import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# from transformers import (
#     pipeline, 
#     AutoModelForCausalLM, 
#     AutoTokenizer,
#     AutoProcessor,
#     AutoModelForVision2Seq,
#     Qwen2_5_VLForConditionalGeneration
# )

model_name = "finalform/foamQwen2.5-7B-Coder-trl"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

ds = pd.read_csv("bernard.csv",)
code_content = []
for index, row in ds.iterrows():
    print(index)
    user = row.user_prompt
    system = row.system_prompt
    messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=0.7,
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
    code_content.append(content)

ds['code_content'] = code_content
ds.to_csv("bernardFOAM.csv", index=False)