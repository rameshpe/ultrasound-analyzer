# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

processor = AutoProcessor.from_pretrained("chaoyinshe/EchoVLM_V2_lingshu_base_7b_instruct_preview")
model = AutoModelForVision2Seq.from_pretrained("chaoyinshe/EchoVLM_V2_lingshu_base_7b_instruct_preview")
model = model.to(torch.device("xpu"))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
