from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import ollama

# https://huggingface.co/ayoubkirouane/moondream2-image-captcha
def model1(image):
    model_id = "ayoubkirouane/moondream2-image-captcha"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id , trust_remote_code=True)
    enc_image = model.encode_image(image)
    result = (model.answer_question(enc_image, "What does the text say?", tokenizer))
    # print(result)
    return result

# https://huggingface.co/google/paligemma-3b-pt-224
def model2(image):
    model_id = "google/paligemma-3b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    prompt = "Read what's written on the paper"
    # Directly encode the text using the __call__ method of the tokenizer
    model_inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True, truncation=True)

    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        # print(decoded)
        return decoded

def main(result, decoded):
    prompt = prompt = f"Please compare these two sentences and provide the corrected version of the sentence with higher accuracy in terms of grammar, spelling, and overall correctness:\n1 : {result}, 2 : {decoded}\n Return only the corrected text."

    
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    print(response["message"])


image =Image.open('image/p.jpg')
result = model1(image)
decoded  = model2(image)
main(result,decoded)