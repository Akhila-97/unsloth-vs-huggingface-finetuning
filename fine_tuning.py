from unsloth import FastLanguageModel
from datasets import load_dataset, DatasetDict
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import time

def print_vram(label):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved() / 1024**3
    peak      = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[VRAM] {label} -->> allocated: {allocated:.2f}GB | reserved: {reserved:.2f}GB | peak: {peak:.2f}GB")

class VRAMCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            print_vram(f"step {state.global_step}")

MODEL_NAME         = "unsloth/llama-3.2-3B-Instruct"
MAX_SEQUENCE_LENGTH = 2048
LOAD_IN_4BIT       = True
LORA_R             = 16
LORA_ALPHA         = 16
LORA_DROPOUT       = 0       

#  Load model 
load_start = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT
)
print(f"Model load time: {time.time() - load_start:.2f}s")
print_vram("after model load")

# Dataset 
TRAIN_DATA = "dataset/dataset/processed/train.jsonl"
full_dataset = load_dataset('json', data_files=TRAIN_DATA, split='train')

split       = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset  = split['test']   # 10% of train used as eval

print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

#  LoRA 
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print_vram("after LoRA setup")

# Trainer 
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    packing=False,
    dataset_text_field="text",
    max_seq_length=MAX_SEQUENCE_LENGTH,
    callbacks=[VRAMCallback()],
    args=TrainingArguments(
        output_dir="output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_8bit",
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        seed=42,
        report_to="none",
    )
)

#  Train 
print("training")
train_start = time.time()
trainer.train()
train_time = time.time() - train_start

print(f"Total train time: {train_time:.2f}s")
print(f"Seconds per step: {train_time / 75:.2f}s")
print_vram("after training")

#  Save
final_output_dir = "output/final"
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

#  Evaluate 
print("evaluation")
eval_result = trainer.evaluate()
print(f"Eval Loss: {eval_result['eval_loss']:.4f}")

#  Inference 
FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "system",
        "content": "You are an expert entity extraction system specialized in parsing resumes and job postings. Extract all relevant entities including names, skills, companies, education, designations, and other key information."
    },
    {
        "role": "user",
        "content": """Extract all entities from this resume text. Identify skills, names, companies, education, designations, and other relevant information.

John Doe, Senior Python Developer at Google. Skills: Python, Django, AWS, Docker, Kubernetes. Education: MIT, Computer Science. Email: john@example.com"""
    }
]

test_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")

print("\nTest Input:")
print(test_prompt)
print("\nModel Output:")

inference_start = time.time()
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    repetition_penalty=1.2,
    top_p=0.9,
    do_sample=True
)
print(f"Inference time: {time.time() - inference_start:.2f}s")
print_vram("after inference")

result = tokenizer.decode(outputs[0], skip_special_tokens=False)
if "<|start_header_id|>assistant<|end_header_id|>" in result:
    result = result.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    result = result.split("<|eot_id|>")[0].strip()

print(result)