from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import time

MODEL_NAME          = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQUENCE_LENGTH = 2048
LORA_R              = 16
LORA_ALPHA          = 16
LORA_DROPOUT        = 0        

def print_vram(label):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved() / 1024**3
    peak      = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[VRAM] {label} ->> allocated: {allocated:.2f}GB | reserved: {reserved:.2f}GB | peak: {peak:.2f}GB")

class VRAMCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            print_vram(f"step {state.global_step}")

#  Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
)

# Load model
load_start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print(f"[METRIC] Model load time: {time.time() - load_start:.2f}s")
print_vram("after model load")

#  LoRA 
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print_vram("after LoRA setup")

# Dataset
TRAIN_DATA   = "dataset/dataset/processed/train.jsonl"
full_dataset = load_dataset('json', data_files=TRAIN_DATA, split='train')

split         = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset  = split['test']

print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

# Trainer 
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[VRAMCallback()],
    args=SFTConfig(
        dataset_text_field="text",
        output_dir="output_baseline",
        max_length=MAX_SEQUENCE_LENGTH,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
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

# Train 
print("training")
train_start = time.time()
trainer.train()
train_time  = time.time() - train_start

print(f"Total train time: {train_time:.2f}s")
print(f"Seconds per step: {train_time / 75:.2f}s")
print_vram("after training")

# Save
model.save_pretrained("output_baseline/final")
tokenizer.save_pretrained("output_baseline/final")

# Evaluate 
print("evaluation")
eval_result = trainer.evaluate()
print(f"[METRIC] Eval Loss: {eval_result['eval_loss']:.4f}")

# Inference
model.eval()

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
print(f"[METRIC] Inference time: {time.time() - inference_start:.2f}s")
print_vram("after inference")

result = tokenizer.decode(outputs[0], skip_special_tokens=False)
if "<|start_header_id|>assistant<|end_header_id|>" in result:
    result = result.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    result = result.split("<|eot_id|>")[0].strip()

print(result)