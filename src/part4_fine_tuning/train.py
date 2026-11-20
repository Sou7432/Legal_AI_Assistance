"""
=============================================================
PART 4 — LLM Fine-Tuning with QLoRA
=============================================================
File: src/part4_fine_tuning/train.py

This script fine-tunes a Mistral-7B (or Llama-3-8B) model
on the 10,000-row Indian legal dataset using QLoRA.

HARDWARE GUIDE:
  ─ Full fine-tuning (all weights):
    Requires 2×80GB A100 GPUs. NOT suitable for home hardware.

  ─ LoRA (Low-Rank Adaptation):
    Trains only a small set of adapter weights.
    Requires ~24GB VRAM (single RTX 3090/4090).

  ─ QLoRA (Quantized LoRA):
    Loads the model in 4-bit, then applies LoRA.
    Requires only ~12-16GB VRAM.
    THIS IS WHAT WE USE — works on RTX 3080/3090/4080/4090.
    Even works on 8GB VRAM with smaller models (Phi-3-mini 3.8B).

HOW TO RUN:
    python src/part4_fine_tuning/train.py

STEP-BY-STEP EXPLANATION:
  1. Load pretrained Mistral-7B in 4-bit (QLoRA)
  2. Apply LoRA adapters (only ~1% of parameters are trained)
  3. Run supervised fine-tuning using TRL's SFTTrainer
  4. Save checkpoints every N steps
  5. Monitor validation loss to detect overfitting
  6. Save final model and tokenizer
=============================================================
"""

import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# Change MODEL_NAME to use a different base model:
#   "mistralai/Mistral-7B-Instruct-v0.2"  ← recommended (7B, good for legal)
#   "meta-llama/Meta-Llama-3-8B-Instruct"  ← requires Meta license
#   "microsoft/Phi-3-mini-4k-instruct"     ← 3.8B, runs on 8GB VRAM
MODEL_NAME      = "mistralai/Mistral-7B-Instruct-v0.2"

DATASET_PATH    = Path("data/dataset/hf_tokenized")  # from Part 3
OUTPUT_DIR      = Path("models/legal_llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── QLoRA Configuration ───
LORA_R          = 16     # LoRA rank — higher = more parameters, more capacity
LORA_ALPHA      = 32     # LoRA scaling factor (usually 2× rank)
LORA_DROPOUT    = 0.05
# Target modules to apply LoRA on (Mistral / Llama attention layers)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# ─── Training Hyperparameters ───
# EXPLANATION:
#  learning_rate: How fast the model updates weights.
#    Too high → unstable training. Too low → slow/no learning.
#    3e-4 is a standard starting point for LoRA.
#
#  per_device_train_batch_size: Examples processed per GPU per step.
#    Smaller = uses less VRAM. Increase if you have enough VRAM.
#
#  gradient_accumulation_steps: Simulates a larger batch size without
#    more VRAM. Effective batch = batch_size × accumulation_steps.
#    Here: 4 × 4 = 16 effective batch size.
#
#  num_train_epochs: Number of full passes through the training data.
#    3 epochs is a good starting point. Too many → overfitting.
#
#  warmup_ratio: Fraction of steps for learning rate warmup.
#    Helps stabilize early training.
LEARNING_RATE   = 3e-4
BATCH_SIZE      = 4
GRAD_ACCUM      = 4
NUM_EPOCHS      = 3
WARMUP_RATIO    = 0.03
MAX_SEQ_LEN     = 2048
SAVE_STEPS      = 100
EVAL_STEPS      = 100
LOGGING_STEPS   = 25


# ─────────────────────────────────────────────
# STEP 1–2: LOAD MODEL IN 4-BIT (QLoRA)
# ─────────────────────────────────────────────
def load_model_and_tokenizer():
    """Load the pretrained model in 4-bit quantization for QLoRA."""

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # nested quantization (saves a bit more VRAM)
        bnb_4bit_quant_type="nf4",        # NF4 = best quality for 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16  # bf16 compute for faster training
    )

    log.info(f"Loading model: {MODEL_NAME} (4-bit QLoRA)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",           # automatically uses your GPU
        trust_remote_code=True
    )

    # Prepare model for k-bit (quantized) training
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # pad on the right for causal LM

    return model, tokenizer


# ─────────────────────────────────────────────
# STEP 3: APPLY LORA ADAPTERS
# ─────────────────────────────────────────────
def apply_lora(model):
    """
    Apply LoRA adapters to the model.
    Only the LoRA adapter weights will be trained — the original
    model weights remain frozen (this is what makes it efficient).
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"LoRA applied. Trainable params: {trainable_params:,} "
        f"({100 * trainable_params / all_params:.2f}% of total {all_params:,})"
    )
    return model


# ─────────────────────────────────────────────
# STEP 4–9: TRAINING
# ─────────────────────────────────────────────
def train(model, tokenizer, dataset_dict):
    """Configure and run the supervised fine-tuning process."""

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        lr_scheduler_type="cosine",        # cosine decay schedule
        fp16=False,                        # use bf16 instead on modern GPUs
        bf16=True,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,       # keep the checkpoint with lowest val loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,                # keep only last 3 checkpoints
        report_to="none",                  # change to "wandb" if you use W&B
        gradient_checkpointing=True,       # saves VRAM at cost of some speed
        optim="paged_adamw_32bit",         # QLoRA-optimized optimizer
        dataloader_pin_memory=False
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        max_seq_length=MAX_SEQ_LEN,
        packing=False,                     # set True to pack short examples together
    )

    log.info("Starting training…")
    trainer.train()

    # Step 10: Save the final fine-tuned model
    final_model_path = OUTPUT_DIR / "final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    log.info(f"✓ Model saved → {final_model_path}")

    return trainer


# ─────────────────────────────────────────────
# STEP 11–12: LOAD MODEL FOR INFERENCE
# ─────────────────────────────────────────────
def load_for_inference(model_path: str):
    """
    Load the fine-tuned model for generating answers.

    HOW INFERENCE WORKS:
    1. User provides only their legal problem (the "problem" field)
    2. We wrap it in the instruction prompt template
    3. The model generates all 7 output fields as a JSON object
    4. We parse the JSON to extract: law, section, solution, etc.
    """
    from peft import PeftModel

    log.info(f"Loading fine-tuned model from {model_path}…")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()   # merge LoRA weights into base for faster inference

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def generate_legal_answer(problem: str, model, tokenizer, max_new_tokens: int = 512) -> dict:
    """
    Given a user problem in everyday language, generate the full legal analysis.
    Returns a dict with keys: law, section, solution, solution_based_on_law,
                              risk_category, applicable_conditions, expected_answer
    """
    try:
        from src.constants import INSTRUCTION
    except ImportError:
        from constants import INSTRUCTION
    import json

    prompt = (
        f"[INST] {INSTRUCTION}\n\n"
        f"User Problem: {problem} [/INST]\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,     # low temperature = more consistent, accurate answers
            do_sample=False,     # greedy decoding for factual legal tasks
            repetition_penalty=1.1
        )

    output_text = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Parse the JSON output
    try:
        result = json.loads(output_text.strip())
    except json.JSONDecodeError:
        # If JSON parsing fails, extract fields with regex
        result = {"raw_output": output_text, "parse_error": True}

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Load the tokenized dataset prepared in Part 3
    if not DATASET_PATH.exists():
        log.error(
            f"Dataset not found at {DATASET_PATH}. "
            "Please run Part 2 and Part 3 first."
        )
        return

    log.info("Loading dataset…")
    dataset_dict = load_from_disk(str(DATASET_PATH))

    # Load model in 4-bit QLoRA
    model, tokenizer = load_model_and_tokenizer()

    # Apply LoRA adapters
    model = apply_lora(model)

    # Train
    trainer = train(model, tokenizer, dataset_dict)

    log.info("\n" + "="*55)
    log.info("FINE-TUNING COMPLETE")
    log.info(f"Best model checkpoint saved in: {OUTPUT_DIR}")
    log.info("Run Part 5 to build the RAG system.")
    log.info("Run Part 6 to evaluate model accuracy.")


if __name__ == "__main__":
    main()
