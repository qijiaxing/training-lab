# te_qwen2_vl.py
import os
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, AutoModelForVision2Seq, AutoConfig
import torch.nn.functional as F

from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import logging

import torch.distributed as dist
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer, Qwen2VLVisionBlock
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
import subprocess
from transformers.models.qwen2_vl.modeling_qwen2_vl import rotate_half


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
)

# transformer engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from transformer_engine.common import recipe

DEBUG = (int(os.getenv("DEBUG", 0)) > 0)
FP8_ENABLED = (int(os.getenv("FP8", 0)) > 0)
if FP8_ENABLED:
    print(f"FP8 enabled!!!")
else:
    print(f"FP8 NOT enabled! use bf16!")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# init dist
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
print("local_rank", local_rank)
print("world_size", world_size)
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)


# model_name = "Qwen/Qwen2-VL-2B-Instruct"
# revision = "895c3a49bc3fa70a340399125c650a463535e71c"
model_name = "Qwen/Qwen2-VL-7B-Instruct"
revision = "a28a094eb66a9f2ac70eef346f040d8a79977472"
# model_name = "Qwen/Qwen2-VL-72B-Instruct"
# revision = "f9b556a74d58e6d9915f73227c21045c87342b42"

dataset_id = "HuggingFaceM4/ChartQA"
processor = Qwen2VLProcessor.from_pretrained(
    model_name, revision=revision)


# Configuration
class Config:
    dataset_id = dataset_id
    output_dir = "/tmp_ckpt"
    batch_size = 16
    num_epochs = 1
    max_steps = 32
    learning_rate = 5e-6
    max_seq_length = 512
    # JQ: is LoRA used?
    lora_rank = 32
    lora_alpha = 64
    lora_dropout = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

system_message = (
    "You are a Vision Language Model specialized in interpreting visual data "
    "from chart images. Answer concisely."
)
def format_data(sample):

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]

def pad_text(inputs, labels):
    # JQ:
    if DEBUG:
        print(f"input shape: {inputs['input_ids'].shape} labels shape: {labels.shape}"
          f", attn mask: {inputs['attention_mask'].shape}"
          f", pixel_values: {inputs['pixel_values'].shape}"
          f", image_grid_thw: {inputs['image_grid_thw'].shape}")

    to_pad = int(8 - inputs['input_ids'].shape[1] % 8)
    p2d = (to_pad, 0) # Pad only the last dim, to the left, otherwise cause error

    inputs['input_ids'] = F.pad(inputs['input_ids'], p2d, "constant", 0)
    inputs['attention_mask'] = F.pad(inputs['attention_mask'], p2d, "constant", 0)
    labels = F.pad(labels, p2d, "constant", 0)

    return inputs, labels


def pad_image():
    # NV: pad image, not used for now
    to_pad = int(8 - (inputs['pixel_values'].shape[0] / 4) % 8) * 4
    if False:
        p2d = (0, 0, 0, to_pad)
        import torch.nn.functional as F
        padded = F.pad(inputs['pixel_values'], p2d, "constant", 0)
        inputs['pixel_values'] = padded
        print(f"After pad: {padded.shape}")


# Training function
def train_model(model, train_loader, optimizer, config, fp8_recipe):
    print("strat training model")
    model.train()
    total_steps = len(train_loader) * config.num_epochs

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=10),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof/7b-fp8-prefetch'),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
      for step, batch in enumerate(train_loader):
        inputs, labels = batch

        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        inputs, labels = pad_text(inputs, labels)

        # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
        # print all shape of inputs
        # input_ids shape: torch.Size([1, 171])
        # attention_mask shape: torch.Size([1, 171])
        # pixel_values shape: torch.Size([440, 1176])
        # image_grid_thw shape: torch.Size([1, 3])

        with te.fp8_autocast(enabled=FP8_ENABLED, fp8_recipe=fp8_recipe):
            outputs = model(**inputs, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        prof.step()   # torch profiler

        step += 1
        if step >= config.max_steps:
            print(f"Reached max steps {step}, stop!")
            break

        epoch = 0
        print(f"Epoch {epoch+1}/{config.num_epochs}, Step {step}/{total_steps}, Loss: {loss.item():.4f}")
        del loss


# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    return batch, labels

def _to_te(module: nn.Module):
    for name, child in list(module.named_children()):
        # 1) Linear → te.Linear
        if isinstance(child, nn.Linear) and child.in_features % 16 == 0 and child.out_features % 16 == 0:
            te_linear = te.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                params_dtype=torch.bfloat16,
            )
            te_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                te_linear.bias.data.copy_(child.bias.data)
            setattr(module, name, te_linear)

        # # 2) LayerNorm → te.LayerNorm
        # elif isinstance(child, nn.LayerNorm):
        #     te_ln = te.LayerNorm(
        #         normalized_shape=child.normalized_shape,
        #         eps=child.eps,
        #         params_dtype=torch.bfloat16,
        #     )
        #     te_ln.weight.data.copy_(child.weight.data)
        #     te_ln.bias.data.copy_(child.bias.data)
        #     setattr(module, name, te_ln)
        else:
            _to_te(child)

# Main function
def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    config = Config()

    # Load model and processor
    logger.info("Loading model and processor...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                )
    model = model.to(torch.bfloat16).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # NV: only apply fp8 on LM for now
    _to_te(model.language_model)

    # fp8 training recipe
    fp8_recipe = DelayedScaling(
        fp8_format=Format.HYBRID,    # E4M3 during forward pass, E5M2 during backward pass
        # fp8_format=Format.E4M3,    # E4M3 used everywhere
        amax_history_len=16,
    )

    # Load dataset
    logger.info("Loading dataset...")
    first_row = load_dataset(
        config.dataset_id,
        split="train[:10]",
        )[3]
    train_dataset = [first_row] * 32 * config.batch_size
    train_dataset = [format_data(sample) for sample in train_dataset]
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=1, #  factor * num_workers batches
        shuffle=False,
    )

    # Train
    logger.info("Starting training...")
    torch.cuda.empty_cache()
    train_model(model, train_dataloader, optimizer, config, fp8_recipe)


if __name__ == "__main__":
    main()
    destroy_process_group()
    logger.info("Training completed.")
