#!/usr/bin/env python3

import en_core_web_trf
import torch
from prompt_to_prompt_pipeline import Prompt2PromptPipeline
from demo_config import RunConfig1, RunConfig2
from prompt_utils import PromptParser
from typing import Dict, List
import spacy
import os
from datetime import datetime


def filter_text(token_indices, prompt_anchor):
    final_idx = []
    final_prompt = []
    for i, idx in enumerate(token_indices):
        if len(idx[1]) == 0:
            continue
        final_idx.append(idx)
        final_prompt.append(prompt_anchor[i])
    return final_idx, final_prompt

def process_prompts(prompts):
    token_indices_list = []
    prompt_anchors = []
    for prompt in prompts:
        nlp = en_core_web_trf.load()  # load spacy

        doc = nlp(prompt)
        prompt_parser.set_doc(doc)
        token_indices = prompt_parser._get_indices(prompt)
        prompt_anchor = prompt_parser._split_prompt(doc)

        token_indices_list.append(token_indices)
        prompt_anchors.append(prompt_anchor)

    return token_indices_list, prompt_anchors


if __name__ == "__main__":

    config = RunConfig2()

    # Create output directory if it doesn't exist
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == "cuda":
        pipe = Prompt2PromptPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                    torch_dtype=torch.float16, use_safetensors=True).to(device)
    else:
        pipe = Prompt2PromptPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                    torch_dtype=torch.float32, use_safetensors=False).to(device)
    seed = 864
    g_cpu = torch.Generator().manual_seed(seed)

    # TODO: Added from ToME
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    prompt_parser = PromptParser("stabilityai/stable-diffusion-xl-base-1.0")
    ##

    # prompts = ["a pink bear riding a bicycle on the beach", "a pink dragon riding a bicycle on the beach"]
    # cross_attention_kwargs = {"edit_type": "replace",
    #                           "n_self_replace": 0.4,
    #                           "n_cross_replace": {"default_": 1.0, "dragon": 0.4},
    #                           }

    # prompts = ["a chocolate cake and a lemon tart", "a confetti chocolate cake and a lemon tart"]
    # prompts = ["a white cat", "a sleeping white cat"]
    prompts = ["a white cat and a black dog", "a white cat with sunglasses and a black dog"]
    # prompt_merged_list = ["a cake and a tart", "a cake and a tart"]
    prompt_merged_list = ["a cat and a dog", "a cat and a dog"]
    # prompts = ["a chocolate cake", "a confetti chocolate cake"]
    # prompts = ["a chocolate cake", "a chocolate cake with confetti"]
    # prompt_merged_list = ["a cake", "a cake"]
    token_indices_list, prompt_anchors = process_prompts(prompts)

    print("###PROMPT ANCHORS: ", prompt_anchors)

    # prompts = ["a white cat", "a sleeping white cat"]
    cross_attention_kwargs = {"edit_type": "refine",
                              "n_self_replace": 0.4,
                              "n_cross_replace": {"default_": 1.0, "sunglasses": 0.8},
                              }

    image = pipe(
        prompts, 
        cross_attention_kwargs=cross_attention_kwargs,
        attention_res = config.attention_res, # add to config
        guidance_scale=config.guidance_scale, # add to config
        indices_to_alter_1=token_indices_list[0],
        indices_to_alter_2=token_indices_list[1],
        # indices_to_alter_1=config.token_indices_1,
        # indices_to_alter_2=config.token_indices_2,
        # prompt_anchor_1=prompt_anchors[0],
        # prompt_anchor_2=prompt_anchors[1],
        prompt_anchor_1=config.prompt_anchor_1,
        prompt_anchor_2=config.prompt_anchor_2,
        prompt_merged_1=prompt_merged_list[0],
        prompt_merged_2=prompt_merged_list[1],
        thresholds=config.thresholds, # add to config
        scale_factor=config.scale_factor, # add to config
        scale_range=config.scale_range, # add to config,
        run_standard_sd=config.run_standard_sd,
        token_refinement_steps=config.token_refinement_steps,
        attention_refinement_steps=config.attention_refinement_steps,
        tome_control_steps=config.tome_control_steps,
        eot_replace_step=config.eot_replace_step,
        negative_prompt="low res, ugly, blurry, artifact, unreal",
        generator=g_cpu
    )

    print(f"Num images: {len(image['images'])}")

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save each image with a unique filename
    for idx, img in enumerate(image['images']):
        filename = f"{output_dir}/image_{timestamp}_{idx}.png"
        img.save(filename)
        print(f"Saved image to: {filename}")

    from IPython.display import display

    for img in image['images']:
        display(img)