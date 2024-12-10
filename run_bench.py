#!/usr/bin/env python3

# import en_core_web_trf
# import torch
# from prompt_to_prompt_pipeline import Prompt2PromptPipeline
# from demo_config_test import (
#     RunConfig3, RunConfig4, RunConfig5,
#     RunConfig6, RunConfig7, RunConfig8, RunConfig9, RunConfig10,
#     RunConfig11, RunConfig12, RunConfig13
# )
# from prompt_utils import PromptParser
# import os
# from datetime import datetime
# import logging
# from typing import List, Type, Any
# import traceback

# # Set up logging
# logging.basicConfig(
#     filename='pipeline_run.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# def setup_device_and_pipeline():
#     """Initialize the device and pipeline"""
#     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#     if device.type == "cuda":
#         pipe = Prompt2PromptPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-base-1.0",
#             torch_dtype=torch.float16,
#             use_safetensors=True
#         ).to(device)
#     else:
#         pipe = Prompt2PromptPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-base-1.0",
#             torch_dtype=torch.float32,
#             use_safetensors=False
#         ).to(device)
    
#     pipe.unet.requires_grad_(False)
#     pipe.vae.requires_grad_(False)
    
#     return device, pipe

# def process_config(config: Any, pipe: Prompt2PromptPipeline, prompt_parser: PromptParser, 
#                   output_dir: str, config_name: str, seed: int = 864):
#     """Process a single config"""
#     try:
#         logging.info(f"Processing {config_name}")
        
#         # Set up generator
#         g_cpu = torch.Generator().manual_seed(seed)
        
#         # Get prompts from config
#         prompts = [config.Original, config.Edit]
        
#         image = pipe(
#             prompts,
#             cross_attention_kwargs=config.cross_attention_kwargs,
#             attention_res=config.attention_res,
#             guidance_scale=config.guidance_scale,
#             indices_to_alter_1=config.token_indices_1,
#             indices_to_alter_2=config.token_indices_2,
#             prompt_anchor_1=config.prompt_anchor_1,
#             prompt_anchor_2=config.prompt_anchor_2,
#             prompt_merged_1=config.prompt_merged_list[0],
#             prompt_merged_2=config.prompt_merged_list[1],
#             thresholds=config.thresholds,
#             scale_factor=config.scale_factor,
#             scale_range=config.scale_range,
#             run_standard_sd=config.run_standard_sd,
#             token_refinement_steps=config.token_refinement_steps,
#             attention_refinement_steps=config.attention_refinement_steps,
#             tome_control_steps=config.tome_control_steps,
#             eot_replace_step=config.eot_replace_step,
#             negative_prompt="low res, ugly, blurry, artifact, unreal",
#             generator=g_cpu
#         )
        
#         # Generate timestamp for unique filenames
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Save each image with config-specific filename
#         for idx, img in enumerate(image['images']):
#             filename = f"{output_dir}/{config_name}_{timestamp}_{idx}.png"
#             img.save(filename)
#             logging.info(f"Saved image to: {filename}")
            
#         return True
        
#     except Exception as e:
#         logging.error(f"Error processing {config_name}: {str(e)}")
#         logging.error(traceback.format_exc())
#         return False

# def main():
#     # Create output directory
#     output_dir = "generated_images"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Initialize device and pipeline
#     device, pipe = setup_device_and_pipeline()
    
#     # Initialize prompt parser
#     prompt_parser = PromptParser("stabilityai/stable-diffusion-xl-base-1.0")
    
#     # List of all config classes
#     configs = [
#         (RunConfig3(), "RunConfig3"),
#         (RunConfig4(), "RunConfig4"),
#         (RunConfig5(), "RunConfig5"),
#         (RunConfig6(), "RunConfig6"),
#         (RunConfig7(), "RunConfig7"),
#         (RunConfig8(), "RunConfig8"),
#         (RunConfig9(), "RunConfig9"),
#         (RunConfig10(), "RunConfig10"),
#         (RunConfig11(), "RunConfig11"),
#         (RunConfig12(), "RunConfig12"),
#         (RunConfig13(), "RunConfig13"),
#     ]
    
#     # Process each config
#     successful_configs = 0
#     failed_configs = 0
    
#     for config, config_name in configs:
#         success = process_config(config, pipe, prompt_parser, output_dir, config_name)
#         if success:
#             successful_configs += 1
#         else:
#             failed_configs += 1
    
#     # Log final statistics
#     logging.info(f"Processing completed. Successful configs: {successful_configs}, Failed configs: {failed_configs}")
#     print(f"Processing completed. Successful configs: {successful_configs}, Failed configs: {failed_configs}")
#     print(f"Check pipeline_run.log for detailed information")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3

import en_core_web_trf
import torch
from prompt_to_prompt_pipeline import Prompt2PromptPipeline
import demo_config_test
from prompt_utils import PromptParser
import os
from datetime import datetime
import logging
from typing import List, Type, Any
import traceback
import inspect
from dataclasses import is_dataclass

# Set up logging
logging.basicConfig(
    filename='pipeline_run.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_all_configs():
    """Discover all dataclass configs in the demo_config_test module"""
    configs = []
    for name, obj in inspect.getmembers(demo_config_test):
        # Check if it's a dataclass and its name starts with RunConfig
        if is_dataclass(obj) and name.startswith('RunConfig'):
            try:
                config_instance = obj()
                configs.append((config_instance, name))
                logging.info(f"Discovered config class: {name}")
            except Exception as e:
                logging.error(f"Error instantiating {name}: {str(e)}")
    
    # Sort by config name to ensure consistent ordering
    configs.sort(key=lambda x: x[1])
    return configs

def setup_device_and_pipeline():
    """Initialize the device and pipeline"""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == "cuda":
        pipe = Prompt2PromptPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)
    else:
        pipe = Prompt2PromptPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            use_safetensors=False
        ).to(device)
    
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    
    return device, pipe

def process_config(config: Any, pipe: Prompt2PromptPipeline, prompt_parser: PromptParser, 
                  output_dir: str, config_name: str, seed: int = 864):
    """Process a single config"""
    try:
        logging.info(f"Processing {config_name}")
        
        # Set up generator
        g_cpu = torch.Generator().manual_seed(seed)
        
        # Get prompts from config
        prompts = [config.Original, config.Edit]
        
        image = pipe(
            prompts,
            cross_attention_kwargs=config.cross_attention_kwargs,
            attention_res=config.attention_res,
            guidance_scale=config.guidance_scale,
            indices_to_alter_1=config.token_indices_1,
            indices_to_alter_2=config.token_indices_2,
            prompt_anchor_1=config.prompt_anchor_1,
            prompt_anchor_2=config.prompt_anchor_2,
            prompt_merged_1=config.prompt_merged_list[0],
            prompt_merged_2=config.prompt_merged_list[1],
            thresholds=config.thresholds,
            scale_factor=config.scale_factor,
            scale_range=config.scale_range,
            run_standard_sd=config.run_standard_sd,
            token_refinement_steps=config.token_refinement_steps,
            attention_refinement_steps=config.attention_refinement_steps,
            tome_control_steps=config.tome_control_steps,
            eot_replace_step=config.eot_replace_step,
            negative_prompt="low res, ugly, blurry, artifact, unreal",
            generator=g_cpu
        )
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each image with config-specific filename
        for idx, img in enumerate(image['images']):
            filename = f"{output_dir}/{config_name}_{timestamp}_{idx}.png"
            img.save(filename)
            logging.info(f"Saved image to: {filename}")
            
        return True
        
    except Exception as e:
        logging.error(f"Error processing {config_name}: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def main():
    # Create output directory
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize device and pipeline
    device, pipe = setup_device_and_pipeline()
    
    # Initialize prompt parser
    prompt_parser = PromptParser("stabilityai/stable-diffusion-xl-base-1.0")
    
    # Automatically discover and load all configs
    configs = get_all_configs()
    logging.info(f"Discovered {len(configs)} config classes")
    
    # Process each config
    successful_configs = 0
    failed_configs = 0
    
    for config, config_name in configs:
        success = process_config(config, pipe, prompt_parser, output_dir, config_name)
        if success:
            successful_configs += 1
        else:
            failed_configs += 1
    
    # Log final statistics
    logging.info(f"Processing completed. Successful configs: {successful_configs}, Failed configs: {failed_configs}")
    print(f"Processing completed. Successful configs: {successful_configs}, Failed configs: {failed_configs}")
    print(f"Check pipeline_run.log for detailed information")

if __name__ == "__main__":
    main()