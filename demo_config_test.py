from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# AAAAAAAA-----------------
        


@dataclass
class RunConfig3:

    Original = "a slanted mountain bicycle on the road in front of a building"
    token_indices_1: List[int] = field(default_factory=lambda: [[[2], [3, 4]], [[6], [7]]])
    Edit = "a slanted rusty mountain bicycle on the road in front of a building"
    token_indices_2: List[int] = field(default_factory=lambda: [[[2], [3, 4, 5]], [[6], [7]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a slanted mountain bicycle"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a slanted rusty mountain bicycle"
            ]
        )
    prompt_merged_list = ["a bicycle on the road in front of a building", "a bicycle on the road in front of a building"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "rusty": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)



@dataclass
class RunConfig4:

    Original = "a round cake with orange frosting on a wooden plate"
    token_indices_1: List[int] = field(default_factory=lambda: [[[2], [3, 5, 6]], [[9], [10]]])
    Edit = "a square cake with orange frosting on a wooden plate"
    token_indices_2: List[int] = field(default_factory=lambda: [[[2], [3, 5, 6]], [[9], [10]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a square cake with orange frosting",
                "a wooden plate"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a round cake with orange frosting",
                "a wooden plate"
            ]
        )
    prompt_merged_list = ["a cake on a plate", "a cake on a plate"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "square": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)






@dataclass
class RunConfig5:

    Original = "a bowl of strawberries and blueberries on a striped tablecloth"
    token_indices_1: List[int] = field(default_factory=lambda: [[[2], [4, 6]], [[9], [10]]])
    Edit = "a bowl of strawberries and blueberries and a lemon on a striped tablecloth"
    token_indices_2: List[int] = field(default_factory=lambda: [[[2], [4, 6, 9]], [[12], [13]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a bowl of strawberries and blueberries",
                "a striped tablecloth"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a bowl of strawberries and blueberries and a lemon",
                "a striped tablecloth"
            ]
        )
    prompt_merged_list = ["a bowl on a tablecloth", "a bowl on a tablecloth"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "lemon": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)




@dataclass
class RunConfig6:

    Original = "a sheep with pink and blue hair standing on a colorful background"
    token_indices_1: List[int] = field(default_factory=lambda: [[[2], [4, 6, 7]], [[11], [12]]])
    Edit = "a sheep with pink and blue hair standing on a colorful background with flowers around"
    token_indices_2: List[int] = field(default_factory=lambda: [[[2], [4, 6, 9]], [[11], [12, 14]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a sheep with pink and blue hair",
                "a colorful background"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a sheep with pink and blue hair",
                "a colorful background with flowers around"
            ]
        )
    prompt_merged_list = ["a sheep standing on a background", "a sheep standing on a background"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "flowers": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)



@dataclass
class RunConfig7:

    Original = "a man reading books in front of a white wall"
    token_indices_1: List[int] = field(default_factory=lambda: [[[2], [4]], [[9], [10]]])
    Edit = "a man reading books in front of a white wall with a painting of a heart"
    token_indices_2: List[int] = field(default_factory=lambda: [[[2], [4]], [[9], [10, 13, 14]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a man reading books",
                "a white wall"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a man reading books",
                "a white wall with a painting of a heart"
            ]
        )
    prompt_merged_list = ["a man in front of a wall", "a man in front of a wall"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "painting of a heart": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)




@dataclass
class RunConfig8:

    Original = "a rabbit sitting in front of colorful eggs"
    token_indices_1: List[int] = field(default_factory=lambda: [[[2], [3]], [[7], [8]]])
    Edit = "a rabbit with a dress sitting in front of colorful eggs"
    token_indices_2: List[int] = field(default_factory=lambda: [[[2], [5, 6]], [[10], [11]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a rabbit sitting",
                "colorful eggs"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a rabbit with a dress sitting",
                "colorful eggs"
            ]
        )
    prompt_merged_list = ["a rabbit sitting in front of eggs", "a rabbit sitting in front of eggs"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "dress": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)




@dataclass
class RunConfig9:

    Original = "mozart portrait"
    token_indices_1: List[int] = field(default_factory=lambda: [[[1], [2]]])
    Edit = "mozart with hat portrait"
    token_indices_2: List[int] = field(default_factory=lambda: [[[1], [3, 4]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "mozart portrait",
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "mozart with hat portrait",
            ]
        )
    prompt_merged_list = ["mozart portrait", "mozart portrait"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "hat": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)




@dataclass
class RunConfig10:

    Original = "a dog looking at the camera"
    token_indices_1: List[int] = field(default_factory=lambda: [[[2], []]])
    Edit = "a dog with a red dog collar looking at the camera"
    token_indices_2: List[int] = field(default_factory=lambda: [[[2], [5, 7, 11]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a dog looking at the camera"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a dog with a red dog collar looking at the camera"
            ]
        )
    prompt_merged_list = ["a dog looking at the camera", "a dog looking at the camera"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "red dog collar": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)




@dataclass
class RunConfig11:

    Original = "a cat standing on fence"
    token_indices_1: List[int] = field(default_factory=lambda: [[[2], []]])
    Edit = "a cat wearing hat standing on fence"
    token_indices_2: List[int] = field(default_factory=lambda: [[[2], [3, 4]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a cat"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a cat wearing hat"
            ]
        )
    prompt_merged_list = ["a cat standing on fence", "a cat standing on fence"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "hat": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)




@dataclass
class RunConfig12:

    Original = "moose in the forest"
    token_indices_1: List[int] = field(default_factory=lambda: [[[1], []], [[4], []]])
    Edit = "moose in rain in the forest"
    token_indices_2: List[int] = field(default_factory=lambda: [[[1], []], [[3], [6]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a moose",
                "the forest"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a moose",
                "rain in the forest"
            ]
        )
    prompt_merged_list = ["moose in the forest", "moose in the forest"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "rain": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)




@dataclass
class RunConfig13:

    Original = "a person using a laptop on a table"
    token_indices_1: List[int] = field(default_factory=lambda: [[[2], [5]], [[8], []]])
    Edit = "a person using a laptop on a table with books"
    token_indices_2: List[int] = field(default_factory=lambda: [[[2], [5]], [[8], [10]]])
    prompt_anchor_1: List[str] = field(
            default_factory=lambda: [
                "a person using a laptop",
                "a table"
            ]
        )
    prompt_anchor_2: List[str] = field(
            default_factory=lambda: [
                "a person using a laptop",
                "a table with books"
            ]
        )
    prompt_merged_list = ["a person on a table", "a person on a table"]
    cross_attention_kwargs = {"edit_type": "refine",
                                "n_self_replace": 0.4,
                                "n_cross_replace": {"default_": 1.0, "books": 0.8},
                                }
    # stable diffusion model path,set to None means sdxl model will be downloaded automatically
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False

    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [89, 122011213139902])
    # Path to save all outputs to
    output_path: Path = Path("./demo")
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: int = 32
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
            6: 21.5,
            7: 21,
            8: 21,
            9: 21,
        }
    )
    # steps to use tome refinement, only works when run_standard_sd is False, first one control token refinement, second one control attention refinement
    tome_control_steps: List[int] = field(default_factory=lambda: [7, 7])
    # token refinement steps per inference step, only works when run_standard_sd is False,
    token_refinement_steps: int = 3
    # attention map refinement steps per inference step, only works when run_standard_sd is False,
    attention_refinement_steps: List[int] = field(default_factory=lambda: [6, 6])
    # the timestep to replace prompt eot to merged prompt eot, only works when run_standard_sd is False, if value < 0, means do not replace eot tokens.
    # Earlier substitutions will result in missing subjects, later substitutions will cause confusion of subjects
    eot_replace_step: int = (
        60  # if larger than n_inference_steps, means do not replace eot tokens
    )
    # Pose loss will widen the distance between different subjects and reduce the situation of being confused into one.
    # Details see Appendix E. and https://arxiv.org/abs/2306.00986
    use_pose_loss: bool = True
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 3
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)