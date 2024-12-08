import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.image_processor import PipelineImageInput

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms as T

from processors import *

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
# TODO: pulled in from ToME
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# TODO: Pulled in from ToME
def get_centroid(attn_map: torch.Tensor) -> torch.Tensor:
    """
    attn_map: h*w*token_len
    """
    h, w, seq_len = attn_map.shape

    attn_x, attn_y = attn_map.sum(0), attn_map.sum(1)  # w|h seq_len
    x = torch.linspace(0, 1, w).to(attn_map.device).reshape(w, 1)
    y = torch.linspace(0, 1, h).to(attn_map.device).reshape(h, 1)

    centroid_x = (x * attn_x).sum(0) / attn_x.sum(0)  # seq_len
    centroid_y = (y * attn_y).sum(0) / attn_y.sum(0)  # bs seq_len
    centroid = torch.stack((centroid_x, centroid_y), -1)  # (seq_len, 2)
    return centroid


# TODO: pulled in from ToME
def register_self_time(pipe, i):
    for name, module in pipe.unet.named_modules():
        # if name in attn_greenlist:
        if (name.startswith("mid_block")) and name.endswith("attn1"):
        # if name.endswith("attn1"):
        # if name.startswith("down_blocks.2") and name.endswith("attn1"):
            setattr(module, 'time', i)


def token_merge(
    prompt_embeds: torch.Tensor, idx_merge: List[List[int]]
) -> torch.Tensor:
    """
    prompt_embeds: 77 dim
    idx_merge: [ [[1],[2]],[[3],[4]] ]
    """

    for idxs in idx_merge:
        noun_idx = idxs[0][0]
        alpha = 1.1
        prompt_embeds[noun_idx] = alpha * prompt_embeds[idxs[0]].sum(
            dim=0
        ) + 1.2 * prompt_embeds[idxs[1]].sum(dim=0)
        if len(idxs[0]) > 1:
            prompt_embeds[idxs[0][1:]] = 0
        prompt_embeds[idxs[1]] = 0

    return prompt_embeds


class Prompt2PromptPipeline(StableDiffusionXLPipeline):
    r"""
    Args:
    Prompt-to-Prompt-Pipeline for text-to-image generation using Stable Diffusion. This model inherits from
    [`StableDiffusionPipeline`]. Check the superclass documentation for the generic methods the library implements for
    all the pipelines (such as downloading or saving, running on a particular device, etc.)
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents. scheduler
        ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    # the sequence in which model compoents should be offloaded from GPU to CPU memory.
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"

    # declaring which components without which the model should be able to function
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "safety_checker", 
        "feature_extractor"
    ]

    # the tensors that can be updated through the callback mechanism
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def check_inputs(
            self,
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # if (callback_steps is None) or (
        #         callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        # ):
        #     raise ValueError(
        #         f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
        #         f" {type(callback_steps)}."
        #     )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # TODO: Not being used anywhere
    def _aggregate_and_get_attention_maps_per_token(self, with_softmax):
        attention_maps = self.controller.aggregate_attention(
            from_where=("up_cross", "down_cross", "mid_cross"),
            # from_where=("up", "down"),
            # from_where=("down",)
        )
        attention_maps_list = self._get_attention_maps_list(
            attention_maps=attention_maps, with_softmax=with_softmax
        )
        return attention_maps_list

    @staticmethod
    def _get_attention_maps_list(
            attention_maps: torch.Tensor, with_softmax
    ) -> List[torch.Tensor]:
        attention_maps *= 100

        if with_softmax:
            attention_maps = torch.nn.functional.softmax(attention_maps, dim=-1)

        attention_maps_list = [
            attention_maps[:, :, i] for i in range(attention_maps.shape[2])
        ]
        return attention_maps_list

    @staticmethod
    def _update_stoken(
        stoken: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the merged token according to the computed loss."""
        loss = loss * step_size
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [stoken])[0]
        stoken = stoken - grad_cond
        return stoken
    
    def opt_token(self, latents: torch.Tensor, t, stoken, prompt_anchor, iter_num=3):
        """
        latents: 128 128 4
        stoken: dim
        prompt_anchor: 77 dim
        """
        stoken.requires_grad_(True)

        print("##OPT TIMESTEP: ", t)
        print("##STOKEN SHAPE: ", stoken.shape)
        print("###LATENT_ANCHOR SHAPE PASSED TO OPT: ", latents.shape)
        print("###PROMPT ANCHOR SHAPE IN OPT: ", prompt_anchor.shape)

        latents = latents.clone().detach().unsqueeze(0)
        print("###LATENT_ANCHOR SHAPE PASSED TO OPT UNSQUEEZED: ", latents.shape)
        iteration = 0

        print("###CROSS_ATTENTION_KWARGS: ", self.cross_attention_kwargs)
        print("###ADDED_COND_KWARGS: ", self.added_cond_kwargs2)

        with torch.no_grad():
            noise_pred_anchor = self.unet(
                latents,
                t,
                encoder_hidden_states=prompt_anchor,
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample
        while True:
            print("###ITERATION: ", iteration)
            iteration += 1
            noise_pred_token = self.unet(
                latents,
                t,
                encoder_hidden_states=stoken.unsqueeze(0).unsqueeze(0),
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred_anchor, noise_pred_token)

            stoken = self._update_stoken(stoken, loss, 10000)
            if iteration >= iter_num:
                print(
                    f"Semantic binding loss optimization Exceeded max number of iterations ({iter_num}) "
                )
                break

        with torch.no_grad():
            noise_pred_null = self.unet(
                latents,
                t,
                encoder_hidden_states=self.negative_prompt_embeds[1:],
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample

            noise_pred = noise_pred_null + self.guidance_scale * (
                noise_pred_null - noise_pred_anchor
            )

            noise_pred = rescale_noise_cfg(
                noise_pred,
                noise_pred_anchor,
                guidance_rescale=self.guidance_rescale,
            )
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            self.scheduler._step_index -= 1
        return stoken, latents[0]
    

    def _entropy_loss(
        self,
        attention_store: AttentionStore,
        indices_to_alter: List[int],
        attention_res: int = 16,
        pose_loss: bool = False,
        prompt_idx: int = 0
    ):
        """Aggregates the attention for each token and computes the max activation value for each token to alter."""
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0,
        )  # h w 77

        loss = 0

        prompt = self.prompts[prompt_idx] if isinstance(self.prompts, list) else self.prompts
        print("###ENTROPY LOSS FOR PROMPT: ", prompt)
        last_idx = len(self.tokenizer(prompt)["input_ids"]) - 1

        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text = torch.nn.functional.softmax(
            attention_for_text / 0.5, dim=-1
        )

        # get pos idx and calculate pos loss
        indices = []
        for i in range(len(indices_to_alter)):
            curr_idx = indices_to_alter[i][0][0]
            indices.append(curr_idx)

        indices = [i - 1 for i in indices]
        cross_map = attention_for_text[:, :, indices]  # 32,32 seq_len
        cross_map = (cross_map - cross_map.amin(dim=(0, 1), keepdim=True)) / (
            cross_map.amax(dim=(0, 1), keepdim=True)
            - cross_map.amin(dim=(0, 1), keepdim=True)
        )
        cross_map = cross_map / cross_map.sum(dim=(0, 1), keepdim=True)

        loss = loss - 2 * (cross_map * torch.log(cross_map + 1e-5)).sum()
        if pose_loss: # this is False in config 2
            idx = 0
            for subject_idx, subject_idx2 in [indices]:
                # Shift indices since we removed the first token
                curr_map = attention_for_text[
                    :, :, [subject_idx, subject_idx2]
                ]  # h w k

                vis_map = curr_map.permute(2, 0, 1)  # k h w
                sub_map, sub_map2 = vis_map[0], vis_map[1]

                sub_map = (sub_map - sub_map.min()) / (sub_map.max() - sub_map.min())
                sub_map2 = (sub_map2 - sub_map2.min()) / (
                    sub_map2.max() - sub_map2.min()
                )

                curr_map = torch.stack([sub_map, sub_map2])  # k h w
                curr_map = curr_map.permute(1, 2, 0)  # h w k
                pair_pos = get_centroid(curr_map) * 32  # (2, 2) k 2

                pos1 = torch.tensor([10.0, 16]).to("cuda")

                pos2 = torch.tensor([25.0, 16]).to("cuda")

                loss = loss + (0.2 * (pair_pos[0] - pos1) ** 2).mean()
                loss = loss + (0.2 * (pair_pos[1] - pos2) ** 2).mean()

                T.ToPILImage()(sub_map.reshape(1, 32, 32)).save("mask_left.png")
                T.ToPILImage()(sub_map2.reshape(1, 32, 32)).save("mask_right.png")
        return loss
    
    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - 0.5 * step_size * grad_cond
        return latents

    @staticmethod
    def _update_text(
        text_embeddings: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [text_embeddings], retain_graph=True
        )[0]
        text_embeddings = text_embeddings - step_size * grad_cond
        return text_embeddings

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        indices_to_alter: List[Tuple[int, int]],
        threshold: float,
        text_embeddings: torch.Tensor,
        attention_store: AttentionStore,
        step_size: float,
        t: int,
        attention_res: int = 32,
        max_refinement_steps: List[int] = [3, 3],
        pose_loss: bool = False,
        prompt_idx: int = 1
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code and text embedding according to our loss objective until the given threshold is reached for all tokens.
        """
        threshold = threshold / 2 * len(indices_to_alter)
        threshold -= 2
        ratio = t / 1000
        if ratio > 0.9:
            max_refinement_steps = max_refinement_steps[0]
        if ratio <= 0.9:
            max_refinement_steps = max_refinement_steps[1]
        iteration = 0
        while True:
            iteration += 1
            torch.cuda.empty_cache()
            latents = latents.clone().detach().requires_grad_(True)
            text_embeddings = text_embeddings.clone().detach().requires_grad_(True)

            noise_pred_text = self.unet(
                latents,
                t,
                encoder_hidden_states=text_embeddings[1].unsqueeze(0),
                timestep_cond=self.timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=self.added_cond_kwargs2,
            ).sample

            print("###GETTING ENTROPY LOSS")
            loss = self._entropy_loss(
                attention_store, indices_to_alter, attention_res, pose_loss=pose_loss, prompt_idx=prompt_idx
            )
            if loss != 0:  # and t/1000 > 0.8:
                latents = self._update_latent(latents, loss, step_size)
                text_embeddings = self._update_text(text_embeddings, loss, step_size)

            if loss < threshold:
                break
            if iteration >= max_refinement_steps:
                print(
                    f"Entropy loss optimization Exceeded max number of iterations ({max_refinement_steps}) "
                )
                break

        return latents, loss, text_embeddings.detach()



    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5, ##TODO: change this to 5.0?
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        attn_res=None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

                The keyword arguments to configure the edit are:
                - edit_type (`str`). The edit type to apply. Can be either of `replace`, `refine`, `reweight`.
                - n_cross_replace (`int`): Number of diffusion steps in which cross attention should be replaced
                - n_self_replace (`int`): Number of diffusion steps in which self attention should be replaced
                - local_blend_words(`List[str]`, *optional*, default to `None`): Determines which area should be
                  changed. If None, then the whole image can be changed.
                - equalizer_words(`List[str]`, *optional*, default to `None`): Required for edit type `reweight`.
                  Determines which words should be enhanced.
                - equalizer_strengths (`List[float]`, *optional*, default to `None`) Required for edit type `reweight`.
                  Determines which how much the words in `equalizer_words` should be enhanced.

            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        callback = None
        callback_steps = None

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # attention_store = kwargs.get("attention_store")
        indices_to_alter_1 = kwargs.get("indices_to_alter_1")
        indices_to_alter_2 = kwargs.get("indices_to_alter_2")
        attention_res = kwargs.get("attention_res")
        run_standard_sd = kwargs.get("run_standard_sd")
        thresholds = kwargs.get("thresholds")
        scale_factor = kwargs.get("scale_factor")
        scale_range = kwargs.get("scale_range")
        smooth_attentions = kwargs.get("smooth_attentions")
        sigma = kwargs.get("sigma")
        kernel_size = kwargs.get("kernel_size")
        prompt_anchor_1 = kwargs.get("prompt_anchor_1")
        prompt_anchor_2 = kwargs.get("prompt_anchor_2")
        prompt_merged_1 = kwargs.get("prompt_merged_1")
        prompt_merged_2 = kwargs.get("prompt_merged_2")
        prompt_length_1 = kwargs.get("prompt_length_1")
        prompt_length_2 = kwargs.get("prompt_length_2")
        token_refinement_steps = kwargs.get("token_refinement_steps")
        attention_refinement_steps = kwargs.get("attention_refinement_steps")
        tome_control_steps = kwargs.get("tome_control_steps")
        eot_replace_step = kwargs.get("eot_replace_step")
        use_pose_loss = kwargs.get("use_pose_loss")

        # 0. Default height and width to unet
        print("###HEIGHT RECEIVED: ", height)
        print("###WIDTH RECEIVED: ", width)
        print("###DEFAULT SAMPLE SIZE: ", self.unet.config.sample_size)
        print("###VAE SCALE FACTOR: ", self.vae_scale_factor)

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        print("###HEIGHT SET: ", height)
        print("###WIDTH SET: ", width)

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attn_res = attn_res

        self.controller = create_controller(
            prompt, cross_attention_kwargs, num_inference_steps, tokenizer=self.tokenizer, device=self.device, attn_res=self.attn_res, enable_edit=True
        )
        self.register_attention_control(self.controller)  # add attention controller
        attention_store =self.controller
        
        # self.controller = AttentionStoreTome()
        # register_attention_control_tome(self, self.controller)
        # attention_store = self.controller

        self.prompts = prompt # for use in _entropy_loss

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        print("###BATCH SIZE: ", batch_size)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 #TODO: not using this in ToME
        # do_classifier_free_guidance = False

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip, ##TODO: added this
        )

        print("###PROMPTS: ",  prompt)
        print("###INDICES TO ALTER 1: ", indices_to_alter_1)
        print("###INDICES TO ALTER 2: ", indices_to_alter_2)
        print("###PROMPT_2 (expecting None): ", prompt_2)
        print("###PROMPT EMBEDS SHAPE: ", prompt_embeds.shape)
        print("###PROMPT EMBEDS: ", prompt_embeds.shape)
        print("###NEGATIVE PROMPT EMBEDS SHAPE: ", negative_prompt_embeds.shape)
        print("###NEGATIVE PROMPT EMBEDS: ", negative_prompt_embeds)

        # TODO: Getting prompt anchor embeddings
        panchors_1 = []
        for panchor in prompt_anchor_1:
            (
                prompt_anchor_emb,
                _,
                _,
                _,
            ) = self.encode_prompt(
                prompt=panchor,
                prompt_2=panchor,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
            panchors_1.append(prompt_anchor_emb)

        print("###PROMPT ANCHOR: ", prompt_anchor_1)
        print("###PROMPT ANCHOR EMBEDS SHAPE: ", prompt_anchor_emb.shape)
        
        panchors_2 = []
        for panchor in prompt_anchor_2:
            (
                prompt_anchor_emb,
                _,
                _,
                _,
            ) = self.encode_prompt(
                prompt=panchor,
                prompt_2=panchor,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
            panchors_2.append(prompt_anchor_emb)
        
        print("###PROMPT ANCHOR 2: ", prompt_anchor_2)
        print("###PROMPT ANCHOR EMBEDS SHAPE: ", prompt_anchor_emb.shape)

        # TODO: Getting merged prompt embeddings
        (
            prompt_merged_emb_1,
            _,
            _,
            _,
        ) = self.encode_prompt(
            prompt=prompt_merged_1,
            prompt_2=prompt_merged_1,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        print("###MERGED PROMPT 1: ", prompt_merged_1)
        print("###MERGED PROMPT EMBEDS SHAPE 1: ", prompt_merged_emb_1.shape)

        (
            prompt_merged_emb_2,
            _,
            _,
            _,
        ) = self.encode_prompt(
            prompt=prompt_merged_2,
            prompt_2=prompt_merged_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        print("###MERGED PROMPT 2: ", prompt_merged_2)
        print("###MERGED PROMPT 2 EMBEDS SHAPE: ", prompt_merged_emb_2.shape)

        if not run_standard_sd and token_refinement_steps:
            print("###TOKEN MERGING FOR PROMPT 1")
            # print("###ORIGINAL EMBEDDINGS 1: ", prompt_embeds[0])
            prompt_embeds[0] = token_merge(prompt_embeds[0], indices_to_alter_1)
            print("###TOKEN MERGING FOR PROMPT 2")
            # print("###ORIGINAL EMBEDDINGS 2: ", prompt_embeds[1])
            prompt_embeds[1] = token_merge(prompt_embeds[1], indices_to_alter_2)
            


        # 4. Prepare timesteps
        # self.scheduler.set_timesteps(num_inference_steps, device=device)
        # timesteps = self.scheduler.timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        ) #TODO: using ToME's timestep setting

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        print("###LATENTS SHAPE: ", latents.shape)
        latents[1] = latents[0] # setting initial latents to be similar


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        print("###ADD_TEXT_EMBEDS SHAPE 1: ", add_text_embeds.shape)
        print("###ADD_TEXT_EMBEDS 1: ", add_text_embeds)
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim # if none should be changed to enc1
        )
        print("###ADD_TIME_IDS SHAPE 1: ", add_time_ids.shape)
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.text_encoder_2.config.projection_dim, # TODO: added from ToME
            )
        else:
            negative_add_time_ids = add_time_ids

        # TODO: Added this
        n1 = negative_prompt_embeds[0:1]  # shape [1, 77, 2048]
        n2 = negative_prompt_embeds[1:2]  # shape [1, 77, 2048]
        p1 = prompt_embeds[0:1]          # shape [1, 77, 2048]
        p2 = prompt_embeds[1:2]          # shape [1, 77, 2048]

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        print("###ADD_TIME_IDS SHAPE 2: ", add_time_ids.shape)

        print("###ADD_TEXT_EMBEDS SHAPE 2: ", add_text_embeds.shape)
        print("###ADD_TEXT_EMBEDS 2: ", add_text_embeds)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids_og = add_time_ids
        add_time_ids_og = add_time_ids_og.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        print("###NUM IMAGES PER PROMPT: ", num_images_per_prompt)
        print("###ADD_TIME_IDS SHAPE 3: ", add_time_ids.shape)
        print("###ADD_TIME_IDS: ", add_time_ids)


        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # TODO: Added from ToME (Optionally get Guidance Scale Embedding)
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.timestep_cond = timestep_cond
        self._num_timesteps = len(timesteps)
        self.timesteps = timesteps

        scale_range = np.linspace(
            scale_range[0], scale_range[1], len(self.scheduler.timesteps)
        )
        ##

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        text_embeds_og = torch.zeros(1, 1280)
        text_embeds_og = text_embeds_og.to(device)
        # TODO: Added from ToME
        added_cond_kwargs2 = {
            # "text_embeds": torch.zeros_like(add_text_embeds[1:]),
            "text_embeds": text_embeds_og,
            "time_ids": add_time_ids_og[1:],
        }

        self.added_cond_kwargs2 = added_cond_kwargs2
        self.negative_prompt_embeds = negative_prompt_embeds
        self.pos = None
        ##

        print("###CURRENT PROMPT EMBEDS SHAPE: ", prompt_embeds.shape)
        print("###CURRENT PROMPT EMBEDS: ", prompt_embeds)

        latent_anchor_1 = None
        latent_anchor_2 = None
        updated_prompt_embeds_1 = None
        updated_prompt_embeds_2 = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                register_self_time(self, None)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                print("###LATENT_MODEL_INPUTS SHAPE: ", latent_model_input.shape)

                # TODO: Setting latent anchors like in ToME, but for both prompts
                latent_anchor_1 = (
                    torch.cat([latents[0:1]] * len(panchors_1))
                    if latent_anchor_1 is None
                    else latent_anchor_1
                )
                latent_anchor_1 = self.scheduler.scale_model_input(latent_anchor_1, t)
                print("###LATENT_ANCHOR 1 SHAPE: ", latent_anchor_1.shape)

                latent_anchor_2 = (
                    torch.cat([latents[1:]] * len(panchors_2))
                    if latent_anchor_2 is None
                    else latent_anchor_2
                )
                latent_anchor_2 = self.scheduler.scale_model_input(latent_anchor_2, t)
                print("###LATENT_ANCHOR 2 SHAPE: ", latent_anchor_2.shape)
                ##

                # ToME: initializing varaibles that will be updated during the ToME refinement process
                updated_latents_1 = (
                    latent_model_input[1:2].clone().detach()
                )

                print("###LATENTS_UP 1 SHAPE: ", updated_latents_1.shape)

                updated_latents_2 = (
                    latent_model_input[3:].clone().detach()
                )

                print("###LATENTS_UP 2 SHAPE: ", updated_latents_2.shape)

                updated_prompt_embeds_1 = (
                    torch.cat([n1, p1], dim=0) if updated_prompt_embeds_1 is None else updated_prompt_embeds_1
                )
                print("###UPDATED_PROMPT_EMBEDS 1 SHAPE: ", updated_prompt_embeds_1.shape)

                updated_prompt_embeds_2 = (
                    torch.cat([n2, p2], dim=0) if updated_prompt_embeds_2 is None else updated_prompt_embeds_2
                )
                print("###UPDATED_PROMPT_EMBEDS 2 SHAPE: ", updated_prompt_embeds_2.shape)
                ##
                
                with torch.enable_grad():
                    if not run_standard_sd:
                        self.controller.enable_token_refine()
                        token_control, attention_control = tome_control_steps

                        #EOT Replace
                        if i == eot_replace_step:
                            print("###IN EOT REPLACE STEP")
                            updated_prompt_embeds_1[1, prompt_length_1 + 1 :] = prompt_merged_emb_1[0][prompt_length_1 + 1 :]
                            updated_prompt_embeds_2[1, prompt_length_2 + 1 :] = prompt_merged_emb_2[0][prompt_length_2 + 1 :]

                        # Applying semantic binding loss for token refinement
                        if i < token_control:
                            # For original prompt
                            for idx, panchor in enumerate(panchors_1):
                                stoken = (
                                    updated_prompt_embeds_1[1, indices_to_alter_1[idx][0][0]].detach().clone()
                                )

                                stoken, latent_anchor_1[idx] = self.opt_token(
                                    latent_anchor_1[idx],
                                    t,
                                    stoken,
                                    panchor,
                                    token_refinement_steps,
                                )
                                updated_prompt_embeds_1[1, indices_to_alter_1[idx][0][0]] = stoken

                            # For edit prompt
                            for idx, panchor in enumerate(panchors_2):
                                stoken = (
                                    updated_prompt_embeds_2[1, indices_to_alter_2[idx][0][0]].detach().clone()
                                )

                                stoken, latent_anchor_2[idx] = self.opt_token(
                                    latent_anchor_2[idx],
                                    t,
                                    stoken,
                                    panchor,
                                    token_refinement_steps,
                                )
                                updated_prompt_embeds_2[1, indices_to_alter_2[idx][0][0]] = stoken

                        # Applying entropy loss for attention refinement
                        if i < attention_control:
                            # For original prompt
                            updated_latents_1, loss, updated_prompt_embeds_1 = (
                                self._perform_iterative_refinement_step(
                                    latents=updated_latents_1,
                                    indices_to_alter=indices_to_alter_1,
                                    threshold=thresholds[i],
                                    text_embeddings=updated_prompt_embeds_1,
                                    attention_store=attention_store,
                                    step_size=scale_factor * scale_range[i],
                                    t=t,
                                    attention_res=attention_res,
                                    max_refinement_steps=attention_refinement_steps,
                                    pose_loss=use_pose_loss,
                                    prompt_idx=0,
                                )
                            )

                            # For edit prompt
                            updated_latents_2, loss, updated_prompt_embeds_2 = (
                                self._perform_iterative_refinement_step(
                                    latents=updated_latents_2,
                                    indices_to_alter=indices_to_alter_2,
                                    threshold=thresholds[i],
                                    text_embeddings=updated_prompt_embeds_2,
                                    attention_store=attention_store,
                                    step_size=scale_factor * scale_range[i],
                                    t=t,
                                    attention_res=attention_res,
                                    max_refinement_steps=attention_refinement_steps,
                                    pose_loss=use_pose_loss,
                                    prompt_idx=1,
                                )
                            )

                            print(f"Iteration {i} | Loss: {loss:0.4f}")

                
                print("###UPDATED_LATENTS 1 SHAPE AFTER REFINE (BEFORE CAT): ", updated_latents_1.shape)
                updated_latents_1 = (
                    torch.cat([updated_latents_1] * 2)
                    if do_classifier_free_guidance
                    else updated_latents_1
                )
                print("###UPDATED_LATENTS 1 SHAPE AFTER REFINE (AFTER CAT): ", updated_latents_1.shape)

                print("###UPDATED_LATENTS 2 SHAPE AFTER REFINE (BEFORE CAT): ", updated_latents_2.shape)
                updated_latents_2 = (
                    torch.cat([updated_latents_2] * 2)
                    if do_classifier_free_guidance
                    else updated_latents_2
                )
                print("###UPDATED_LATENTS 2 SHAPE AFTER REFINE (AFTER CAT): ", updated_latents_2.shape)


                latent_model_input = torch.cat([updated_latents_1, updated_latents_2], dim=0)
                # print("###LATENT_MODEL_INPUTS SHAPE AFTER REFINE AND CAT: ", latent_model_input.shape)

                n1 = updated_prompt_embeds_1[0:1]   # First row of first_combination
                n2 = updated_prompt_embeds_2[0:1]  # First row of second_combination
                p1 = updated_prompt_embeds_1[1:2]   # Second row of first_combination
                p2 = updated_prompt_embeds_2[1:2]  # Second row of second_combination

                final_prompt_embeds = torch.cat([n1, n2, p1, p2], dim=0)  # shape [4, 77, 2048]
                print("###FINAL_PROMPT_EMBEDS SHAPE: ", final_prompt_embeds.shape)

                self.controller.disable_token_refine()


                # predict the noise residual
                # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,
                #                        added_cond_kwargs=added_cond_kwargs, ).sample
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=final_prompt_embeds,
                                       added_cond_kwargs=added_cond_kwargs, ).sample
                

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # step callback
                latents = self.controller.step_callback(latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 8. Post-processing
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                list(reversed(self.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            # attn_procs[name] = P2PCrossAttnProcessor(controller=controller, place_in_unet=place_in_unet)
            attn_procs[name] = AttendExciteCrossAttnProcessor(
                attnstore=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count

    
