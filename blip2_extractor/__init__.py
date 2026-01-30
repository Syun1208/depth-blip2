"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
from omegaconf import OmegaConf
from common.registry import registry

from blip2_extractor.blip2 import Blip2Base
# from blip2_extractor.blip2_qformer import Blip2Qformer1
from blip2_extractor.depth_blip2_opt import Blip2OPT
from blip2_extractor.depth_blip2_vicuna_instruct import Blip2VicunaInstruct
# from lavis.models.blip2_models.blip2_image_text_matching import Blip2ITM

from lavis.models.vit import VisionTransformerEncoder

from lavis.processors.base_processor import BaseProcessor
from processors.blip_processors import *
from processors.clip_processors import *


__all__ = [
    "load_model",
    "Blip2Base",
    "Blip2OPT",
    "Blip2VicunaInstruct",
    "VisionTransformerEncoder",
]


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.

    To list all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.

    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)


def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """
    print(registry.mapping)
    print(registry.list_processors())
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else BaseProcessor()
        )

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = _build_proc_from_cfg(vis_train_cfg)
    vis_processors["eval"] = _build_proc_from_cfg(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = _build_proc_from_cfg(txt_train_cfg)
    txt_processors["eval"] = _build_proc_from_cfg(txt_eval_cfg)

    return vis_processors, txt_processors


def load_model_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    """
    Load model and its related preprocessors.

    List all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.
    """
    from transformers import AutoModelForCausalLM

    model_cls = registry.get_model_class(name)

    # load model
    print("Load model")
    # model = AutoModelForCausalLM.from_pretrained(
    #     "facebook/opt-2.7b",
    #     device_map="auto",
    #     load_in_8bit=True,      # ✅ works without BitsAndBytesConfig
    #     low_cpu_mem_usage=True
    # )
    model = model_cls.from_pretrained(model_type=model_type)
    print("End loading model")


    if is_eval:
        model.eval()

    # load preprocess
    print("Start OmegaConf")
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    print("End OmegaConf")

    if cfg is not None:
        preprocess_cfg = cfg.preprocess
        print(f"preprocess_cfg: {preprocess_cfg}")
        print('Start loading visual and textual model')
        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
        print('End loading visual and textual model')
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    model = model.float() if device == torch.device("cpu") else model.to(device)

    return model, vis_processors, txt_processors


class ModelZoo:
    """
    A utility class to create string representation of available model architectures and types.

    >>> from lavis.models import model_zoo
    >>> # list all available models
    >>> print(model_zoo)
    >>> # show total number of models
    >>> print(len(model_zoo))
    """

    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_CONFIG_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        return (
                "=" * 50
                + "\n"
                + f"{'Architectures':<30} {'Types'}\n"
                + "=" * 50
                + "\n"
                + "\n".join(
            [
                f"{name:<30} {', '.join(types)}"
                for name, types in self.model_zoo.items()
            ]
        )
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()
