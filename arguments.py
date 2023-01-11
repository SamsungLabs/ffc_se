import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from omegaconf import MISSING, OmegaConf

import datasets
import models
import utils

args = utils.ClassRegistry()


@args.add_to_registry("exp")
@dataclass
class ExperimentArgs:
    config_dir: str = MISSING
    config: str = MISSING
    project: str = "ffc_se"
    name: str = MISSING
    seed: int = 1
    cudnn_benchmark_off: bool = False
    root: str = os.getenv("EXP_ROOT", ".")
    notes: str = "empty notes"
    tags: Optional[Tuple[str]] = None


@args.add_to_registry("training")
@dataclass
class TrainingArgs:
    trainer: str = MISSING
    device: str = MISSING


@args.add_to_registry("data")
@dataclass
class DataArgs:
    name: str = MISSING
    loader: str = MISSING
    root: str = MISSING
    root_dir: Optional[str] = None
    sampling_rate: int = MISSING
    input_freq: int = MISSING
    num_workers: int = MISSING
    dir4inference: str = MISSING


DatasetArgs = datasets.datasets.make_dataclass_from_args("DatasetArgs")
args.add_to_registry("dataset")(DatasetArgs)

LoaderArgs = datasets.loaders.make_dataclass_from_args("LoaderArgs")
args.add_to_registry("loader")(LoaderArgs)


@args.add_to_registry("gen")
@dataclass
class GenArgs:
    model: str = "ffc_se"


GenNetsArgs = models.generators.make_dataclass_from_args("GenNetsArgs")
args.add_to_registry("gennets")(GenNetsArgs)


@args.add_to_registry("checkpoint")
@dataclass
class CheckpointArgs:
    save_every: int = 100
    save_full_every: int = 200
    checkpoint_dir: str = "checkpoints/"
    checkpointing_off: bool = False
    checkpoint4inference: str = MISSING


Args = args.make_dataclass_from_classes("Args")


def load_config():
    config = OmegaConf.structured(Args)

    conf_cli = OmegaConf.from_cli()
    config.exp.config = conf_cli.exp.config
    config.exp.config_dir = conf_cli.exp.config_dir

    config_path = os.path.join(config.exp.config_dir, config.exp.config)
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, conf_file)

    config = OmegaConf.merge(config, conf_cli)

    return config
