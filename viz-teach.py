import argparse
import collections
import os
from pathlib import Path
import typing

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import timm
import torch
import torch.nn as nn
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from src.model.emp import EMP
from src.model.layers.lane_embedding import LaneEmbeddingLayer
from src.model.layers.multimodal_decoder_emp import MultimodalDecoder
from src.model.layers.transformer_blocks import Block
from src.model.layers.transformer_blocks import Mlp
from src.datamodule.av2_dataset import Av2Dataset, collate_fn
from src.model.trainer_forecast import Trainer as Model_Teacher
from src.model.teacher_student import TeacherStudentTrainer as Model
from src.utils.vis import visualize_scenario

# Get all torch.nn classes and functions
torch_nn_globals = []

# Add all classes from torch.nn
for name in dir(torch.nn):
    attr = getattr(torch.nn, name)
    if hasattr(attr, '__module__') and hasattr(attr, '__qualname__'):
        torch_nn_globals.append(attr)

# Add all classes from torch.nn.modules (more comprehensive)
import torch.nn.modules
for module_name in dir(torch.nn.modules):
    try:
        module = getattr(torch.nn.modules, module_name)
        if hasattr(module, '__module__'):
            # Add all classes in this submodule
            for class_name in dir(module):
                cls = getattr(module, class_name)
                if hasattr(cls, '__module__') and hasattr(cls, '__qualname__'):
                    torch_nn_globals.append(cls)
    except:
        continue

torch.serialization.add_safe_globals(torch_nn_globals)

torch.serialization.add_safe_globals([omegaconf.DictConfig, omegaconf.ListConfig, omegaconf.OmegaConf, omegaconf.base.ContainerMetadata,
                                    omegaconf.nodes.AnyNode,
                                    omegaconf.base.Metadata, 
                                    typing.Any, 
                                    dict, 
                                    collections.defaultdict, 
                                    list, 
                                    tuple, 
                                    str, 
                                    int, 
                                    float, 
                                    bool, 
                                    EMP,
                                    torch.nn.modules.linear.Linear,
                                    torch.nn.modules.container.ModuleList,
                                    Block,
                                    torch.nn.modules.normalization.LayerNorm,
                                    torch.nn.modules.activation.MultiheadAttention,
                                    torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
                                    torch.nn.modules.linear.Identity,
                                    Mlp,
                                    torch.nn.modules.activation.GELU,
                                    timm.layers.drop.DropPath,
                                    LaneEmbeddingLayer,
                                    MultimodalDecoder,
                                    ])


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-p", "--predict", help="", action="store_true")
    args = parser.parse_args()
    predict = args.predict
    predict = True

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    split = "emp/val"
    data_root = Path("D:/Uni-work/UniFreiburg/DL_Lab/Project/DeepLeaning-Lab/data_root")
    dataset = Av2Dataset(data_root=data_root, cached_split=split)

    if predict:
        chkpt_fpath = "checkpoints/empd-base.ckpt"
        # Upload the teacher separately as we encountered last minute bugs
        assert os.path.exists(chkpt_fpath), "chkpt files does not exist, update path to checkpoint"
        model_Teacher = Model_Teacher.load_from_checkpoint(chkpt_fpath, pretrained_weights=chkpt_fpath)
        model_Teacher = model_Teacher.eval().cuda()
        chkpt_fpath = "outputs/emp-forecast_av2/2025-08-05/20-52-06/checkpoints/last.ckpt"
        assert os.path.exists(chkpt_fpath), "chkpt files does not exist, update path to checkpoint"
        model = Model.load_from_checkpoint(chkpt_fpath)
        model = model.eval().cuda()

    B = 64
    dataloader = TorchDataLoader(
        dataset,
        batch_size=B,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    ###################################################################################################################################################################################################

    for data in tqdm(dataloader):
        if predict:
            for k in data.keys():
                if torch.is_tensor(data[k]): data[k] = data[k].cuda()
            with torch.no_grad():
                batch_pred, scores = model.predict(data, full=True)
                batch_pred_teacher, scores = model_Teacher.predict(data, full=True)

        if "y" not in data.keys(): data["y"] = torch.zeros((data["x"].shape[0], data["x"].shape[1], 60, 2), device=data["x"].device)

        for b in range(0, data["x"].shape[0], 1):
            scene_id = data["scenario_id"][b]
            scene_file = data_root / "val" / scene_id / ("scenario_" + scene_id + ".parquet")
            map_file = data_root / "val" / scene_id / ("log_map_archive_" + scene_id + ".json")
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scene_file)
            static_map = ArgoverseStaticMap.from_json(map_file)
            if predict:
                prediction = batch_pred[0][b].squeeze()
                prediction_teacher = batch_pred_teacher[0][b].squeeze()
                visualize_scenario(scenario, static_map, title="{}".format(scene_id), prediction=prediction, tight=True, timestep=49 if split == "test" else 50)
                visualize_scenario(scenario, static_map, title="{}".format(scene_id), prediction=prediction_teacher, tight=True, timestep=49 if split == "test" else 50, teacher=True)

            else:
                visualize_scenario(scenario, static_map, title="{}".format(scene_id), tight=True, timestep=49 if split == "test" else 50)
            plt.show()

    return


if __name__ == "__main__":
    main()
