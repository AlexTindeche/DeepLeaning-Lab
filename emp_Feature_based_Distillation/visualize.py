import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from src.datamodule.av2_dataset import Av2Dataset, collate_fn

from src.model.trainer_forecast_hint_distilation import hintBasedTrainer as HintDistilModel
from src.model.trainer_forecast import Trainer as OriginalModel
from src.model.emp import EMP


from src.utils.vis import visualize_scenario


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-p", "--predict", help="", action="store_true")
    args = parser.parse_args()
    predict = args.predict

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    split = "emp/val"
    data_root = Path("/Users/alexanderpaulus/Desktop/UniFreiburg/SS_25/deep_learning_lab/mainProject/data/datasets")
    dataset = Av2Dataset(data_root=data_root, cached_split=split)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    isHintBasedDistil = False

    if predict:
        # chkpt_fpath = "checkpoints/empm.ckpt"
        chkpt_fpath = "outputs/emp-forecast_av2/2025-08-01/17-30-54/checkpoints/last.ckpt"
        assert os.path.exists(chkpt_fpath), "chkpt files does not exist, update path to checkpoint"
        
        if isHintBasedDistil:

            student = EMP(
                embed_dim=32,
                encoder_depth=2,
                num_heads=4,
                mlp_ratio=2.0,
                qkv_bias=False,
                drop_path=0.2,
                decoder="mlp"
            )
            
            checkpoint = torch.load(chkpt_fpath, map_location=device)
            state_dict = checkpoint['state_dict']
            
            # filters out teacher keywords
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('teacher.')}
            
            # replaces the checkpoints state_dict with filtered one
            checkpoint['state_dict'] = filtered_state_dict

            # ignores unexpected keys with strict=false
            model = HintDistilModel.load_from_checkpoint(
                chkpt_fpath,
                map_location=device,
                teacher_model=None,
                student_model=student,
                pretrained=False,
                pretrained_weights_teacher=None,
                pretrained_weights_student=None,
                strict=False  
            )
            
            # loads filtered state_dict manually for no mis mathcing 
            model.load_state_dict(filtered_state_dict, strict=False)
            
            model = model.eval().to(device)
        else:
            model = OriginalModel.load_from_checkpoint(chkpt_fpath, pretrained_weights=chkpt_fpath)
            model = model.eval().to(device)



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
                # changed to work on cpu
                # if torch.is_tensor(data[k]): data[k] = data[k].cuda()
                if torch.is_tensor(data[k]): data[k] = data[k].to(device)
            with torch.no_grad():
                batch_pred, scores = model.predict(data, full=True)

        if "y" not in data.keys(): data["y"] = torch.zeros((data["x"].shape[0], data["x"].shape[1], 60, 2), device=data["x"].device)

        for b in range(0, data["x"].shape[0], 1):
            scene_id = data["scenario_id"][b]
            scene_file = data_root / "val" / scene_id / ("scenario_" + scene_id + ".parquet")
            map_file = data_root / "val" / scene_id / ("log_map_archive_" + scene_id + ".json")
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scene_file)
            static_map = ArgoverseStaticMap.from_json(map_file)
            if predict:
                prediction = batch_pred[0][b].squeeze()
                # print(prediction[-1, :, :][np.newaxis, :, :])
                visualize_scenario(scenario, static_map, title="{}".format(scene_id), prediction=prediction, tight=True, timestep=49 if split == "test" else 50)
            else:
                visualize_scenario(scenario, static_map, title="{}".format(scene_id), tight=True, timestep=49 if split == "test" else 50)
            plt.show()

    return


if __name__ == "__main__":
    main()
