from onnx_trainer_forecast import Trainer as Model
import torch
import onnxruntime


def main():
    # create ONNX model
    onnx_fpath = "emp.onnx"
    model = Model().eval()

    # create dummy input to onnx export
    B = 32 #batch size
    N_agent = 10 #maximum number of agents per scenario
    N_lane = 80 #maximum number of lane segments per scenario
    h_steps = 50 #num historical timesteps for agent tracks
    f_steps = 60 #num future timesteps for agent tracks
    lane_sampling_pts = 20 #sampling points per lane segment
    data = {
                "x": torch.rand(B, N_agent, h_steps, 2), #agent tracks as local differences
                "x_attr": torch.zeros((B, N_agent, 3), dtype=torch.int), #categorical agent attributes
                "x_positions": torch.rand(B, N_agent, h_steps, 2), #agent tracks in scene coordinates
                "x_centers": torch.rand(B, N_agent, 2), #center of agent track
                "x_angles": torch.rand(B, N_agent, h_steps+f_steps), #agent headings
                "x_velocity": torch.rand(B, N_agent, h_steps+f_steps), #velocity of agents as absolute values
                "x_velocity_diff": torch.rand(B, N_agent, h_steps), #velocity changes of agents
                "lane_positions": torch.rand(B, N_lane, lane_sampling_pts, 2), #lane segments in scene coordinates
                "lane_centers": torch.rand(B, N_lane, 2), #center of lane segments
                "lane_angles": torch.rand(B, N_lane), #orientation of lane segments
                "lane_attr": torch.rand(B, N_lane, 3), #categorial lane attributes
                "is_intersections": torch.rand(B, N_lane), # categorical lane attribute
                "y": torch.rand(B, N_agent, f_steps, 2), #agent future tracks as x,y positions
                "x_padding_mask": torch.zeros((B, N_agent, h_steps+f_steps), dtype=torch.bool), #padding mask for agent tracks
                "lane_padding_mask": torch.zeros((B, N_lane, lane_sampling_pts), dtype=torch.bool), #padding mask for lane segment points
                "x_key_padding_mask": torch.zeros((B, N_agent), dtype=torch.bool), #batch padding mask for agent tracks
                "lane_key_padding_mask": torch.zeros((B, N_lane), dtype=torch.bool), #batch padding mask for lane segments
                "num_actors": torch.full((B,), fill_value=N_agent, dtype=torch.int64),
                "num_lanes": torch.full((B,), fill_value=N_lane, dtype=torch.int64),
                "scenario_id": [] * B,
                "track_id": [] * B,
                "origin": torch.rand(B, 2), #scene to global coordinates position
                "theta": torch.rand(B), #scene to global coordinates orientation
            } 

    # dummy data inference pytorch model
    print( "MODEL INF PYTORCH:", type(model(data)) )

    model.to_onnx(onnx_fpath, input_sample=data, export_params=True, opset_version=11)
    
    # create onnx runtime session
    sess_opt = onnxruntime.SessionOptions()
    ort_session = onnxruntime.InferenceSession(onnx_fpath, sess_opt)

    for k in data.keys():
        if torch.is_tensor(data[k]): data[k] = data[k].cpu().numpy()
    for session_input in ort_session.get_inputs():
        print(session_input.name)

    # map data inputs to onnx inputs
    ort_input = {
        'onnx::Concat_0': data["x"],
        'onnx::Gather_1': data["x_attr"],  
        'onnx::Concat_3': data["x_centers"],  
        'onnx::Gather_4': data["x_angles"],  
        'onnx::Unsqueeze_6': data["x_velocity_diff"],  
        'onnx::Sub_7': data["lane_positions"],  
        'onnx::Unsqueeze_8': data["lane_centers"],  
        'onnx::Concat_9': data["lane_angles"],  
        'onnx::Slice_13': data["x_padding_mask"],  
        'onnx::Unsqueeze_14': data["lane_padding_mask"],  
        'onnx::Concat_15': data["x_key_padding_mask"],  
        'onnx::Concat_16': data["lane_key_padding_mask"],  
    }
    # dummy data inference onnx model
    ort_outs = ort_session.run(None, ort_input)
    print( "MODEL INF ORT:", [oo.shape for oo in ort_outs])
    return


if __name__ == "__main__":
    main()