import argparse
import os
import glob
import time
from pprint import pprint
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from matplotlib import font_manager

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed



class SingleVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_feature_path, cfg):
        """
        Initialize dataset for a single video
        
        Args:
            video_feature_path (str): Path to the .npy feature file of the video
            cfg (dict): Configuration dictionary
        """
        # Configuration parameters
        input_dim = cfg['dataset']['input_dim']  # 2048 for I3D features
        
        # Load video features
        features = np.load(video_feature_path)
        print(f"Original features shape: {features.shape}")
        
        # Ensure the format is (frames, features)
        assert features.ndim == 2, "Features must be a 2D array (frames x features)"
        assert features.shape[1] == input_dim, \
            f"Feature dimension mismatch. Expected {input_dim}, got {features.shape[1]}"
        
        # Convert to tensor and transpose to [features, frames]
        # This matches your desired format where shape is [2048, frames]
        self.features = torch.from_numpy(features).float().transpose(0, 1)
        print(f"Transposed features shape: {self.features.shape}")
        
        # Create corresponding mask tensor (all True)
        self.mask = torch.ones(self.features.shape[1], dtype=torch.bool)
        
        # Prepare video metadata
        self.video_id = os.path.splitext(os.path.basename(video_feature_path))[0]

    def __len__(self):
        return 1  # Single video dataset

    def __getitem__(self, idx):
        """
        Returns the video features directly in the required format
        """
        # Return features directly in the [2048, frames] format without extra dimensions
        return {
            'feats': self.features,         # Shape: [2048, frames]
            'masks': self.mask,             # Shape: [frames]
            'video_id': self.video_id,
        }

def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"

    # Checkpoint file handling
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    # Validate input video feature file
    assert os.path.isfile(args.video), f"Video feature file {args.video} does not exist!"
    assert os.path.isfile(args.video_path), f"Video file {args.video_path} does not exist!"

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    # Create custom single video dataset
    val_dataset = SingleVideoDataset(args.video, cfg)
    
    # Get a single video data sample - no need for DataLoader here
    video_data = val_dataset[0]
    print(f"Shape of video features: {video_data['feats'].shape}")
    print(f"Mask shape: {video_data['masks'].shape}")

    """3. create model"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(int(cfg['devices'][0].split(':')[1]))
    )

    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    """5. Predict on single video"""
    print(f"\nStart predicting actions in video: {args.video}")
    model.eval()
    
    # Extract test configuration parameters
    test_cfg = cfg['model'].get('test_cfg', {})
    voting_thresh = test_cfg.get('voting_thresh', 0.7)
    pre_nms_topk = test_cfg.get('pre_nms_topk', 2000)
    max_seg_num = test_cfg.get('max_seg_num', 200)
    min_score = test_cfg.get('min_score', 0.001)
   
   
    label_mapping = {
        0: "BaseballPitch",
        1: "BasketballDunk",
        2: "Billiards",
        3: "CleanAndJerk",
        4: "CliffDiving",
        5: "CricketBowling",
        6: "CricketShot",
        7: "Diving",
        8: "FrisbeeCatch",
        9: "GolfSwing",
        10: "HammerThrow",
        11: "HighJump",
        12: "JavelinThrow",
        13: "LongJump",
        14: "PoleVault",
        15: "Shotput",
        16: "SoccerPenalty",
        17: "TennisSwing",
        18: "ThrowDiscus",
        19: "VolleyballSpiking",
    }
    with torch.no_grad():
        # Create a list with just one element to match the model's expectation
        # This is crucial - pass exactly what the model expects
        video_list = [video_data]
        
        # Debug prints
        print("Input Feature Shape:", video_list[0]['feats'].shape)
        print("Input Mask Shape:", video_list[0]['masks'].shape)

        feats = video_list[0]['feats']



        video_data = {
                'video_id': 'testVideo',
                'feats': feats,
                'fps': 30.0,
                'duration': 120.0,
                'feat_stride': 4,
                'feat_num_frames': 16
            }

        video_list = [video_data]

        print(video_list)
        
        
        # Pass to model (they expect a list of dictionaries)
        with torch.no_grad():
            output = model(video_list)
        
        
        # Process and print predictions
        num_vids = len(output)
        for vid_idx in range(num_vids):
            if output[vid_idx]['segments'].shape[0] > 0:
                print("\nDetected Actions:")
                print(f"  Filtering with:")
                print(f"  - Voting Threshold: {voting_thresh}")
                print(f"  - Pre-NMS Top-K: {pre_nms_topk}")
                print(f"  - Max Segments: {max_seg_num}")
                print(f"  - Minimum Score: {min_score}")
                
                # For Action total length Print
                # label_times = {}

                for i in range(output[vid_idx]['segments'].shape[0]):
                    if(output[vid_idx]['scores'][i] > 0.2):
                        start = output[vid_idx]['segments'][i, 0]
                        end = output[vid_idx]['segments'][i, 1]
                        label = output[vid_idx]['labels'][i].item() 
                        score = output[vid_idx]['scores'][i]

                        action_name = label_mapping.get(label, "Unknown")
                        
                        print(f"Action {i+1}:")
                        print(f"  - Start time: {start:.2f} sec")
                        print(f"  - End time: {end:.2f} sec")
                        print(f"  - Label: {action_name}")
                        print(f"  - Confidence: {score:.4f}")

                        # For Action total length Print 
                        # if label not in label_times:
                        #     label_times[label] = {'min_start': start, 'max_end': end}

                        # else:
                        #     label_times[label]['min_start'] = min(label_times[label]['min_start'], start)
                        #     label_times[label]['max_end'] = max(label_times[label]['max_end'], end)

                    else: 
                        continue
                # For Action total length Print 
                # for label, times in sorted(label_times.items(), key=lambda x: x[0]):  
                #         print(f"\nLabel {label}:")
                #         print(f"  - Start time: {times['min_start']:.2f} sec")
                #         print(f"  - End time: {times['max_end']:.2f} sec")

            else:
                print("No actions detected.")

    print("\nPrediction completed!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    parser = argparse.ArgumentParser(
      description='Evaluate action localization on a single video')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('video', type=str,
                        help='path to the video feature (.npy) file')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)