
import argparse
import os
import glob
import time
from pprint import pprint
import numpy as np

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
        Initialize dataset for a single video with strict preprocessing
        
        Args:
            video_feature_path (str): Path to the .npy feature file of the video
            cfg (dict): Configuration dictionary
        """
        # Configuration parameters
        input_dim = cfg['dataset']['input_dim']  # 2048 for I3D features
        feat_stride = cfg['dataset']['feat_stride']  # 4
        num_frames = cfg['dataset']['num_frames']  # 16
        max_seq_len = cfg['dataset']['max_seq_len']  # 2304
        
        # Load video features
        features = np.load(video_feature_path)
        
        # Dimension and stride validation
        assert features.ndim == 2, "Features must be a 2D array (frames x features)"
        assert features.shape[1] == input_dim, \
            f"Feature dimension mismatch. Expected {input_dim}, got {features.shape[1]}"
        
        # Resampling strategy to match stride and segment length
        resampled_features = []
        for start in range(0, features.shape[0], feat_stride):
            segment = features[start:start+num_frames]
            
            # Pad or truncate to exactly num_frames
            if segment.shape[0] < num_frames:
                pad_length = num_frames - segment.shape[0]
                segment = np.pad(
                    segment, 
                    ((0, pad_length), (0, 0)), 
                    mode='constant'
                )
            elif segment.shape[0] > num_frames:
                segment = segment[:num_frames]
            
            # Only add if we have a full segment
            if segment.shape[0] == num_frames:
                resampled_features.append(segment)
        
        # Convert to numpy array
        resampled_features = np.array(resampled_features)
        
        # Truncate to max sequence length
        if resampled_features.shape[0] > max_seq_len // num_frames:
            resampled_features = resampled_features[:max_seq_len // num_frames]
        
        # Convert to tensor 
        # Shape should be (num_segments, num_frames, input_dim)
        self.features = torch.from_numpy(resampled_features).float()
        
        # Create corresponding mask tensor
        self.mask = torch.ones(
            self.features.shape[0],  # num_segments
            self.features.shape[1],  # num_frames
            dtype=torch.bool
        )
        
        # Prepare video metadata
        self.video_id = os.path.splitext(os.path.basename(video_feature_path))[0]

    def __len__(self):
        return 1  # Single video dataset

    def __getitem__(self, idx):
        """
        Returns a list containing video info for model compatibility
        Ensures the correct tensor shape for processing
        """
        return [{
            'feats': self.features,  # Shape: (num_segments, num_frames, input_dim)
            'masks': self.mask,       # Shape: (num_segments, num_frames)
            'video_id': self.video_id,
        }]

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

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    # Create custom single video dataset
    val_dataset = SingleVideoDataset(args.video, cfg)
    
    # Create data loader (batch size 1)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0
    )

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
    
    # Extracting test configuration parameters
    voting_thresh = cfg['test_cfg']['voting_thresh']
    pre_nms_topk = cfg['test_cfg']['pre_nms_topk']
    max_seg_num = cfg['test_cfg']['max_seg_num']
    min_score = cfg['test_cfg']['min_score']
    
    with torch.no_grad():
        for video_list in val_loader:
            # Original input shape: [1, 144, 16, 2048]
            # Batch, Segments, Frames, Features
            B, num_segments, num_frames, feature_dim = video_list[0]['feats'].shape
            
            # Reshape features to match backbone expectation
            # Desired shape: [Batch, Features, Total_Sequence_Length]
            reshaped_feats = video_list[0]['feats'].permute(0, 3, 1, 2)  # [1, 2048, 144, 16]
            reshaped_feats = reshaped_feats.reshape(B, feature_dim, -1)  # [1, 2048, 2304]
            
            # Reshape mask to match feature sequence length
            reshaped_mask = video_list[0]['masks'].reshape(B, 1, -1)  # [1, 1, 2304]
            
            # Update video list
            video_list[0]['feats'] = reshaped_feats
            video_list[0]['masks'] = reshaped_mask
            
            # Debug prints
            print("Reshaped Feats Shape:", reshaped_feats.shape)
            print("Reshaped Mask Shape:", reshaped_mask.shape)
            
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
                    
                    for i in range(output[vid_idx]['segments'].shape[0]):
                        if(output[vid_idx]['scores'][i] > 0.2):
                            start = output[vid_idx]['segments'][i, 0]
                            end = output[vid_idx]['segments'][i, 1]
                            label = output[vid_idx]['labels'][i]
                            score = output[vid_idx]['scores'][i]
                            
                            print(f"Action {i+1}:")
                            print(f"  - Start time: {start:.2f} sec")
                            print(f"  - End time: {end:.2f} sec")
                            print(f"  - Label: {label}")
                            print(f"  - Confidence: {score:.4f}")
                        else: 
                            continue
                else:
                    print("No actions detected.")

    print("\nPrediction completed!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
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