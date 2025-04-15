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

# CONFIG for visualization
clip_len = 0.1
gap = 1
save_dir = './tal_viz_output_one_jpg_v2'
fontsize = 24

os.makedirs(save_dir, exist_ok=True)

color_gt = '#1f77b4'
color_pred = '#ff7f0e'

font_prop_label = font_manager.FontProperties(size=14)
font_prop_title = font_manager.FontProperties(size=18, weight='bold')

# HELPER: check if timestamp is in a segment
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import os

def generate_video_plot_one_jpg_v2(
    video_id: str,
    pred_json: Dict,
    gt_json: Dict,
    video_path: str,
    min_segment_length: float = 0.1,
    gap: float = 1.0,
    clip_len: float = 1.0
) -> None:
    """
    Generate a visualization plot for video annotations comparing ground truth and predictions.
    
    Args:
        video_id: Identifier for the video
        pred_json: Predicted annotations
        gt_json: Ground truth annotations
        video_path: Path to video file
        save_dir: Directory to save output plot
        min_segment_length: Minimum length for segments
        gap: Time gap between frames
        clip_len: Length of video clip
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract data
    gt_annos = gt_json[args.video]['annotations']
    pred_annos = pred_json[args.video]
    duration = gt_json[args.video]['duration']

    # Calculate frame times
    fig_num = int(duration / gap)
    times = np.linspace(0, duration - clip_len, fig_num)

    # Load video frames
    frames = _load_frames(video_path, times)

    # Initialize figure
    fig = plt.figure(figsize=(fig_num * 1.5, 8), constrained_layout=True)
    gs = fig.add_gridspec(7, fig_num, height_ratios=[5, 0.4, 0.6, 0.4, 1, 0.4, 1])

    # Plot frames
    _plot_frames(fig, gs, times, frames, gt_annos, pred_annos)

    # Plot ground truth and prediction bars
    _plot_annotation_bars(fig, gs, duration, gt_annos, pred_annos)

    # Save as JPG and PDF
    jpg_path = os.path.join(save_dir, f"{video_id}_final_clean.jpg")
    pdf_path = os.path.join(save_dir, f"{video_id}_final_clean.pdf")

    plt.savefig(jpg_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[✅ Saved]: {jpg_path}")
    print(f"[✅ Saved]: {pdf_path}")


def _load_frames(video_path: str, times: np.ndarray) -> List[np.ndarray]:
    """Load video frames at specified times."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        rval, frame = cap.read()
        if not rval:
            frames.append(np.ones((100, 100, 3), dtype=np.uint8) * 255)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames

def _is_in_segments(t: float, segments: List[List[float]]) -> bool:
    """Check if time t falls within any segment."""
    return any(start <= t <= end for start, end in segments)

def _plot_frames(
    fig: plt.Figure,
    gs: plt.GridSpec,
    times: np.ndarray,
    frames: List[np.ndarray],
    gt_annos: List[Dict],
    pred_annos: List[Dict]
) -> None:
    """Plot video frames with appropriate highlighting."""
    for i, (t, frame) in enumerate(zip(times, frames)):
        ax = fig.add_subplot(gs[0, i])
        
        gt_hit = _is_in_segments(t, [a['segment'] for a in gt_annos])
        pred_hit = _is_in_segments(t, [a['segment'] for a in pred_annos])

        # Determine frame highlight color
        color = 'green' if gt_hit and pred_hit else 'red' if gt_hit else 'black' if pred_hit else None

        # Apply zoom effect
        scale_factor = 1.35 if color == 'green' else 1.25 if gt_hit else 1.0
        frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, 
                                 interpolation=cv2.INTER_LINEAR)

        ax.imshow(frame_resized)
        ax.axis('off')

        if color:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3.5)

        ax.set_title(f"{t:.1f}s", fontsize=12, color=color or 'gray', pad=8)

def _plot_annotation_bars(
    fig: plt.Figure,
    gs: plt.GridSpec,
    duration: float,
    gt_annos: List[Dict],
    pred_annos: List[Dict]
) -> None:
    """Plot ground truth and prediction annotation bars."""
    
    def draw_segment_bar(
        ax: plt.Axes,
        segment: Dict,
        color: str,
        ypos: float,
        label_above: bool
    ) -> None:
        start, end = segment['segment']
        width = end - start
        label = segment['label']

        # Draw rounded rectangle
        ax.add_patch(patches.FancyBboxPatch(
            (start, ypos), width, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.9
        ))

        # Center label in bar
        ax.text(
            (start + end) / 2, ypos + 0.2,
            label,
            ha='center', va='center',
            fontsize=12, weight='bold',
            color='white',
            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=2)
        )

        # Add timestamps
        time_y = ypos - 0.3 if label_above else ypos + 0.5
        ax.text(start, time_y, f"{start:.1f}", ha='center', fontsize=10, color='red')
        ax.text(end, time_y, f"{end:.1f}", ha='center', fontsize=10, color='black')

    # Ground Truth Bar
    ax_gt = fig.add_subplot(gs[4, :])
    ax_gt.set_xlim(0, duration)
    ax_gt.set_ylim(0, 1)
    ax_gt.axis('off')
    
    # Improved label positioning
    ax_gt.text(-0.05 * duration, 0.5, "Ground Truth", 
              fontsize=14, weight='bold', va='center', 
              bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=5))

    for seg in gt_annos:
        draw_segment_bar(ax_gt, seg, '#f4a261', ypos=0.3, label_above=True)

    # Prediction Bar
    ax_pred = fig.add_subplot(gs[6, :])
    ax_pred.set_xlim(0, duration)
    ax_pred.set_ylim(0, 1)
    ax_pred.axis('off')
    
    # Improved label positioning
    ax_pred.text(-0.05 * duration, 0.5, "Our", 
                fontsize=14, weight='bold', va='center',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, pad=5))

    for seg in pred_annos:
        draw_segment_bar(ax_pred, seg, '#90be6d', ypos=0.3, label_above=False)



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
        # print(f"Original features shape: {features.shape}")
        
        # Ensure the format is (frames, features)
        assert features.ndim == 2, "Features must be a 2D array (frames x features)"
        assert features.shape[1] == input_dim, \
            f"Feature dimension mismatch. Expected {input_dim}, got {features.shape[1]}"
        
        # Convert to tensor and transpose to [features, frames]
        # This matches your desired format where shape is [2048, frames]
        self.features = torch.from_numpy(features).float().transpose(0, 1)
        # print(f"Transposed features shape: {self.features.shape}")
        
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
    # print(f"Shape of video features: {video_data['feats'].shape}")
    # print(f"Mask shape: {video_data['masks'].shape}")

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
   
   
    pred_json = {}

    json_path = '/content/CSE499A--BBFormer/data/gt.json'  
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    gt_json = json_data['GT']


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
        # print("Input Feature Shape:", video_list[0]['feats'].shape)
        # print("Input Mask Shape:", video_list[0]['masks'].shape)

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
                pred_json[args.video] = []
                for i in range(output[vid_idx]['segments'].shape[0]):
                    if output[vid_idx]['scores'][i] > 0.32:
                        start = output[vid_idx]['segments'][i, 0].item()
                        end = output[vid_idx]['segments'][i, 1].item()
                        label_id = output[vid_idx]['labels'][i].item()
                        score = output[vid_idx]['scores'][i].item()

                        # Map label ID to action name
                        action_name = label_mapping.get(label_id, "Unknown")

                        pred_json[args.video].append({
                            'label': action_name,  # Use action name instead of label ID
                            'segment': [start, end],
                            'score': score
                        })
                        print(f"Action {i+1}:")
                        print(f"  - Start time: {start:.2f} sec")
                        print(f"  - End time: {end:.2f} sec")
                        print(f"  - Label: {action_name}")
                        print(f"  - Confidence: {score:.4f}")
            else:
                print("No actions detected.")

    print("\nPrediction completed!")

    generate_video_plot_one_jpg_v2(args.video, pred_json, gt_json, args.video_path, min_segment_length=0.1)
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
    parser.add_argument('video_path', type=str,
                        help='path to the original video file for visualization')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)



















# import argparse
# import os
# import glob
# import time
# from pprint import pprint
# import numpy as np
# import json
# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image
# from matplotlib import font_manager

# # torch imports
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.utils.data

# # our code
# from libs.core import load_config
# from libs.datasets import make_dataset, make_data_loader
# from libs.modeling import make_meta_arch
# from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed

# # CONFIG for visualization
# clip_len = 0.1
# gap = 1
# save_dir = './tal_viz_output_one_jpg_v2'
# fontsize = 24

# os.makedirs(save_dir, exist_ok=True)

# color_gt = '#1f77b4'
# color_pred = '#ff7f0e'

# font_prop_label = font_manager.FontProperties(size=14)
# font_prop_title = font_manager.FontProperties(size=18, weight='bold')

# # HELPER: check if timestamp is in a segment
# def is_in_segments(t, segments):
#     return any(start <= t <= end for (start, end) in segments)

# # HELPER: apply visual template to frame
# def apply_template(frame, template_path):
#     frame = Image.fromarray(frame)
#     template = Image.open(template_path).convert("RGBA")
#     width, height = frame.size
#     new_size = (width, int(height * 1.3))
#     white_bg = Image.new('RGBA', new_size, (255, 255, 255, 255))
#     white_bg.paste(frame, (0, int(height * 0.15)))
#     template = template.resize(new_size, Image.ANTIALIAS)
#     result = Image.alpha_composite(white_bg, template)
#     return result.convert('RGB')

# # DRAW BAR WITH LABELS
# def draw_labeled_bar(ax, segments, duration, color, label_prefix):
#     for seg in segments:
#         start, end = seg['segment']
#         label = seg['label']
#         width = end - start
#         ax.barh(y=0, width=width, left=start, height=0.6, color=color, edgecolor='black', alpha=0.8)
#         center = start + width / 2
#         ax.text(center, 0, f"{label}", fontsize=12, va='center', ha='center', color='white', weight='bold')
#     ax.set_xlim(0, duration)
#     ax.set_yticks([])
#     ax.set_xticks(np.linspace(0, duration, 6))
#     ax.set_ylabel(label_prefix, rotation=0, labelpad=30, fontsize=14, weight='bold')
#     ax.invert_yaxis()

# # MAIN VISUALIZER
# def generate_video_plot_one_jpg_v2(video_id, pred_json, gt_json, video_path, min_segment_length=0.1):
#     print(f"video id::::{video_id}")
#     print(f"pred_json::::{pred_json}")
#     print(f"gt_json::::{gt_json}")
#     print(f"video_path::::{video_path}")

#     gt_annos = gt_json[video_id]['annotations']
#     print(f"gt_annos print::::{gt_annos}")
#     pred_annos = pred_json[args.video]
#     print(f"pred_annos print::::{pred_annos}")
#     duration = gt_json[video_id]['duration']
#     fig_num = int(duration / gap)
#     times = np.linspace(0, duration - clip_len, fig_num)

#     gt_segments = [a for a in gt_annos]
#     print(f"gt_segment print::::{gt_segments}")
#     # pred_segments = [a for a in pred_annos if (a['segment'][1] - a['segment'][0]) >= min_segment_length]
#     pred_segments = [a for a in pred_annos]
#     print(f"pred_segment print::::{pred_segments}")


#     # 🎞️ Extract video frames
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     for t in times:
#         cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
#         rval, frame = cap.read()
#         if not rval:
#             frames.append(np.ones((100, 100, 3), dtype=np.uint8) * 255)
#             continue
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(np.array(frame))
#     cap.release()

#     # 🖼️ Create stacked layout
#     fig_height = 10
#     fig = plt.figure(figsize=(fig_num * 1.4, fig_height))
#     gs = fig.add_gridspec(5, fig_num, height_ratios=[4, 0.3, 1, 0.3, 1])

#     # 🎞️ Row 1: Frames
#     for i, (t, frame) in enumerate(zip(times, frames)):
#         ax = fig.add_subplot(gs[0, i])
#         ax.imshow(frame)
#         ax.axis('off')

#         gt_hit = is_in_segments(t, [a['segment'] for a in gt_annos])
#         pred_hit = is_in_segments(t, [a['segment'] for a in pred_segments])

#         # Frame title color logic
#         color = 'black'
#         if gt_hit and pred_hit:
#             color = 'green'
#         elif gt_hit:
#             color = 'blue'
#         elif pred_hit:
#             color = 'red'

#         ax.set_title(f"{t:.1f}s", fontsize=fontsize, color=color)

#     # 📏 Row 3: GT Bar
#     ax_gt = fig.add_subplot(gs[2, :])
#     draw_labeled_bar(ax_gt, gt_segments, duration, color_gt, label_prefix='GT')

#     # 📏 Row 5: Pred Bar
#     ax_pred = fig.add_subplot(gs[4, :])
#     draw_labeled_bar(ax_pred, pred_segments, duration, color_pred, label_prefix='Pred')

#     plt.tight_layout()
#     save_path = os.path.join(save_dir, f"{video_id}_full_labeled.jpg")
#     plt.savefig(save_path, dpi=100)
#     plt.close()
#     print(f"[✅ Saved]: {save_path}")



# class SingleVideoDataset(torch.utils.data.Dataset):
#     def __init__(self, video_feature_path, cfg):
#         """
#         Initialize dataset for a single video
        
#         Args:
#             video_feature_path (str): Path to the .npy feature file of the video
#             cfg (dict): Configuration dictionary
#         """
#         # Configuration parameters
#         input_dim = cfg['dataset']['input_dim']  # 2048 for I3D features
        
#         # Load video features
#         features = np.load(video_feature_path)
#         print(f"Original features shape: {features.shape}")
        
#         # Ensure the format is (frames, features)
#         assert features.ndim == 2, "Features must be a 2D array (frames x features)"
#         assert features.shape[1] == input_dim, \
#             f"Feature dimension mismatch. Expected {input_dim}, got {features.shape[1]}"
        
#         # Convert to tensor and transpose to [features, frames]
#         # This matches your desired format where shape is [2048, frames]
#         self.features = torch.from_numpy(features).float().transpose(0, 1)
#         print(f"Transposed features shape: {self.features.shape}")
        
#         # Create corresponding mask tensor (all True)
#         self.mask = torch.ones(self.features.shape[1], dtype=torch.bool)
        
#         # Prepare video metadata
#         self.video_id = os.path.splitext(os.path.basename(video_feature_path))[0]

#     def __len__(self):
#         return 1  # Single video dataset

#     def __getitem__(self, idx):
#         """
#         Returns the video features directly in the required format
#         """
#         # Return features directly in the [2048, frames] format without extra dimensions
#         return {
#             'feats': self.features,         # Shape: [2048, frames]
#             'masks': self.mask,             # Shape: [frames]
#             'video_id': self.video_id,
#         }

# def main(args):
#     """0. load config"""
#     # sanity check
#     if os.path.isfile(args.config):
#         cfg = load_config(args.config)
#     else:
#         raise ValueError("Config file does not exist.")
#     assert len(cfg['val_split']) > 0, "Test set must be specified!"

#     # Checkpoint file handling
#     if ".pth.tar" in args.ckpt:
#         assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
#         ckpt_file = args.ckpt
#     else:
#         assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
#         if args.epoch > 0:
#             ckpt_file = os.path.join(
#                 args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
#             )
#         else:
#             ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
#             ckpt_file = ckpt_file_list[-1]
#         assert os.path.exists(ckpt_file)

#     # Validate input video feature file
#     assert os.path.isfile(args.video), f"Video feature file {args.video} does not exist!"
#     assert os.path.isfile(args.video_path), f"Video file {args.video_path} does not exist!"

#     if args.topk > 0:
#         cfg['model']['test_cfg']['max_seg_num'] = args.topk
#     pprint(cfg)

#     """1. fix all randomness"""
#     # fix the random seeds (this will fix everything)
#     _ = fix_random_seed(0, include_cuda=True)

#     """2. create dataset / dataloader"""
#     # Create custom single video dataset
#     val_dataset = SingleVideoDataset(args.video, cfg)
    
#     # Get a single video data sample - no need for DataLoader here
#     video_data = val_dataset[0]
#     print(f"Shape of video features: {video_data['feats'].shape}")
#     print(f"Mask shape: {video_data['masks'].shape}")

#     """3. create model"""
#     # model
#     model = make_meta_arch(cfg['model_name'], **cfg['model'])
#     # not ideal for multi GPU training, ok for now
#     model = nn.DataParallel(model, device_ids=cfg['devices'])

#     """4. load ckpt"""
#     print("=> loading checkpoint '{}'".format(ckpt_file))
#     # load ckpt, reset epoch / best rmse
#     checkpoint = torch.load(
#         ckpt_file,
#         map_location = lambda storage, loc: storage.cuda(int(cfg['devices'][0].split(':')[1]))
#     )

#     # load ema model instead
#     print("Loading from EMA model ...")
#     model.load_state_dict(checkpoint['state_dict_ema'])
#     del checkpoint

#     """5. Predict on single video"""
#     print(f"\nStart predicting actions in video: {args.video}")
#     model.eval()
    
#     # Extract test configuration parameters
#     test_cfg = cfg['model'].get('test_cfg', {})
#     voting_thresh = test_cfg.get('voting_thresh', 0.7)
#     pre_nms_topk = test_cfg.get('pre_nms_topk', 2000)
#     max_seg_num = test_cfg.get('max_seg_num', 200)
#     min_score = test_cfg.get('min_score', 0.001)
   
   
#     pred_json = {}

#     json_path = '/content/CSE499A--BBFormer/data/gt.json'  
#     with open(json_path, 'r') as f:
#         json_data = json.load(f)

#     gt_json = json_data['GT']


#     label_mapping = {
#         0: "BaseballPitch",
#         1: "BasketballDunk",
#         2: "Billiards",
#         3: "CleanAndJerk",
#         4: "CliffDiving",
#         5: "CricketBowling",
#         6: "CricketShot",
#         7: "Diving",
#         8: "FrisbeeCatch",
#         9: "GolfSwing",
#         10: "HammerThrow",
#         11: "HighJump",
#         12: "JavelinThrow",
#         13: "LongJump",
#         14: "PoleVault",
#         15: "Shotput",
#         16: "SoccerPenalty",
#         17: "TennisSwing",
#         18: "ThrowDiscus",
#         19: "VolleyballSpiking",
#     }
#     with torch.no_grad():
#         # Create a list with just one element to match the model's expectation
#         # This is crucial - pass exactly what the model expects
#         video_list = [video_data]
        
#         # Debug prints
#         print("Input Feature Shape:", video_list[0]['feats'].shape)
#         print("Input Mask Shape:", video_list[0]['masks'].shape)

#         feats = video_list[0]['feats']



#         video_data = {
#                 'video_id': 'testVideo',
#                 'feats': feats,
#                 'fps': 30.0,
#                 'duration': 120.0,
#                 'feat_stride': 4,
#                 'feat_num_frames': 16
#             }

#         video_list = [video_data]

#         print(video_list)
        
        
#         # Pass to model (they expect a list of dictionaries)
#         with torch.no_grad():
#             output = model(video_list)
        
        
#         # Process and print predictions
#         num_vids = len(output)
#         for vid_idx in range(num_vids):
#             if output[vid_idx]['segments'].shape[0] > 0:
#                 pred_json[args.video] = []
#                 for i in range(output[vid_idx]['segments'].shape[0]):
#                     if output[vid_idx]['scores'][i] > 0.32:
#                         start = output[vid_idx]['segments'][i, 0].item()
#                         end = output[vid_idx]['segments'][i, 1].item()
#                         label_id = output[vid_idx]['labels'][i].item()
#                         score = output[vid_idx]['scores'][i].item()

#                         # Map label ID to action name
#                         action_name = label_mapping.get(label_id, "Unknown")

#                         pred_json[args.video].append({
#                             'label': action_name,  # Use action name instead of label ID
#                             'segment': [start, end],
#                             'score': score
#                         })
#                         print(f"Action {i+1}:")
#                         print(f"  - Start time: {start:.2f} sec")
#                         print(f"  - End time: {end:.2f} sec")
#                         print(f"  - Label: {action_name}")
#                         print(f"  - Confidence: {score:.4f}")
#             else:
#                 print("No actions detected.")

#     print("\nPrediction completed!")

#     pr = pred_json[args.video]

#     print(f"after p complete print pred json:{ pr}")

#     # Generate video plot after predictions
#     # generate_video_plot_one_jpg_v2('video_test_0000004', pred_json, gt_json, args.video_path)
#     # generate_video_plot_one_jpg_v2('video_test_0000004', pred_json, gt_json, args.video_path, score_threshold=0.1)
#     generate_video_plot_one_jpg_v2('video_test', pred_json, gt_json, args.video_path, min_segment_length=0.1)
#     return

# ################################################################################
# if __name__ == '__main__':
#     """Entry Point"""
#     parser = argparse.ArgumentParser(
#       description='Evaluate action localization on a single video')
#     parser.add_argument('config', type=str, metavar='DIR',
#                         help='path to a config file')
#     parser.add_argument('ckpt', type=str, metavar='DIR',
#                         help='path to a checkpoint')
#     parser.add_argument('video', type=str,
#                         help='path to the video feature (.npy) file')
#     parser.add_argument('video_path', type=str,
#                         help='path to the original video file for visualization')
#     parser.add_argument('-epoch', type=int, default=-1,
#                         help='checkpoint epoch')
#     parser.add_argument('-t', '--topk', default=-1, type=int,
#                         help='max number of output actions (default: -1)')
#     parser.add_argument('-p', '--print-freq', default=10, type=int,
#                         help='print frequency (default: 10 iterations)')
#     args = parser.parse_args()
#     main(args)