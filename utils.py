import collections
import cv2
import torch
import numpy as np
from collections import deque

def img_crop(img):
    return img[30:-60,:,:]

# GENERAL Atari preprocessing steps
def downsample(img):
    # We will take only half of the image resolution
    return img[::2, ::2]

def transform_reward(reward):
    return reward  # Avoiding oversimplification of rewards

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

# Normalize grayscale image from -1 to 1.
def normalize_grayscale(img):
    return (img - 128) / 128 - 1  

def preprocess_state(state, device, image_shape=(80, 80)):
    state = np.array(state)
    state = img_crop(state)
    state = downsample(state)
    state = to_grayscale(state)
    state = normalize_grayscale(state)
    
    return torch.tensor(state, dtype=torch.float32)

class FrameStack:
    def __init__(self, num_frames, height, width):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        self.shape = (num_frames, height, width)
    
    def reset(self, frame):
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self._get_stacked_frames()
    
    def append(self, frame):
        self.frames.append(frame)
        return self._get_stacked_frames()
    
    def _get_stacked_frames(self):
        # Convert deque to numpy array
        stacked_frames = np.array(self.frames)
        # Check if an extra channel dimension exists and remove it
        if stacked_frames.ndim == 4 and stacked_frames.shape[1] == 1:
            stacked_frames = np.squeeze(stacked_frames, axis=1)
        # Stack frames along the first axis
        stacked_frames = np.stack(self.frames, axis=0)
        return torch.tensor(stacked_frames, dtype=torch.float32)
