# Start with importing the necessary libraries
import torch
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import pandas as pd
import os
import shutil
from torchvision.transforms import Compose, ToTensor, Resize
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torchinfo import summary

# Hyperparameters and Constants
BATCH_SIZE = 32
NUMBER_OF_LABELS = 101
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_OF_WORKERS = os.cpu_count()