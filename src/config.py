"""
Configurations for the system.
"""

BATCH_SIZE = 32
IMG_SIZE = 224  # 224 # 416 # 105
DESCRIPTOR_SIZE = 128  # 4096  # 128
NUM_CHANNELS = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE) if NUM_CHANNELS == 1 else (IMG_SIZE, IMG_SIZE, 3)
