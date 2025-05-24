import numpy as np
from torch.utils.data import Dataset
import math
# import torch

# EMBEDDING_DIM = 768

class DataGenerator(Dataset):
    def __init__(self, x, y): #, window_size
        'Initialization'
        self.x = x
        self.y = y
        # self.window_size = window_size
        # self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches'
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]  # No windowing
        y = self.y[index]
        return x,y # torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # ## this output one pad data with shape( window_size, embedding_shape)
    # def __getitem__(self, index):
    #     'Generate one batch of data'
    #     # # x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
    #     # # y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
    #     x = np.array(self.x[index])  # Ensure it's a NumPy array
    #     y = np.array(self.y[index])
    #     # # print(x.shape)  # (768,)
        
    #     # # x = pad_sequences(x, dtype='object', padding='post',
    #     # #                   value=np.zeros(EMBEDDING_DIM)).astype(np.float32)

    #     # # # 最大设置为40暂定不需要最大number的
    #     # # num_tokens = x.shape[0]
    #     # # # print('x.shape[0]:', num_tokens)

    #     # # mix_num_boxes = min(int(num_tokens), self.window_size)
    #     # # # # mix_boxes_pad = np.zeros((self._max_region_num, 5))
        
    #     # # # mix_features_pad = np.zeros((self.window_size, 768))
    #     # # # mix_features_pad[:mix_num_boxes,:] = x[:mix_num_boxes,:]
        
    #     # # mix_features_pad = np.zeros((self.window_size, ))  # TF, 1/4/2025
    #     # # mix_features_pad[:mix_num_boxes] = x[:mix_num_boxes]
        
    #     # # x = mix_features_pad
    #     # # # x = pad_sequence([torch.from_numpy(np.array(x)) for x in input_x], batch_first=True).float()
    #     # # # print('x 类型：',type(x))
    #     return x, y


    # def __init__(self, data, window_size=10, use_masking=False, mask_ratio=0.15, augment_fn=None, labels=None):
    #     """
    #     Args:
    #         data: Tensor or ndarray of shape (total_len, feature_dim)
    #         window_size: How many time steps per sample
    #         use_masking: Whether to apply random masking
    #         mask_ratio: Portion of the sequence to randomly mask (if use_masking is True)
    #         augment_fn: Optional function to apply to input (like noise injection, etc.)
    #         labels: Optional anomaly labels (for validation or visualization)
    #     """
    #     if isinstance(data, list):
    #         data = torch.tensor(data, dtype=torch.float32)
    #     self.data = data
    #     self.window_size = window_size
    #     self.use_masking = use_masking
    #     self.mask_ratio = mask_ratio
    #     self.augment_fn = augment_fn
    #     self.labels = labels

    # def __len__(self):
    #     return self.data.shape[0] - self.window_size + 1

    # def __getitem__(self, idx):
    #     x = self.data[idx : idx + self.window_size]  # (window_size, feature_dim)
    #     label = None if self.labels is None else self.labels[idx + self.window_size - 1]

    #     # Optional masking (like input dropout or MLM-style masking)
    #     if self.use_masking:
    #         x = self.random_masking(x)

    #     return (x, label) if label is not None else x

    # def random_masking(self, x):
    #     # x: (window_size, feature_dim)
    #     mask = torch.rand_like(x[:, 0]) < self.mask_ratio
    #     x[mask] = 0.0  # Or replace with a learned [MASK] embedding
    #     return x