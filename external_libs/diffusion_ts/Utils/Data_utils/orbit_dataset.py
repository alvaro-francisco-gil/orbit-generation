import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask

class OrbitDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        window=100, 
        period='train',
        neg_one_to_one=True,
        save2npy=True,
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate',
        distribution='geometric',
        mean_mask_length=3,
        scaler=None,
        **kwargs
    ):
        super().__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), 'predict_length and missing_ratio should not be set during training'
        
        self.window = window
        self.period = period
        self.auto_norm = neg_one_to_one
        self.save2npy = save2npy
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)
        
        # Load and preprocess data
        self.rawdata = np.load(data_path)
        print(f"[DEBUG] Raw data shape: {self.rawdata.shape}")
        
        # Handle different input shapes
        if self.rawdata.ndim == 2:
            # If data is (n_samples, n_features), reshape to (n_samples, window, n_features)
            n_samples = self.rawdata.shape[0] // self.window
            self.rawdata = self.rawdata[:n_samples * self.window].reshape(n_samples, self.window, -1)
        elif self.rawdata.ndim == 3:
            # If data is already (n_samples, window, n_features), use as is
            pass
        else:
            raise ValueError(f"Unexpected data shape: {self.rawdata.shape}")
            
        print(f"[DEBUG] Reshaped data: {self.rawdata.shape}")
        
        self.var_num = self.rawdata.shape[-1]
        self.sample_num = self.rawdata.shape[0]
        
        # Initialize scaler
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.rawdata.reshape(-1, self.var_num))
        else:
            self.scaler = scaler
            
        # Normalize data
        self.samples = self.normalize(self.rawdata)
        
        # Setup masking for test period
        self.pred_len, self.missing_ratio = predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data()
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError('Either missing_ratio or predict_length must be specified for test period')

    def normalize(self, rawdata):
        """Normalize the data using the scaler and optionally convert to [-1, 1] range."""
        print(f"[DEBUG] Normalizing data shape: {rawdata.shape}")
        d = self.__normalize(rawdata.reshape(-1, self.var_num))
        data = d.reshape(rawdata.shape)  # Reshape back to original shape
        
        if self.save2npy:
            # Save with consistent naming
            base_name = "periodic_orbits_prop100_cr3bp_nasa"
            np.save(os.path.join(self.dir, f"{base_name}_ground_truth_{self.window}_{self.period}.npy"), rawdata)
            if self.auto_norm:
                np.save(os.path.join(self.dir, f"{base_name}_norm_truth_{self.window}_{self.period}.npy"), 
                       unnormalize_to_zero_to_one(data))
            else:
                np.save(os.path.join(self.dir, f"{base_name}_norm_truth_{self.window}_{self.period}.npy"), data)
        
        return data

    def __normalize(self, rawdata):
        """Internal normalization function."""
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, data):
        """Convert normalized data back to original scale."""
        d = self.__unnormalize(data.reshape(-1, self.var_num))
        return d.reshape(data.shape)  # Reshape back to original shape

    def __unnormalize(self, data):
        """Internal unnormalization function."""
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        return self.scaler.inverse_transform(data)

    def mask_data(self, seed=2023):
        """Generate masks for missing data simulation."""
        masks = np.ones_like(self.samples)
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, 
                            self.style, self.distribution)
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"orbit_masking_{self.window}.npy"), masks)

        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.period == 'test':
            x = self.samples[idx, :, :]  # (window, n_features)
            m = self.masking[idx, :, :]  # (window, n_features)
            # Convert to torch tensor and transpose to (n_features, window)
            x = torch.from_numpy(x).float().transpose(0, 1)
            m = torch.from_numpy(m).float().transpose(0, 1)
            print(f"[DEBUG] Dataset test sample {idx} shape: {x.shape}, values range: [{x.min():.3f}, {x.max():.3f}]")
            return x, m
        
        x = self.samples[idx, :, :]  # (window, n_features)
        # Convert to torch tensor and transpose to (n_features, window)
        x = torch.from_numpy(x).float().transpose(0, 1)
        print(f"[DEBUG] Dataset train sample {idx} shape: {x.shape}, values range: [{x.min():.3f}, {x.max():.3f}]")
        return x

    def __len__(self):
        """Return the total number of samples."""
        return self.sample_num 
