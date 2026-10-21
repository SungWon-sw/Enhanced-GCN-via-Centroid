"""Cache deterministic VN preprocessing without duplicating the PPA graphs."""

import hashlib
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import Dataset

import utils_vn_connect
from utils_vn_connect import VNData, add_ppa_virtual_nodes


class CachedVNDataset(Dataset):
    """Build VN fields on first access, then reuse them across epochs/workers/runs.

    The wrapped dataset must contain untransformed graphs in their original order.
    Cache identity includes the VN implementation and processed-file metadata.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        fingerprint = hashlib.sha256(b'ppa-vn-cache-v1')
        fingerprint.update(Path(utils_vn_connect.__file__).read_bytes())
        for filename in dataset.processed_paths:
            path = Path(filename).resolve()
            stat = path.stat()
            fingerprint.update(str((str(path), stat.st_size, stat.st_mtime_ns)).encode())
        fingerprint.update(str(len(dataset)).encode())
        self.cache_dir = (
            Path(dataset.processed_dir) / 'vn_cache' / fingerprint.hexdigest()[:20]
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        data = self.dataset[index]
        # Bound the number of files in each directory for the 158k-graph dataset.
        path = self.cache_dir / str(index // 1000) / f'{index}.pt'
        try:
            fields = torch.load(path, map_location='cpu', weights_only=True)
        except FileNotFoundError:
            result = add_ppa_virtual_nodes(data)
            fields = {
                key: getattr(result, key)
                for key in ('vn_edge_index')
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            # Publish only complete files, including when multiple workers/runs
            # happen to request the same uncached graph simultaneously.
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as f:
                temporary_path = Path(f.name)
            try:
                torch.save(fields, temporary_path)
                os.replace(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)
            return result

        result = VNData(**data.to_dict())
        result.num_nodes = data.num_nodes
        result.x = torch.zeros(data.num_nodes, dtype=torch.long)
        for key, value in fields.items():
            setattr(result, key, value)
        return result
