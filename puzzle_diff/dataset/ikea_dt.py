from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class Sind_Vist_dt(Dataset):
    def __init__(self, download=False, split="train"):
        super().__init__()
        data_path = Path(f"datasets/sind/{split}.story-in-sequence.json")