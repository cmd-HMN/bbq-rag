from collections.abc import Mapping
from typing import Dict

class Img(Mapping):
    def __init__(self, dataset, idx: Dict[str, int]):
        self.dataset = dataset
        self.idx = idx

    def __getitem__(self, key: str | int):
        if isinstance(key, int):
            return self.dataset[key]["image"]

        if key not in self.idx:
            raise KeyError(key)

        return self.dataset[self.idx[key]]["image"]

    def __len__(self):
        return len(self.idx)

    def __iter__(self):
        return iter(self.idx)
