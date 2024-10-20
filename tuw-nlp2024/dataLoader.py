
from torch.utils.data import Dataset
import torch
from enum import Enum
from pathlib import Path

class Language(Enum):
    EN = "EN"
    RU = "RU"
    BG = "BG"
    PT = "PT"
    HI = "HI"

class Subset(Enum):
    TRAIN = 1
    TEST = 2
    EVAL = 3

class AnotationLevel(Enum):
    NESTED = 1
    TOP_LEVEL = 2
    BOTH = 3

class PropagandaDataset(Dataset):
    def __init__(self,
            base_path: Path,
            language: Language,
            subset: Subset,
            level: AnotationLevel,
            transform = None,
            target_transform = None
        ):

        self.subset = subset
        self.base_path = base_path
        self.language = language
        self.level = level
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = self.base_path / self.language.value
        self.__load_labels(base_path/"labels.txt")
        self.__load_data(base_path/"training_data_16_October_release"/language.value/"subtask-2-annotations.txt")

    def __load_labels(self, path: Path):
        top_level_annotations = []
        annotations = []
        with open(path, "r") as file:
            curr_top_level = None
            for line in file.readlines():
                line = line.strip()
                if line[0] != "-":
                    curr_top_level = line
                    top_level_annotations.append(curr_top_level)
                    annotations.append(f"{curr_top_level}: Other")
                else:
                    annotations.append(f"{curr_top_level}: {line[2:]}")
        annotations.append("Other")
        top_level_annotations.append("Other")

        self.num_top_level_labels = len(top_level_annotations)
        self.num_nested_labels = len(annotations)
        self.top_level_labels = top_level_annotations
        self.nested_labels = annotations
        self.tlti = dict(zip(self.top_level_labels, range(0,self.num_top_level_labels)))
        self.nlti = dict(zip(self.nested_labels, range(0,self.num_nested_labels)))
    
    def __load_data(self, path:Path) -> list[str]:
        with open(path, "r") as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        features = None
        top_level_vec = None
        nested_vec = None
        file, top_level, nested = line.split("\t")
        

        if self.level in (AnotationLevel.TOP_LEVEL, AnotationLevel.BOTH):
            top_level_vec = torch.zeros(self.num_top_level_labels)
            if top_level == "Other":
                idx = self.tlti["Other"]
                top_level_vec[idx] = 1
            else:
                for el in top_level.split(";"):
                    el = el.split(":")[1].strip()
                    idx = self.tlti[el]
                    top_level_vec[idx] = 1


        if self.level in (AnotationLevel.NESTED, AnotationLevel.BOTH):
            nested_vec = torch.zeros(self.num_nested_labels)
            if nested.strip() == "Other":
                idx = self.nlti["Other"]
                nested_vec[idx] = 1
            else:
                for el in nested.split(";"):
                    _, top, sub = el.split(":")
                    category = ": ".join([top.strip(), sub.strip()])
                    idx = self.nlti[category]
                    print(category, idx)
                    nested_vec[idx] = 1

        return features, top_level_vec, nested_vec
        
    

base = Path("/home/tibor/Documents/msc-datascience/2024w/TUW-NLP2024/data")
dataset = PropagandaDataset(base, Language.EN, Subset.TRAIN, AnotationLevel.BOTH)
print(dataset.nlti)
print(dataset[2][2])
