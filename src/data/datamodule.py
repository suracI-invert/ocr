from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from src.data.components.datasets import OCRDataset
from src.models.tokenizer import Tokenizer

from sklearn.model_selection import train_test_split

class OCRDataModule(LightningDataModule):
    def __init__(self, data_dir: str,
                map_file: str,
                tokenizer: Tokenizer,
                test_dir: str = None,
                train_val_split: Tuple[int, int] = None,
                h: int = 70,
                min_w: int = 100,
                max_w: int = 300,
                batch_size: int = 1,
                num_workers: int = 0,
                pin_memory: bool = False,
                transforms= None,
                collate_fn= None,
                sampler= None,
            ) -> None:
        super().__init__()

        self.save_hyperparameters(logger= False, ignore= ['tokenizer'])
        self.tokenizer = tokenizer
        self.prepare_data_per_node = False

        self.data_train: Optional[Dataset] = None
        self.data_valid: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.mapping = []
        with open(map_file, encoding= 'utf8') as f:
            for l in f.readlines():
                path, label = l.strip().split()
                self.mapping.append((path, label))

        self.setup_called = False

    @property
    def num_classes(self) -> int:
        pass

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.setup_called:
            self.setup_called = True

            if not self.hparams.train_val_split:
                self.data_train = OCRDataset(self.hparams.data_dir,
                                             self.mapping, h= self.hparams.h, min_w= self.hparams.min_w, max_w= self.hparams.max_w,
                                             tokenizer= self.tokenizer, transforms= self.hparams.transforms)
            else:
                train_mapping, val_mapping = train_test_split(self.mapping, 
                                                            train_size= self.hparams.train_val_split[0],
                                                            test_size= self.hparams.train_val_split[1],
                                                            random_state= 42)
                self.data_train = OCRDataset(self.hparams.data_dir,
                                             train_mapping, h= self.hparams.h, min_w= self.hparams.min_w, max_w= self.hparams.max_w,
                                             tokenizer= self.tokenizer, transforms= self.hparams.transforms)
                self.data_valid = OCRDataset(self.hparams.data_dir,
                                             val_mapping, h= self.hparams.h, min_w= self.hparams.min_w, max_w= self.hparams.max_w,
                                             tokenizer= self.tokenizer, transforms= self.hparams.transforms)

            if self.hparams.test_dir and not self.data_test:
                self.data_test = OCRDataset(self.hparams.test_dir, test_data= True, 
                                            h= self.hparams.h, min_w= self.hparams.min_w, max_w= self.hparams.max_w, 
                                            transforms= self.hparams.transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset= self.data_train,
            batch_size= self.hparams.batch_size,
            num_workers= self.hparams.num_workers,
            collate_fn= self.hparams.collate_fn,
            sampler= self.hparams.sampler(self.data_train, self.hparams.batch_size, True),
            pin_memory= self.hparams.pin_memory,
            drop_last= True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset= self.data_valid,
            batch_size= 1,
            num_workers= self.hparams.num_workers,
            collate_fn= self.hparams.collate_fn,
            pin_memory= self.hparams.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset= self.data_test,
            batch_size= 1,
            num_workers= self.hparams.num_workers,
            pin_memory= self.hparams.pin_memory,
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
