from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.datasets import OCRDataset
from src.models.tokenizer import Tokenizer


class OCRDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        map_file: str,
        tokenizer: Tokenizer,
        test_dir: str = None,
        train_val_split: Tuple[int, int] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        transforms=None,
        collate_fn=None,
        sampler=None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=True, ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.prepare_data_per_node = False

        self.data_train: Optional[Dataset] = None
        self.data_valid: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.setup_called = False

    @property
    def num_classes(self) -> int:
        pass

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.setup_called:
            self.setup_called = True
            dataset = OCRDataset(
                self.hparams.data_dir,
                map_file=self.hparams.map_file,
                tokenizer=self.tokenizer,
                transforms=self.hparams.transforms,
            )
            if not self.hparams.train_val_split:
                self.data_train = dataset
            else:
                self.data_train, self.data_valid = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_val_split,
                    generator=Generator().manual_seed(42),
                )
            if self.hparams.test_dir and not self.data_test:
                self.data_test = OCRDataset(
                    self.hparams.test_dir, test_data=True, transforms=self.hparams.transforms
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.hparams.collate_fn,
            sampler=self.hparams.sampler,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.hparams.collate_fn,
            sampler=self.hparams.sampler,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from omegaconf import DictConfig, OmegaConf

    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs")
    output_path = path / "outputs"
    print(f"config_path: {config_path}")
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg.data, resolve=True))

        ocr = hydra.utils.instantiate(cfg.data)
        ocr.setup()

        loader = ocr.test_dataloader()
        img, tgt_input, tgt_key_padding_mask = next(iter(loader))
        print(img)

    main()
