import sys
from pathlib import Path

sys.path.append("./yolov5")

from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import check_dataset

def create_data_source():
    """
    Creates COCO 2017 validation data loader. The method downloads COCO 2017
    dataset if it does not exist.
    """
    if not Path("datasets/coco128").exists():
        urls = ["https://ultralytics.com/assets/coco128.zip"]
        download(urls, dir="datasets")

    data = check_dataset(DATASET_CONFIG)
    val_dataloader = create_dataloader(
        data["val"], imgsz=640, batch_size=1, stride=32, pad=0.5, workers=1
    )[0]

    return val_dataloader


data_source = create_data_source()
from openvino.tools.pot.api import DataLoader

class YOLOv5POTDataLoader(DataLoader):
    """Inherit from DataLoader function and implement for YOLOv5."""

    def __init__(self, data_source):
        super().__init__({})
        self._data_loader = data_source
        self._data_iter = iter(self._data_loader)

    def __len__(self):
        return len(self._data_loader.dataset)

    def __getitem__(self, item):
        try:
            batch_data = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._data_loader)
            batch_data = next(self._data_iter)

        im, target, path, shape = batch_data

        im = im.float()
        im /= 255
        nb, _, height, width = im.shape
        img = im.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        annotation = dict()
        annotation["image_path"] = path
        annotation["target"] = target
        annotation["batch_size"] = nb
        annotation["shape"] = shape
        annotation["width"] = width
        annotation["height"] = height
        annotation["img"] = img

        return (item, annotation), img

pot_data_loader = YOLOv5POTDataLoader(data_source)
