from .dota import DOTADataset
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class RailDataset(DOTADataset):
    CLASSES = ('rail', )
    PALETTE = [(0, 255, 0)]
