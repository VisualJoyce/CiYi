from typing import Union, Dict

import torch
from allennlp.data.fields.field import Field


class FloatField(Field[torch.Tensor]):

    __slots__ = ["value"]

    def __init__(
            self,
            value: Union[float, int],
    ) -> None:
        self.value = value

    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        tensor = torch.tensor(self.value, dtype=torch.float)
        return tensor

    def __len__(self):
        return 1
