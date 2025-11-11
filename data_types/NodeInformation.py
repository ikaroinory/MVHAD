from typing import Literal, TypedDict


class NodeInformation(TypedDict):
    value_type: Literal['float', 'enum']
    index: list[int]
