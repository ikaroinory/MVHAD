from typing import Literal, TypedDict


class NodeConfig(TypedDict):
    value_type: Literal['float', 'enum']
    index: list[int]
