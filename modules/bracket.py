from functools import cached_property

from .const import BracketType


class Bracket:
    def __init__(self, bracket_type: BracketType, elements: list):
        self.bracket_type = bracket_type
        self.elements = sorted(elements)

    @cached_property
    def is_empty(self):
        return not self.elements

    @cached_property
    def max(self) -> int:
        return max(self.elements)

    @cached_property
    def size(self):
        return len(self.elements)

    def __repr__(self):
        res = []
        if self.bracket_type == BracketType.SQUARE:
            start, end = '[', ']'
        else:
            start, end = '(', ')'
        res.append(f'{start}{",".join(map(str, self.elements))}{end}')
        return ''.join(res)
