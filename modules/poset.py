from collections import deque
from functools import cached_property
from itertools import chain, combinations
from typing import Dict, List, Tuple

import numpy as np
from sympy import linear_eq_to_matrix
from sympy.combinatorics import Permutation

from .bracket import Bracket
from .const import BracketType, K
from .utils import PosetValidationError, permutation_matrix, all_ones_matrix


class StringPoset:
    def __init__(self, bracket_list: List[Bracket]):
        self.bracket_list = [br for br in bracket_list if not br.is_empty]

    def __repr__(self):
        if self.is_ch_zero or not self.bracket_list:
            return '0'
        res = []
        for br in self.bracket_list:
            if br.bracket_type == BracketType.SQUARE:
                start, end = '[', ']'
            else:
                start, end = '(', ')'
            res.append(f'{start}{",".join(map(str, br.elements))}{end}')
        return ''.join(res)

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

    def __len__(self) -> int:
        return len(self.sequence)

    @cached_property
    def brackets_len(self):
        return len(self.bracket_list)

    @cached_property
    def sequence(self) -> list:
        return [
            elem - 1
            for elem in chain(*[br.elements for br in self.bracket_list])
        ]

    @cached_property
    def orient(self) -> int:
        return Permutation(self.sequence).signature()

    @cached_property
    def signature(self) -> List[int]:
        if not self.degree:
            return [len(self)]
        res = []
        prev_sum = 0
        for br in self.bracket_list:
            if br.bracket_type == BracketType.SQUARE:
                res.append(prev_sum)
            prev_sum += br.size
        return res

    @cached_property
    def expansion(self):  # супер важная функция, которая должна быстро работать
        if self == StringPoset([]) or self.degree == 0:
            return self

        expansion = []
        prev = []
        br_list = self.bracket_list
        for n, br in enumerate(br_list):
            if br.bracket_type == BracketType.SQUARE:
                expansion.append(StringPoset([
                    Bracket(BracketType.ROUND, prev),
                    br,
                    Bracket(BracketType.ROUND, [
                        elem for elem in chain(*[
                            br.elements for br in br_list[n + 1:self.brackets_len]
                        ])
                    ])
                ]))
            prev += br.elements
        return expansion

    # def __lt__(self, other) -> bool:
    #     if self.degree == other.degree:
    #         return self.signature < other.signature
    #     return self.degree < other.degree

    def __lt__(self, other) -> bool:
        if self.degree == other.degree:
            if self.signature < other.signature:
                return True
            if self.signature > other.signature:
                return False
            if self.degree == 1:
                if self.brackets_len == 2:
                    if self.bracket_list[0].bracket_type == BracketType.SQUARE:
                        B1 = self.bracket_list[0].elements
                        B2 = other.bracket_list[0].elements
                    else:
                        B1 = self.bracket_list[1].elements
                        B2 = other.bracket_list[1].elements
                    return B1 < B2
                if self.brackets_len == 3:
                    A1 = self.bracket_list[0].elements
                    A2 = other.bracket_list[0].elements
                    if A1 < A2:
                        return True
                    if A1 > A2:
                        return False
                    B1 = self.bracket_list[1].elements
                    B2 = other.bracket_list[1].elements
                    if B1 < B2:
                        return True
                    if B1 > B2:
                        return False
                    else:
                        return True
            return all(self.expansion[i] < other.expansion[i] for i in range(self.degree))
        return self.degree < other.degree

    @cached_property
    def degree(self) -> int:
        return sum(
            br.size - 1
            for br in self.bracket_list
            if br.bracket_type == BracketType.SQUARE
        )

    @cached_property
    def is_ch_zero(self) -> bool:
        return any(
            br.size >= K
            for br in self.bracket_list
            if br.bracket_type == BracketType.SQUARE
        )

    @cached_property
    def is_tidy(self) -> bool:
        return self.tidiness_index == (0, 0)

    def _get_cur_index(self, cur, first):
        return (cur - first) // 2 + 1, sum(br.size for br in self.bracket_list[0:cur])

    @cached_property
    def tidiness_index(self) -> Tuple[int, int]:
        if not self.bracket_list:
            return 0, 0

        first_bracket = self.bracket_list[0]
        first = 0 if first_bracket.bracket_type == BracketType.SQUARE else 1
        if self.brackets_len == 1:
            return (
                (0, 0)
                if first_bracket.bracket_type == BracketType.ROUND
                else (1, 0)
            )

        i = 0
        index = 0, 0
        if first == 1:
            i += 1  # skip first round bracket

        for i in range(self.brackets_len - 1):
            left, right = self.bracket_list[i], self.bracket_list[i + 1]
            if (
                    left.bracket_type == BracketType.ROUND and
                    right.bracket_type == BracketType.SQUARE
            ):
                continue
            if (
                    left.bracket_type == BracketType.SQUARE and
                    right.bracket_type == BracketType.SQUARE
            ):
                index = self._get_cur_index(i, first)
                continue
            if left.max > right.max:
                index = self._get_cur_index(i, first)

        if right.bracket_type == BracketType.SQUARE:
            index = self._get_cur_index(i, first)
        return index

    def tidy(self):
        dirty_poset_index = self.tidiness_index[0]
        self_expansion = self.expansion
        dirty_poset: StringPoset = self_expansion[dirty_poset_index - 1]
        if dirty_poset.brackets_len == 3:
            square_br = dirty_poset.bracket_list[1]
            A = dirty_poset.bracket_list[0]
            B = Bracket(BracketType.ROUND, dirty_poset.bracket_list[2].elements + [square_br.max])
            pre_elem = StringPoset([A, Bracket(BracketType.SQUARE, [square_br.elements[0]]), B])
        if dirty_poset.brackets_len == 2:
            if dirty_poset.bracket_list[0].bracket_type == BracketType.SQUARE:
                square_br = dirty_poset.bracket_list[0]
                B = Bracket(BracketType.ROUND, dirty_poset.bracket_list[1].elements + [square_br.max])
                pre_elem = StringPoset([Bracket(BracketType.SQUARE, [square_br.elements[0]]), B])
            if dirty_poset.bracket_list[1].bracket_type == BracketType.SQUARE:
                square_br = dirty_poset.bracket_list[1]
                A = dirty_poset.bracket_list[0]
                pre_elem = StringPoset([A, Bracket(BracketType.SQUARE, [square_br.elements[0]]),
                                        Bracket(BracketType.ROUND, [square_br.max])])

        relation = PosetCombination.boundary(pre_elem)
        if relation.poset_dict.get(repr(dirty_poset)) == 1:
            relation *= -1
        relation.poset_dict.pop(repr(dirty_poset))
        res = sum(relation * sp for sp in self_expansion if sp != dirty_poset)
        return res

    @classmethod
    def from_string(cls, string_repr: str):
        string_repr = string_repr.replace(' ', '')
        poset = deque(string_repr)
        brackets = []
        while poset:
            start = poset.popleft()
            if start == '[':
                bracket_type = BracketType.SQUARE
                end = ']'
            else:
                bracket_type = BracketType.ROUND
                end = ')'
                if brackets and brackets[-1].bracket_type == BracketType.ROUND:
                    raise PosetValidationError('poset is not separated')

            # TODO not good but better than dealing with strings
            elem = poset.popleft()
            elements, cur_elems = [], []
            while elem != end:
                if elem == ',':
                    elements.append(''.join(cur_elems))
                    cur_elems = []
                else:
                    cur_elems.append(elem)
                if poset:
                    elem = poset.popleft()
            elements.append(''.join(cur_elems))

            if not elements:
                raise PosetValidationError('empty bracket')

            elements = [int(x) for x in elements if x.isdigit()]

            # if type == BracketType.SQUARE and len(elements) == 1:
            #     raise PosetValidationError('square bracket with a single element')

            bracket = Bracket(bracket_type, elements)
            brackets.append(bracket)
        return cls(brackets)

    def to_matrix(self):
        sp_len = len(self)
        matrix = np.identity(sp_len, dtype=int)
        curr = 0
        for bracket in self.bracket_list:
            # equalise elements of a single square bracket
            if bracket.bracket_type == BracketType.SQUARE:
                matrix[
                curr:curr + bracket.size,
                curr:curr + bracket.size
                ] = all_ones_matrix(bracket.size, bracket.size)

            matrix[
            curr: curr + bracket.size,
            curr + bracket.size: sp_len
            ] = all_ones_matrix(bracket.size, sp_len - bracket.size - curr)

            curr += bracket.size

        perm = permutation_matrix(self.sequence)
        matrix = perm.dot(matrix).dot(perm.transpose())
        return matrix

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        sums = {}
        for row in range(len(matrix)):
            cur_sum = matrix[row].sum()  # number of '1' in a given row
            if cur_sum in sums:
                sums[cur_sum].append(row)
            else:
                sums[cur_sum] = [row]

        sorted_sums = dict(sorted(sums.items(), reverse=True))

        brackets_list = []
        for elements in sorted_sums.values():
            if len(elements) != 1:
                rows = [matrix[el] for el in elements]
                if all(np.array_equal(row, rows[0]) for row in rows[1:]):
                    bracket_type = BracketType.SQUARE
                else:
                    bracket_type = BracketType.ROUND
            else:
                bracket_type = BracketType.ROUND

            elements = [el + 1 for el in elements]
            brackets_list.append(Bracket(bracket_type, elements))
        return cls(brackets_list)

    @cached_property
    def is_pre_elementary(self) -> bool:
        # if not sp.degree == 0:
        #     return False
        bracket_list = self.bracket_list
        if (
                self.brackets_len in (2, 3) and
                bracket_list[1].size == K - 2
        ) or (
                self.brackets_len == 2 and
                bracket_list[0].size == K - 2
        ):
            return True
        else:
            return False

    def __mul__(self, other):
        if self == StringPoset([]):
            return StringPoset([])
        if isinstance(other, int):
            return PosetCombination({self: other})
        if self == other:
            return StringPoset([])
        exp1 = self.expansion
        exp2 = other.expansion
        if any(x in exp2 for x in exp1):  # intersection
            return StringPoset([])
        m1 = self.to_matrix()
        m2 = other.to_matrix()
        m = np.array(m1 + m2, dtype=bool)
        m_prev = m
        m = m.dot(m)
        while not np.array_equal(m, m_prev):
            m_prev = m
            m = m.dot(m)
        res = StringPoset.from_matrix(np.array(m, dtype=int))
        if res.is_ch_zero:
            return StringPoset([])
        exp = exp1 + exp2
        EXP = [(exp[i], i) for i in range(len(exp))]
        EXP.sort()
        sorted_exp, perm = zip(*EXP)
        sign = Permutation(perm).signature()
        if res.is_tidy:
            return (
                PosetCombination({res: 1})
                if sign == 1
                else PosetCombination({res: -1})
            )
        return res.tidy() if sign == 1 else (-1) * res.tidy()

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, PosetCombination):
            return PosetCombination({self: 1}) + other
        if not self.bracket_list:
            return other
        if not other.bracket_list:
            return self
        if self == other:
            return PosetCombination({self: 2})
        else:
            return PosetCombination({self: 1, other: 1})

    def __radd__(self, other):
        return self + other

    def generate_tidy(n: int):
        elements = [x for x in range(1, n + 1)]
        tidy_posets = []
        for A in list(powerset(elements)):
            bar_A = list(set(elements) - set(A))
            for i in bar_A:
                for j in bar_A:
                    if i >= j:
                        continue
                    B = bar_A.copy()
                    B.remove(i)
                    B.remove(j)
                    if B:
                        if max(B) < j:
                            continue
                        if A:
                            sp = StringPoset([Bracket(BracketType.ROUND, A), Bracket(BracketType.SQUARE, [i, j]),
                                              Bracket(BracketType.ROUND, B)])
                        else:
                            sp = StringPoset([Bracket(BracketType.SQUARE, [i, j]), Bracket(BracketType.ROUND, B)])
                        tidy_posets.append(sp)
        return tidy_posets


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r)
        for r in range(len(s) + 1)
    )


def print_coef(coef: int, first: bool) -> str:
    if coef == 1:
        return '' if first else '+'
    elif coef == -1:
        return '-'
    elif coef == 0:
        return ''
    elif first:
        return f'{coef}*'
    elif coef > 0:
        return f'+{coef}*'
    else:
        return f'{coef}*'


class PosetCombination:
    def __init__(self, poset_dict: Dict[StringPoset, int]):
        self.poset_dict = {
            repr(poset): coef  # TODO do not you poset as dict key!
            for poset, coef in sorted(poset_dict.items())
            if coef != 0 and not poset.is_ch_zero and poset != StringPoset([])
        }

    def __repr__(self):
        res = []
        posets = list(self.poset_dict.keys())
        if len(posets) == 0 or all(coef == 0 for coef in list(self.poset_dict.values())):
            return '0'

        res.append(
            f'{print_coef(self.poset_dict.get(posets[0]), True)}{repr(posets[0])}'
        )
        for poset in posets[1:]:
            res.append(
                f'{print_coef(self.poset_dict.get(poset), False)}{repr(poset)}'
            )
        return ''.join(res)

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

    def __len__(self) -> int:
        return len(list(self.poset_dict.keys()))

    @classmethod
    def boundary(cls, poset: StringPoset):
        if not poset.is_pre_elementary:
            raise PosetValidationError('poset is not pre-elementary')

        boundary_dict = {}
        if poset.brackets_len == 3:
            A = poset.bracket_list[0].elements
            i = poset.bracket_list[1].elements[0]
            B = poset.bracket_list[2].elements
            for a in A:
                bar_A = Bracket(BracketType.ROUND, [x for x in A if x != a])
                bnd = StringPoset([
                    bar_A,
                    Bracket(BracketType.SQUARE, [i, a]),
                    Bracket(BracketType.ROUND, B)
                ])
                boundary_dict[bnd] = bnd.orient * poset.orient
            for b in B:
                bar_B = Bracket(BracketType.ROUND, [x for x in B if x != b])
                bnd = StringPoset([
                    Bracket(BracketType.ROUND, A),
                    Bracket(BracketType.SQUARE, [i, b]),
                    bar_B
                ])
                boundary_dict[bnd] = bnd.orient * poset.orient

        if poset.brackets_len == 2:
            if poset.bracket_list[0].size == 1:
                i = poset.bracket_list[0].elements[0]
                B = poset.bracket_list[1].elements
                for b in B:
                    bar_B = Bracket(BracketType.ROUND, [x for x in B if x != b])
                    bnd = StringPoset([
                        Bracket(BracketType.SQUARE, [i, b]),
                        bar_B
                    ])
                    boundary_dict[bnd] = bnd.orient * poset.orient
            elif poset.bracket_list[1].size == 1:
                i = poset.bracket_list[1].elements[0]
                A = poset.bracket_list[0].elements
                for a in A:
                    bar_A = Bracket(BracketType.ROUND, [x for x in A if x != a])
                    bnd = StringPoset([
                        bar_A,
                        Bracket(BracketType.SQUARE, [i, a])
                    ])
                    boundary_dict[bnd] = bnd.orient * poset.orient

        return cls(boundary_dict)

    def __add__(self, other):
        if isinstance(other, int):
            if other == 0:
                return self
            else:
                raise TypeError
        if type(other) == StringPoset:
            other = PosetCombination({other: 1})
        poset_list_1 = list(self.poset_dict.keys())
        poset_list_2 = list(other.poset_dict.keys())
        poset_list_common = list(set(poset_list_1) & set(poset_list_2))
        # poset_list_common = [x for x in poset_list_1 if x in poset_list_2]
        poset_list_1 = list(set(poset_list_1) - set(poset_list_common))
        poset_list_2 = list(set(poset_list_2) - set(poset_list_common))
        # poset_list_1 = [x for x in poset_list_1 if not x in poset_list_common]
        # poset_list_2 = [x for x in poset_list_2 if not x in poset_list_common]
        res_poset_dict = {}
        for x in poset_list_1:
            res_poset_dict[x] = self.poset_dict.get(x)
        for x in poset_list_2:
            res_poset_dict[x] = other.poset_dict.get(x)
        for x in poset_list_common:
            res_poset_dict[x] = self.poset_dict.get(x) + other.poset_dict.get(x)

        # todo create PosetCombination from {poset_repr: val} not from {poset_val}
        return PosetCombination({
            StringPoset.from_string(poset): value
            for poset, value in res_poset_dict.items()
        })

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, int) and isinstance(self, PosetCombination):
            return PosetCombination({
                # todo test it mb just poset not StringPoset.from_string(poset)
                StringPoset.from_string(poset): self.poset_dict.get(poset) * other
                for poset in list(self.poset_dict.keys())
            })
        else:
            if type(other) == StringPoset:
                other = PosetCombination({other: 1})

            res = PosetCombination({})
            for a in self.poset_dict.items():
                for b in other.poset_dict.items():
                    poset_mul = StringPoset.from_string(a[0]) * StringPoset.from_string(b[0])
                    res += poset_mul * (a[1] * b[1])
        return res

    def __rmul__(self, other):
        return self * other
