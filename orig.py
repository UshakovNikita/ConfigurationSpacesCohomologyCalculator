import itertools
import string
from enum import Enum
from sympy.combinatorics.permutations import Permutation
import numpy as np

K = 3


class BracketType(Enum):
    SQUARE = 'square'
    ROUND = 'round'


class PosetValidationError(Exception):
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return f'Poset is not valid:  {self.error}'


class Bracket:
    def __init__(self, bracket_type: BracketType, elements: list):
        self.bracket_type = bracket_type
        self.elements = sorted(elements)

    @property
    def is_empty(self):
        return not self.elements

    @property
    def max(self) -> int:
        return max(self.elements)

    def __len__(self):
        return len(self.elements)


class StringPoset:
    def __init__(self, bracket_list):
        self.bracket_list = [br for br in bracket_list if not br.is_empty]

    def __repr__(self):
        if self.is_ch_zero:
            return '0'
        rep_str = ""
        for br in self.bracket_list:
            elem_rep = list(map(str, br.elements))
            if br.bracket_type == BracketType.SQUARE:
                rep_str += '[' + ','.join(elem_rep) + ']'
            else:
                rep_str += '(' + ','.join(elem_rep) + ')'
        return rep_str

    def __len__(self):
        return len(self.order)

    @property
    def degree(self) -> int:
        deg = 0
        for br in self.bracket_list:
            if br.bracket_type == BracketType.SQUARE:
                deg += len(br.elements) - 1
        return deg

    @property
    def orient(self) -> int:
        return Permutation(self.order).signature()

    @property
    def order(self) -> list:
        order = []
        for br in self.bracket_list:
            order += br.elements
        order = [x - 1 for x in order]
        return order

    @property
    def is_ch_zero(self) -> bool:
        if all(len(br) < K for br in self.bracket_list if br.bracket_type == BracketType.SQUARE):
            return False
        else:
            return True

    def tidiness_check(self) -> bool:
        if len(self.bracket_list) == 1:
            if self.bracket_list[0].bracket_type == BracketType.SQUARE:
                return False
            else:
                return True
        if self.bracket_list[0].bracket_type == BracketType.SQUARE:
            i = 0
        else:
            i = 1  # skip first round bracket
        while i < len(self.bracket_list) - 1:
            left = self.bracket_list[i]
            right = self.bracket_list[i + 1]
            if left.bracket_type == BracketType.SQUARE and right.bracket_type == BracketType.ROUND:
                if left.max < right.max:
                    i += 2
                    continue
                else:
                    return False
            else:
                return False
        if i == len(self.bracket_list) - 1 and self.bracket_list[i].bracket_type == BracketType.SQUARE:
            return False

        return True

def print_coef(coef: int, first: bool) -> string:
    if coef == 1:
        if first: return ''
        else: return '+'
    elif coef == -1:
        return '-'
    elif first:
        return str(coef)+'*'
    elif coef > 0:
        return '+' + str(coef) + '*'
    else:
        return str(coef) + '*'


class PosetCombination:
    def __init__(self, poset_dict: dict):
        self.poset_dict = {poset: coef for (poset, coef) in poset_dict.items() if
                           not coef == 0 and not poset.is_ch_zero}

    def __repr__(self):
        rep_str = ""
        posets = list(self.poset_dict.keys())
        if len(posets) == 0:
            return '0'
        poset = posets[0]
        rep_str += print_coef(self.poset_dict.get(poset), True) + StringPoset.__repr__(poset)
        if len(posets) == 1:
            return rep_str
        else:
            for poset in posets[1:]:
                rep_str += print_coef(self.poset_dict.get(poset), False) + StringPoset.__repr__(poset)
        return rep_str


def string_to_poset(s: string) -> StringPoset:
    s = s.replace(" ", "")
    br_list = []
    while not len(s) == 0:
        if s[0] == '[':
            type = BracketType.SQUARE
            end = s.find(']')
        else:
            if not len(br_list) == 0:
                if br_list[len(br_list) - 1].bracket_type == BracketType.ROUND:
                    raise PosetValidationError('poset is not separated')
            type = BracketType.ROUND
            end = s.find(')')
        elements = s[1:end]
        if not elements:
            raise PosetValidationError('empty bracket')
        elements = elements.split(',')
        elements = [int(x) for x in elements if x.isdigit()]

        # if type == BracketType.SQUARE and len(elements) == 1:
        #     raise PosetValidationError('square bracket with a single element')
        br = Bracket(type, elements)
        br_list.append(br)
        s = s[end + 1:]
    return StringPoset(br_list)


def permutation_matrix(ord: list):
    matrix = np.zeros((len(ord), len(ord)), dtype=int)
    for i in range(len(ord)):
        matrix[ord[i], i] = 1
    return matrix


def poset_to_matrix(sp: StringPoset):
    sp_len = len(sp)
    matrix = np.identity(sp_len, dtype=int)
    curr = 0

    for br in sp.bracket_list:
        br_len = len(br.elements)

        if br.bracket_type == BracketType.SQUARE:  # equalise elements of a single square bracket
            ones = (br_len, br_len)
            ones = np.ones(ones, dtype=int)
            matrix[curr: curr + br_len, curr: curr + br_len] = ones

        ones = (br_len, sp_len - br_len - curr)
        ones = np.ones(ones, dtype=int)
        matrix[curr: curr + br_len, curr + br_len:sp_len] = ones
        curr += br_len
    perm = permutation_matrix(sp.order)
    matrix = perm.dot(matrix).dot(perm.transpose())
    return matrix


def matrix_to_poset(matrix: np.array) -> StringPoset:
    sums = {}
    for row in range(len(matrix)):
        sum = matrix[row].sum()  # number of '1' in a given row
        if sum in sums:
            sums[sum].append(row)
        else:
            sums[sum] = [row]
    sums = sums.items()
    sorted_sums = dict(sorted(sums, reverse=True))
    br_list = []
    for sum in sorted_sums.keys():
        elements = sorted_sums[sum]
        if len(elements) == 1:
            elements = [el + 1 for el in elements]
            br = Bracket(BracketType.ROUND, elements)
        else:
            rows = [matrix[el] for el in elements]
            control = rows[0]
            if all(np.array_equal(row, control) for row in rows):
                type = BracketType.SQUARE
            else:
                type = BracketType.ROUND
            elements = [el + 1 for el in elements]
            br = Bracket(type, elements)
        br_list.append(br)
    return StringPoset(br_list)


def is_pre_elementary(sp: StringPoset) -> bool:
    # if not sp.degree == 0:
    #     return False
    list = sp.bracket_list
    if (len(list) == 3 and len(list[1].elements) == K - 2) or (
            len(list) == 2 and (len(list[0].elements) == K - 2 or len(list[1].elements) == K - 2)):
        return True
    else:
        return False


def get_boundary(sp: StringPoset) -> PosetCombination:
    if not is_pre_elementary(sp):
        raise PosetValidationError('poset is not pre-elementary')
    boundary_dict = {}
    if len(sp.bracket_list) == 3:
        A = sp.bracket_list[0].elements
        i = sp.bracket_list[1].elements[0]
        B = sp.bracket_list[2].elements
        for a in A:
            bar_A = Bracket(BracketType.ROUND, [x for x in A if x != a])
            bnd = StringPoset([bar_A, Bracket(BracketType.SQUARE, [i, a]), Bracket(BracketType.ROUND, B)])
            boundary_dict[bnd] = bnd.orient
        for b in B:
            bar_B = Bracket(BracketType.ROUND, [x for x in B if x != b])
            bnd = StringPoset([Bracket(BracketType.ROUND, A), Bracket(BracketType.SQUARE, [i, b]), bar_B])
            boundary_dict[bnd] = bnd.orient
    if len(sp.bracket_list) == 2:
        if len(sp.bracket_list[0].elements) == 1:
            i = sp.bracket_list[0].elements[0]
            B = sp.bracket_list[1].elements
            for b in B:
                bar_B = Bracket(BracketType.ROUND, [x for x in B if x != b])
                bnd = StringPoset([Bracket(BracketType.SQUARE, [i, b]), bar_B])
                boundary_dict[bnd] = bnd.orient
        elif len(sp.bracket_list[1].elements) == 1:
            i = sp.bracket_list[1].elements[0]
            A = sp.bracket_list[0].elements
            for a in A:
                bar_A = Bracket(BracketType.ROUND, [x for x in A if x != a])
                bnd = StringPoset([bar_A, Bracket(BracketType.SQUARE, [i, a])])
                boundary_dict[bnd] = bnd.orient
    return PosetCombination(boundary_dict)


def transtive_closure(sp1, sp2: StringPoset) -> StringPoset:
    m1 = poset_to_matrix(sp1)
    m2 = poset_to_matrix(sp2)
    m = np.array(m1 + m2, dtype=bool)
    m_prev = m
    m = m.dot(m)
    while not np.array_equal(m, m_prev):
        m_prev = m
        m = m.dot(m)
    return matrix_to_poset(np.array(m, dtype=int))


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r)
        for r in range(len(s) + 1)
    )


class OutputType(Enum):
    SHORT = 'short'
    FULL = 'full'


def print_tidy(n: int, output_type: OutputType):
    counter = 0
    elements = [x for x in range(1, n + 1)]
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
                    counter += 1
                    if output_type == OutputType.SHORT:
                        continue
                    if A:
                        res = f'({",".join(map(str, A))})[{i},{j}]({",".join(map(str, B))})'
                    else:
                        res = f'[{i},{j}]({",".join(map(str, B))})'
                    if output_type == OutputType.FULL:
                        print(counter, ':', res)
    if output_type == OutputType.SHORT:
        print(counter)


def main():
    # # sps = ['(1,2)', '[1,2]', '(1,2)[3,4]', '[1,2][3,4]', '[1,2](3,4)', '(1)[34](2)', '(6)[12](3)[45]', '(5,6)[2,3](1,4)']
    # # # sps = ['[1,2]']
    # # sps = list(map(string_to_poset, sps))
    # # for sp in sps:
    # #     print(sp)
    # #     matrix = poset_to_matrix(sp)
    # #     print(matrix)
    # #     print(matrix_to_poset(matrix))
    # sp1 = string_to_poset('[1,2](3)')
    # sp2 = string_to_poset('(1)[2,3]')
    # print(__mul__(sp1, sp2))
    # # print(__mul__(sp1, sp2).is_ch_zero)
    # # print(string_to_poset('[1,2](3)[4,5,6)').is_ch_zero)
    # s = '(1)[2](3,4)'
    # sp = string_to_poset(s)
    # print(get_boundary(sp))
    print_tidy(6, OutputType.SHORT)



if __name__ == '__main__':
    main()
