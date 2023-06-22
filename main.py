from modules.poset import StringPoset, PosetCombination


def main():
    # sp2 = StringPoset.from_string('(1,2,3,4,5)')  --  такого посета нет!! исправить ифы

    # sp1 = StringPoset.from_string('[1,2](3,4)')
    # sp2 = StringPoset.from_string('(1,2)[3,4]')
    # 0

    # sp1 = StringPoset.from_string('[1,2](3,4,5)')
    # sp2 = StringPoset.from_string('(1,2)[3,4](5)')
    # 0

    # sp1 = StringPoset.from_string('[1,2](3,4,5,6)')
    # sp2 = StringPoset.from_string('(1,2)[3,4](5,6)')
    # 0

    # sp1 = StringPoset.from_string('[1,3](2,4,5,6)')
    # sp2 = StringPoset.from_string('(1,2,3)[4,5](6)')
    # [1,2](3)[4,5](6)

    # sp1 = StringPoset.from_string('(4)[1,3](2,5,6,7)')
    # sp2 = StringPoset.from_string('(1,2,3,4)[5,6](7)')
    # (4)[1,2](3)[5,6](7)−[1,2](3,4)[5,6](7)−[1,3](2,4)[5,6](7)

    # sp1 = StringPoset.from_string('(6)[1,2](3,4,5)')
    # sp2 = StringPoset.from_string('(1,2,6)[3,4](5)')
    # -[1,2](6)[3,4](5)

    # sp1 = StringPoset.from_string('(1,6)[2,3](4,5,7)')
    # sp2 = StringPoset.from_string('(1,6,2,3)[4,5](7)')
    # (6)[1,2](3)[4,5](7) + (1)[2,3](6)[4,5](7) + [1,2](3,6)[4,5](7)

    # sp1 = StringPoset.from_string('(7,8,6)[1,2](3,4,5)')
    # sp2 = StringPoset.from_string('(5,6,7,8)[3,4](1,2)')
    # 0

    # sp1 = StringPoset.from_string('(1,2,3)[4,5](6)')
    # sp2 = StringPoset.from_string('[1,2](3,4,5,6)')
    # -'[1,2](3)[4,5](6)'

    # print(sp1*sp2)

    # sps = list(map(StringPoset.from_string, ['(1)[2](3)','(1)[3](2)','(3)[1](2)','(3)[2](1)','(2)[3](1)','(2)[1](3)']))
    # print([PosetCombination.boundary(sp) for sp in sps])
    # # [[1,2](3)+(1)[2,3], [1,3](2)-(1)[2,3], -[1,3](2)+(3)[1,2], -[2,3](1)-(3)[1,2], [2,3](1)-(2)[1,3], -[1,2](3)+(2)[1,3]]

    # sp1 = StringPoset.from_string('(1)[2,3](4,5,6)')
    # sp2 = StringPoset.from_string('(2)[1,3](4,5,6)')
    # sp3 = StringPoset.from_string('(1,2,3)[4,5](6)')
    # print((sp1 + sp2)*sp3)
    # 0

    # n = 6
    #
    # tidy_posets = StringPoset.generate_tidy(n)
    # betti_1 = len(tidy_posets)
    #
    # circles = []
    # nontriv = []
    # torus_relations = []
    # long_relations = []
    # products = {}
    #
    # for i in range(betti_1):
    #     products[(i, i)] = PosetCombination({})
    #     for j in range(i + 1, betti_1):
    #
    #         left = tidy_posets[i]
    #         right = tidy_posets[j]
    #         product = tidy_posets[i] * tidy_posets[j]  # tidy
    #
    #         if isinstance(product, StringPoset):
    #             if repr(product) == '0' or product.is_ch_zero:
    #                 products[(i, j)] = PosetCombination({})
    #                 products[(j, i)] = PosetCombination({})
    #                 continue
    #
    #             product = PosetCombination({product: 1})
    #
    #         if repr(product) != '0' and repr(product) != '-0':
    #             nontriv = set(list(nontriv) + [i, j])
    #             products[(i, j)] = product
    #             products[(j, i)] = (-1) * product
    #
    #         else:
    #             products[(i, j)] = PosetCombination({})
    #             products[(j, i)] = PosetCombination({})
    #
    # for i in range(betti_1):
    #     if all(products[(i, j)] == PosetCombination({}) for j in range(betti_1)):
    #         circles.append(tidy_posets[i])
    #
    # print(betti_1)
    # print(len(circles))
    # new_circles = []
    # nontriv = list(nontriv)
    #
    # for i in nontriv:
    #     for j in nontriv[nontriv.index(i) + 1:]:
    #         if all(products[i, k] + products[k, j] == PosetCombination({}) for k in range(betti_1)):
    #             new_circles.append(tidy_posets[i] + (-1) * tidy_posets[j])
    #         if all(products[i, k] + products[j, k] == PosetCombination({}) for k in range(betti_1)):
    #             new_circles.append(tidy_posets[i] + tidy_posets[j])
    # i = 0
    # print('kk')
    # new_circles_remove = []
    # while i != len(new_circles):
    #     j = 0
    #     while j != len(new_circles):
    #         if i == j:
    #             j += 1
    #             continue
    #         if new_circles[i] + (-1) * new_circles[j] in new_circles:
    #             new_circles_remove.append(new_circles[i] + (-1) * new_circles[j])
    #         if new_circles[j] + (-1) * new_circles[i] in new_circles:
    #             new_circles_remove.append(new_circles.remove(new_circles[j] + (-1) * new_circles[i]))
    #         if (-1) * new_circles[i] + (-1) * new_circles[j] in new_circles:
    #             new_circles_remove.append(new_circles.remove((-1) * new_circles[i] + (-1) * new_circles[j]))
    #         if new_circles[i] + new_circles[j] in new_circles:
    #             new_circles_remove.append(new_circles.remove(new_circles[i] + new_circles[j]))
    #         j += 1
    #     i += 1
    # new_circles = [x for x in new_circles if not x in new_circles_remove]
    # print(len(new_circles))
    # # print(circles) #(1,2)[3,4](5,6)
    # # print(new_circles) #(1)[2,3](4,5,6)'+'(2)[1,3](4,5,6)


if __name__ == '__main__':
    main()
