import operator as op

import graphql


def convert(self, target):
    node = target()
    for key, attr in zip(self.keys, op.attrgetter(*self.keys)(self)):
        setattr(node, key, attr)
    node.convert()

    return node


class ExtendedNameNode(graphql.language.ast.NameNode):

    def generic_getter(self):
        def wrap(x):
            return x.get(self.value)

        return wrap

    def convert(self):
        ...


NO_SCALE = True
class AExtendedNameNode(ExtendedNameNode):

    def generic_getter(self):
        def wrap(x):
            # print(x)
            try:
                res = x.get(self.value)
                res()
                #print(self.value)
                if hasattr(res, "_repr3d"):
                    return res._repr3d.root()
                elif hasattr(res, 'root'):

                    return res.root()
                else:
                    return res
                # if NO_SCALE:
                #    if hasattr(res, "_repr3d"):
                #        return res._repr3d.root()
                #    elif hasattr(res, 'root'):
                #
                #        return res.root()
                #    else:
                #        return res
                # else:
                #    if hasattr(res, "_repr3d"):
                #
                #        res.scale(0.001,0.001,0.001)
                #        res.rotate()
                #        return res._repr3d.root()
                #    if hasattr(res, "root"):
                #        res._matrix = [0.001, 0, 0, 0, 0, 2.220446049250313e-19, 0.001, 0, 0, 0.001,
                #                          2.220446049250313e-19, 0, 0, 0, 0, 1]
                #        return res.root()
                #    else:
                #        return res
            except:
                return x.get(self.value)

        return wrap


class ExtendedSelectionSetNode(graphql.language.ast.SelectionSetNode):

    def generic_getter(self):
        def wrap(x):
            dct = dict()
            for field in self.selections:
                dct[field.name.value] = field.generic_getter()(x)
            return dct

        return wrap

    def convert(self):
        lst = []
        for v in self.selections:
            node = convert(v, ExtendedFieldNode)
            lst.append(node)
        self.selections = tuple(lst)


class ExtendedFieldNode(graphql.language.ast.FieldNode):

    def generic_getter(self):
        if self.selection_set is None:
            return lambda x: self.name.generic_getter()(x)
        else:
            def wrap(x):
                xx = self.name.generic_getter()(x)
                if isinstance(xx, list):
                    lst = []
                    for xxx in xx:
                        lst.append(self.selection_set.generic_getter()(xxx))
                    return lst
                else:
                    return self.selection_set.generic_getter()(xx)

            return wrap

    def convert(self):
        self.name = convert(self.name, AExtendedNameNode)
        if self.selection_set is not None:
            self.selection_set = convert(self.selection_set, ExtendedSelectionSetNode)
