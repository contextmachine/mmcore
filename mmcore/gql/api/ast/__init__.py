import graphql
import operator as op

from mmcore.collections import ElementSequence


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


class ExtendedSelectionSetNode(graphql.language.ast.SelectionSetNode):

    def generic_getter(self):

        def wrap(x):
            l = dict()
            for field in self.selections:
                l[field.name.value] = field.generic_getter()(x)
            return l

        return wrap

    def convert(self):

        l = []
        for v in self.selections:
            node = convert(v, ExtendedFieldNode)
            l.append(node)
        self.selections = tuple(l)


class ExtendedFieldNode(graphql.language.ast.FieldNode):

    def generic_getter(self):

        if self.selection_set is None:

            return lambda x:self.name.generic_getter()(x)


        else:
            def wrap(x):
                xx=self.name.generic_getter()(x)
                if isinstance(xx,list):
                    l=[]
                    for xxx in xx:
                       l.append(self.selection_set.generic_getter()(xxx))

                    return l

                else:
                    return self.selection_set.generic_getter()(xx)

            return wrap

    def convert(self):

        self.name = convert(self.name, ExtendedNameNode)
        if self.selection_set is not None:
            self.selection_set = convert(self.selection_set, ExtendedSelectionSetNode)
