import itertools


class DomNode:
    def __init__(self, *children, **kwargs):
        super().__init__()
        self.children = children
        self.attrs = kwargs

    def apply_visitor(self, v):
        return v.apply(self)


class Renderer:
    """
    >>> print(Renderer()(DomNode(DomNode(DomNode(),
    ...                    DomNode(attr1=1,attr2=2)
    ...                    ),
    ...            DomNode(attr1=1,attr2=2),
    ...            DomNode(
    ...                    DomNode(attr2=2)
    ...                    ),
    ...            root=True
    ...            )
    ...    )
    ...    )
<DomNode root={True}>
<DomNode >
  <DomNode />
  <DomNode attr1={1}, attr2={2}/>
</DomNode>
  <DomNode attr1={1}, attr2={2}/>
  <DomNode >
    <DomNode attr2={2}/>
  </DomNode>
</DomNode>

    """

    def __init__(self, indent=2):
        self.indent = indent

    def __call__(self, root):
        self._canvas = ''
        self.counter = itertools.count()
        self.count = 0
        self._space = ' ' * self.indent
        root.apply_visitor(self)
        return self._canvas

    def apply(self, node):
        count = self.count
        attrs = ", ".join(f'{k}={{{v}}}' for k, v in node.attrs.items())
        if len(node.children) > 0:

            self._canvas = self._canvas + '\n' + self._space * count + f'<{node.__class__.__name__} {attrs}>'
            self.count = next(self.counter)
            for child in node.children:
                child.apply_visitor(self)
            self._canvas = self._canvas + '\n' + self._space * count + f'</{node.__class__.__name__}>'
        else:

            self._canvas = self._canvas + '\n' + self._space * count + f'<{node.__class__.__name__} {attrs}/>'
