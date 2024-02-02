from mmcore.common.dom.node import DomNode
from mmcore.common.dom.visitor import AbstractDOMVisitor

from io import BytesIO
import itertools


class ToyReact(AbstractDOMVisitor):
    """
    Visitor implementation for DOMNode on a funny example of an ToyReact object that accepts a DOM and returns
    jsx-like code
    preserving the DOM structure.
    ---
    Реализация посетителя для DOMNode на забавном примере класса ToyReact, объект которого, принимает DOM и
    возвращает код похожий на jsx, отражающий структуру DOM.

    >>> from mmcore.common.dom.node import DomNode

    >>> model=DomNode(DomNode(DomNode(),
    ...                    DomNode(attr1=1,attr2=2)
    ...                    ),
    ...            DomNode(attr1=1,attr2=2),
    ...            DomNode(
    ...                    DomNode(attr2=2)
    ...                    ),
    ...            root=True
    ...            )
    ...         )
    ...

    >>> from mmcore.common.dom.visitor_example import ToyReact

    >>> with ToyReact() as render:
    ...     buf=render(model)
    ...     print(buf.decode())

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


    >>> with ToyReact(indent=4) as render:
    ...  buf=render(model)
    ...  print(buf.decode())

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
        self._buffer = None
        self.counter = None
        self.count = None
        self._space = None
        self._size = 0

    def prepare_render(self):
        self.counter = itertools.count()
        self.count = 0
        self._size = 0
        self._space = ' ' * self.indent
        self._buffer = None

    def __enter__(self):
        self.prepare_render()
        return self.render

    def render(self, node: DomNode):
        with BytesIO() as self._buffer:
            node.apply_visitor(self)
            return self._buffer.getvalue()

    def write(self, data: 'str|bytes'):
        bts = data.encode() if isinstance(data, str) else data
        self._size += len(bts)
        self._buffer.write(bts)

    def apply(self, node):
        count = self.count
        attrs = ", ".join(f'{k}={{{v}}}' for k, v in node.attrs.items())
        if len(node.children) > 0:
            self.write('\n' + self._space * count + f'<{node.__class__.__name__} {attrs}>')
            self.count = next(self.counter)
            for child in node.children:
                child.apply_visitor(self)

            self.write('\n' + self._space * count + f'</{node.__class__.__name__}>')

        else:

            self.write('\n' + self._space * count + f'<{node.__class__.__name__} '
                                                    f'{attrs}/>'
                       )
