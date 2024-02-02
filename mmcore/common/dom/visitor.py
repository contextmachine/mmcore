import abc
from mmcore.common.dom.node import DomNode


class AbstractDOMVisitor:
    """
    Example of DOM Visitor based class:

    """

    @abc.abstractmethod
    def apply(self, node) -> None: ...

    def __enter__(self):
        return self.render

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_type(exc_val)

    def render(self, node: DomNode):
        node.apply_visitor(self)
        return self
