from mmcore.base.params import BufferNode, Node
from mmcore.geom.parametric import Linear
from mmcore.geom.point import GeometryBuffer

buffer = GeometryBuffer(uuid="paramline_buffer")

paramline = Node(uuid="paramline",
                 t=0.5,
                 line=dict(
                     start=BufferNode(value=buffer.append([1, 2, 3]),
                                      buffer=buffer),
                     end=BufferNode(value=buffer.append([3, 2, 5]),
                                    buffer=buffer)
                 )
                 )


@paramline.line.bind
class Line:
    def __init__(self, start=None, end=None):
        print(start, end)
        self._line = Linear.from_two_points(start, end)

    def __call__(self, t):
        return self._line.evaluate(t)


@paramline.bind
def evaluate(t=0.5, line=None):
    return line(t)


class Asset:
    def __new__(cls, *args, **kwargs):
        copied = paramline.copy_with_graph()
        return copied(*args, **kwargs)


if __name__ == "__main__":
    paramline.dump("mmcore/base/params/asset_example/asset.pkl")
