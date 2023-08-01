from mmcore.base.sharedstate import serve
from mmcore.geom.point import GeometryBuffer

buff = GeometryBuffer()
buff.add_items([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
serve.start()
# if __name__ == "__main__":

#    serve.start_embed_ipython()
