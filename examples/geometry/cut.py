import pickle

import numpy as np

from mmcore.base import AGroup
from mmcore.geom.parametric import PlaneLinear

poly_bytes = (
    b'\x80\x04\x95\xcd3\x00\x00\x00\x00\x00\x00\x8c\x17mmcore.geom.shapes.base\x94\x8c\nPolyHedron\x94\x93\x94'
    b')\x81\x94}\x94(\x8c\x05table\x94]\x94('
    b'G\xc0\x13\xcaP@\x00\x00\x00G@\x1eZk`\x00\x00\x00G@\x1f\xfa\x82@\x00\x00\x00\x87\x94G\xc0\x13\xcaP@\x00'
    b'\x00\x00G@\x1eZk`\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00\x87\x94G\xc0\x13\xcaP@\x00\x00\x00G\xc0'
    b'\x0f\xce\xc7\xc0\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00\x87\x94G\xc0\x13\xcaP@\x00\x00\x00G\xc0'
    b'\x0f\xce\xc7\xc0\x00\x00\x00G@/\xfa\x82@\x00\x00\x00\x87\x94G\xc0\x13\xcaP@\x00\x00\x00G@\x1eZk`\x00'
    b'\x00\x00G@/\xfa\x82@\x00\x00\x00\x87\x94G\xc0\x13\xcaP@\x00\x00\x00G\xc0/\x14\x99\x80\x00\x00\x00G@\x1f'
    b'\xfa\x82@\x00\x00\x00\x87\x94G\xc0\x13\xcaP@\x00\x00\x00G\xc0/\x14\x99\x80\x00\x00\x00G@/\xfa\x82@\x00'
    b'\x00\x00\x87\x94G\xc0\x13\xcaP@\x00\x00\x00G\xc0/\x14\x99\x80\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x87\x94G?\xfb\xaa;\xe0\x00\x00\x00G\xc0/\x14\x99\x80\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x87\x94G?\xfb\xaa;\xe0\x00\x00\x00G\xc0/\x14\x99\x80\x00\x00\x00G@/\xfa\x82@\x00\x00\x00\x87\x94G@ '
    b'\xcf\xb7 \x00\x00\x00G\xc0/\x14\x99\x80\x00\x00\x00G@\x1f\xfa\x82@\x00\x00\x00\x87\x94G@ \xcf\xb7 '
    b'\x00\x00\x00G\xc0/\x14\x99\x80\x00\x00\x00G@/\xfa\x82@\x00\x00\x00\x87\x94G@ \xcf\xb7 '
    b'\x00\x00\x00G\xc0/\x14\x99\x80\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00\x87\x94G@ \xcf\xb7 '
    b'\x00\x00\x00G\xc0\x0f\xce\xc7\xc0\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00\x87\x94G@ \xcf\xb7 '
    b'\x00\x00\x00G\xc0\x0f\xce\xc7\xc0\x00\x00\x00G@/\xfa\x82@\x00\x00\x00\x87\x94G@ \xcf\xb7 '
    b'\x00\x00\x00G@\x1eZk`\x00\x00\x00G@\x1f\xfa\x82@\x00\x00\x00\x87\x94G@ \xcf\xb7 '
    b'\x00\x00\x00G@\x1eZk`\x00\x00\x00G@/\xfa\x82@\x00\x00\x00\x87\x94G@ \xcf\xb7 '
    b'\x00\x00\x00G@\x1eZk`\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00\x87\x94G?\xfb\xaa;\xe0\x00\x00\x00G'
    b'@\x1eZk`\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00\x87\x94G?\xfb\xaa;\xe0\x00\x00\x00G@\x1eZk`\x00'
    b'\x00\x00G@/\xfa\x82@\x00\x00\x00\x87\x94e\x8c\x10attributes_table\x94}\x94('
    b'\x8c\r0x36329c240_0\x94}\x94\x8c\r0x36329c240_1\x94}\x94\x8c\r0x36329c240_2\x94}\x94\x8c\r0x36329c240_3'
    b'\x94}\x94\x8c\r0x36329c240_4\x94}\x94\x8c\r0x36329c240_5\x94}\x94\x8c\r0x36329c240_6\x94}\x94\x8c'
    b'\r0x36329c240_7\x94}\x94\x8c\r0x36329c240_8\x94}\x94\x8c\r0x36329c240_9\x94}\x94\x8c\x0e0x36329c240_10'
    b'\x94}\x94\x8c\x0e0x36329c240_11\x94}\x94\x8c\x0e0x36329c240_12\x94}\x94\x8c\x0e0x36329c240_13\x94}\x94'
    b'\x8c\x0e0x36329c240_14\x94}\x94\x8c\x0e0x36329c240_15\x94}\x94\x8c\x0e0x36329c240_16\x94}\x94\x8c'
    b'\x0e0x36329c240_17\x94}\x94\x8c\x0e0x36329c240_18\x94}\x94\x8c\x0e0x36329c240_19\x94}\x94\x8c'
    b'\x0e0x36329c240_20\x94}\x94\x8c\x0e0x36329c240_21\x94}\x94\x8c\x0e0x36329c240_22\x94}\x94\x8c'
    b'\x0e0x36329c240_23\x94}\x94\x8c\x0e0x36329c240_24\x94}\x94\x8c\x0e0x36329c240_25\x94}\x94\x8c'
    b'\x0e0x36329c240_26\x94}\x94\x8c\x0e0x36329c240_27\x94}\x94\x8c\x0e0x36329c240_28\x94}\x94\x8c'
    b'\x0e0x36329c240_29\x94}\x94\x8c\x0e0x36329c240_30\x94}\x94\x8c\x0e0x36329c240_31\x94}\x94\x8c'
    b'\x0e0x36329c240_32\x94}\x94\x8c\x0e0x36329c240_33\x94}\x94\x8c\x0e0x36329c240_34\x94}\x94\x8c'
    b'\x0e0x36329c240_35\x94}\x94u\x8c\x04uuid\x94\x8c\x0b0x36329c240\x94\x8c\x06shapes\x94]\x94(]\x94('
    b'K\x00K\x01K\x02e]\x94(K\x00K\x03K\x04e]\x94(K\x05K\x00K\x02e]\x94(K\x00K\x05K\x03e]\x94('
    b'K\x05K\x06K\x03e]\x94(K\x07K\x05K\x02e]\x94(K\x05K\x07K\x08e]\x94(K\x05K\tK\x06e]\x94('
    b'K\tK\x05K\x08e]\x94(K\x08K\nK\te]\x94(K\nK\x0bK\te]\x94(K\x0cK\nK\x08e]\x94(K\nK\x0cK\re]\x94('
    b'K\nK\x0eK\x0be]\x94(K\x0fK\nK\re]\x94(K\nK\x0fK\x0ee]\x94(K\x0fK\x10K\x0ee]\x94(K\x11K\x0fK\re]\x94('
    b'K\x0fK\x11K\x12e]\x94(K\x0fK\x13K\x10e]\x94(K\x13K\x0fK\x12e]\x94(K\x12K\x00K\x13e]\x94('
    b'K\x00K\x04K\x13e]\x94(K\x01K\x00K\x12e]\x94(K\x12K\x02K\x01e]\x94(K\x12K\x11K\re]\x94('
    b'K\rK\x02K\x12e]\x94(K\x02K\rK\x08e]\x94(K\x08K\rK\x0ce]\x94(K\x07K\x02K\x08e]\x94('
    b'K\x13K\x04K\x03e]\x94(K\x13K\x0eK\x10e]\x94(K\x0eK\x13K\x03e]\x94(K\x03K\tK\x0ee]\x94('
    b'K\tK\x0bK\x0ee]\x94(K\x06K\tK\x03ee\x8c\x0e_normals_table\x94N\x8c\t_pt_faces\x94}\x94\x8c\x05cache\x94'
    b'}\x94(\x8c\x06repr3d\x94\x8c\x11mmcore.base.basic\x94\x8c\x05AMesh\x94\x93\x94)\x81\x94}\x94('
    b'\x8c\x05_uuid\x94hf\x8c\nchild_keys\x94\x8f\x94\x8c\t_children\x94h\x93\x8c\x08ChildSet\x94\x93\x94'
    b']\x94\x85\x94R\x94}\x94\x8c\x08instance\x94h\x96sb\x8c\x04name\x94\x8c\nPolyhedron\x94\x8c\t_geometry'
    b'\x94\x8c(7c5bbb453ab847d34bdf60d73abb141c2496625b\x94\x8c\t_material\x94\x8c'
    b'\x189868950meshphongmaterial\x94ub\x8c\x0bloop_shapes\x94}\x94('
    b'K\x00h\x00\x8c\x08Triangle\x94\x93\x94)\x81\x94}\x94('
    b'h\x05h\x06\x8c\x0fattribute_table\x94h\x1c\x8c\x03ixs\x94]\x94('
    b'K\x00K\x01K\x02e\x8c\x06points\x94h\x00\x8c\tShapeDCLL\x94\x93\x94)\x81\x94}\x94('
    b'h\x05h\x06\x8c\x04head\x94h\x00\x8c\nVertexNode\x94\x93\x94)\x81\x94}\x94('
    b'h\x05h\x06\x8c\x03ptr\x94K\x00\x8c\x04prev\x94h\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x02h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x01h\xbdh\xba\x8c\x04next\x94h\xbeubh\xc2h\xbaubh\xc2h\xc0ub\x8c\x05count\x94K\x03\x8c'
    b'\x05_temp\x94Nub\x8c\x05_tess\x94]\x94]\x94('
    b'K\x01K\x02K\x00ea\x8c\tmesh_data\x94\x8c\x10mmcore.base.geom\x94\x8c\x08MeshData\x94\x93\x94)\x81\x94'
    b'}\x94(\x8c\x08vertices\x94]\x94(h\x07h\x08h\te\x8c\x07normals\x94N\x8c\x07indices\x94\x8c\x15numpy.core'
    b'.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85'
    b'\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x03\x86\x94h\xd5\x8c\x05dtype\x94\x93\x94\x8c\x02i8\x94\x89\x88'
    b'\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x18\x01\x00\x00'
    b'\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94b\x8c'
    b'\x02uv\x94Nhe\x8c 23600a2dac6b4982b00dc190af88ce00\x94\x8c\x04_buf\x94Nubh\x98\x8c\r0x36329c240_0'
    b'\x94ubK\x01h\xac)\x81\x94}\x94(h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x00K\x03K\x04eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x00h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x04h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x03h\xbdh\xefh\xc2h\xf1ubh\xc2h\xefubh\xc2h\xf3ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94('
    b'K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x07h\nh\x0beh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'a6b2c42faa4042a587e661831066f846\x94h\xe8Nubh\x98\x8c\r0x36329c240_1\x94ubK\x02h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x05K\x00K\x02eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x05h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x02h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x00h\xbdj\x07\x01\x00\x00h\xc2j\t\x01\x00\x00ubh\xc2j\x07\x01\x00\x00ubh\xc2j\x0b\x01'
    b'\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x0ch\x07h\teh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'0e725cf1a4394a5e97e2bbc51ebd159c\x94h\xe8Nubh\x98\x8c\r0x36329c240_2\x94ubK\x03h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x00K\x05K\x03eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x00h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x03h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x05h\xbdj\x1f\x01\x00\x00h\xc2j!\x01\x00\x00ubh\xc2j\x1f\x01\x00\x00ubh\xc2j#\x01\x00'
    b'\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x07h\x0ch\neh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'086daa9412cb49549d06fc7d269dfb04\x94h\xe8Nubh\x98\x8c\r0x36329c240_3\x94ubK\x04h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x05K\x06K\x03eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x05h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x03h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x06h\xbdj7\x01\x00\x00h\xc2j9\x01\x00\x00ubh\xc2j7\x01\x00\x00ubh\xc2j;\x01\x00\x00ubh'
    b'\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x0ch\rh\neh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'a13950065fe941b7a9d31de8cfda8bc1\x94h\xe8Nubh\x98\x8c\r0x36329c240_4\x94ubK\x05h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x07K\x05K\x02eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x07h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x02h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x05h\xbdjO\x01\x00\x00h\xc2jQ\x01\x00\x00ubh\xc2jO\x01\x00\x00ubh\xc2jS\x01\x00\x00ubh'
    b'\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x0eh\x0ch\teh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'dde548eecdac4bdc8ea87c9fcf41b0c9\x94h\xe8Nubh\x98\x8c\r0x36329c240_5\x94ubK\x06h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x05K\x07K\x08eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x05h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x08h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x07h\xbdjg\x01\x00\x00h\xc2ji\x01\x00\x00ubh\xc2jg\x01\x00\x00ubh\xc2jk\x01\x00\x00ubh'
    b'\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x0ch\x0eh\x0feh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'ac22772dafe64dc0b6cb53c53df82ed6\x94h\xe8Nubh\x98\x8c\r0x36329c240_6\x94ubK\x07h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x05K\tK\x06eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x05h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x06h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\th\xbdj\x7f\x01\x00\x00h\xc2j\x81\x01\x00\x00ubh\xc2j\x7f\x01\x00\x00ubh\xc2j\x83\x01'
    b'\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x0ch\x10h\reh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'11e9b728397649248ccaead014eb708b\x94h\xe8Nubh\x98\x8c\r0x36329c240_7\x94ubK\x08h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\tK\x05K\x08eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\th\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x08h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x05h\xbdj\x97\x01\x00\x00h\xc2j\x99\x01\x00\x00ubh\xc2j\x97\x01\x00\x00ubh\xc2j\x9b'
    b'\x01\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x10h\x0ch\x0feh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'f0498d908f334fc8a96dc676e25e958a\x94h\xe8Nubh\x98\x8c\r0x36329c240_8\x94ubK\th\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x08K\nK\teh\xb2h\xb4)\x81\x94}\x94(h\x05h\x06h\xb7h\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x08h\xbdh\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\th\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\nh\xbdj\xaf\x01\x00\x00h\xc2j\xb1\x01\x00\x00ubh\xc2j\xaf\x01\x00\x00ubh\xc2j\xb3\x01'
    b'\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x0fh\x11h\x10eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'a4f6e5910b1e4bf7aa06a258cce1857f\x94h\xe8Nubh\x98\x8c\r0x36329c240_9\x94ubK\nh\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\nK\x0bK\teh\xb2h\xb4)\x81\x94}\x94(h\x05h\x06h\xb7h\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\nh\xbdh\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\th\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0bh\xbdj\xc7\x01\x00\x00h\xc2j\xc9\x01\x00\x00ubh\xc2j\xc7\x01\x00\x00ubh\xc2j\xcb'
    b'\x01\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x11h\x12h\x10eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'cfa5d0f802d148868008bcdf07eaa5fa\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_10\x94ubK\x0bh\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x0cK\nK\x08eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x0ch\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x08h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\nh\xbdj\xdf\x01\x00\x00h\xc2j\xe1\x01\x00\x00ubh\xc2j\xdf\x01\x00\x00ubh\xc2j\xe3\x01'
    b'\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x13h\x11h\x0feh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'4499a25873994c96ac197eb1db058a5c\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_11\x94ubK\x0ch\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\nK\x0cK\reh\xb2h\xb4)\x81\x94}\x94(h\x05h\x06h\xb7h\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\nh\xbdh\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\rh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0ch\xbdj\xf7\x01\x00\x00h\xc2j\xf9\x01\x00\x00ubh\xc2j\xf7\x01\x00\x00ubh\xc2j\xfb'
    b'\x01\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x11h\x13h\x14eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'c4d40941f89443ce9d8e59cec8657d63\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_12\x94ubK\rh\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\nK\x0eK\x0beh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\nh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0bh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0eh\xbdj\x0f\x02\x00\x00h\xc2j\x11\x02\x00\x00ubh\xc2j\x0f\x02\x00\x00ubh\xc2j\x13'
    b'\x02\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x11h\x15h\x12eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'3486f20d567242b99662990700b4ef31\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_13\x94ubK\x0eh\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x0fK\nK\reh\xb2h\xb4)\x81\x94}\x94(h\x05h\x06h\xb7h\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0fh\xbdh\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\rh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\nh\xbdj\'\x02\x00\x00h\xc2j)\x02\x00\x00ubh\xc2j\'\x02\x00\x00ubh\xc2j+\x02\x00\x00ubh'
    b'\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x16h\x11h\x14eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'd2c76ca9dd444027ad9a1eaee72af4a3\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_14\x94ubK\x0fh\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\nK\x0fK\x0eeh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\nh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0eh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0fh\xbdj?\x02\x00\x00h\xc2jA\x02\x00\x00ubh\xc2j?\x02\x00\x00ubh\xc2jC\x02\x00\x00ubh'
    b'\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x11h\x16h\x15eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'58697cf225774724a3d36fb7ea610231\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_15\x94ubK\x10h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x0fK\x10K\x0eeh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x0fh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0eh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x10h\xbdjW\x02\x00\x00h\xc2jY\x02\x00\x00ubh\xc2jW\x02\x00\x00ubh\xc2j['
    b'\x02\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x16h\x17h\x15eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'2981877a94df4603ad9a1cff117d8858\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_16\x94ubK\x11h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x11K\x0fK\reh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x11h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\rh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0fh\xbdjo\x02\x00\x00h\xc2jq\x02\x00\x00ubh\xc2jo\x02\x00\x00ubh\xc2js\x02\x00\x00ubh'
    b'\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x18h\x16h\x14eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'385d783315344717a27b59ed3f5dadfa\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_17\x94ubK\x12h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x0fK\x11K\x12eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x0fh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x12h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x11h\xbdj\x87\x02\x00\x00h\xc2j\x89\x02\x00\x00ubh\xc2j\x87\x02\x00\x00ubh\xc2j\x8b'
    b'\x02\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x16h\x18h\x19eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'cb174ce228434bc9b1f2bd6376699b5f\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_18\x94ubK\x13h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x0fK\x13K\x10eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x0fh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x10h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x13h\xbdj\x9f\x02\x00\x00h\xc2j\xa1\x02\x00\x00ubh\xc2j\x9f\x02\x00\x00ubh\xc2j\xa3'
    b'\x02\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x16h\x1ah\x17eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'ecb050e662c14764b9828edea09c943e\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_19\x94ubK\x14h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x13K\x0fK\x12eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x13h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x12h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0fh\xbdj\xb7\x02\x00\x00h\xc2j\xb9\x02\x00\x00ubh\xc2j\xb7\x02\x00\x00ubh\xc2j\xbb'
    b'\x02\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x1ah\x16h\x19eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'bd53ab48e08c4ecfb5483d15b0b9a140\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_20\x94ubK\x15h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x12K\x00K\x13eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x12h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x13h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x00h\xbdj\xcf\x02\x00\x00h\xc2j\xd1\x02\x00\x00ubh\xc2j\xcf\x02\x00\x00ubh\xc2j\xd3'
    b'\x02\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x19h\x07h\x1aeh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'818008418a9e43dea111b5ebb5b58193\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_21\x94ubK\x16h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x00K\x04K\x13eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x00h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x13h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x04h\xbdj\xe7\x02\x00\x00h\xc2j\xe9\x02\x00\x00ubh\xc2j\xe7\x02\x00\x00ubh\xc2j\xeb'
    b'\x02\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x07h\x0bh\x1aeh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'3e9aaf34e0574bc19cb88f9bdc627426\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_22\x94ubK\x17h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x01K\x00K\x12eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x01h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x12h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x00h\xbdj\xff\x02\x00\x00h\xc2j\x01\x03\x00\x00ubh\xc2j\xff\x02\x00\x00ubh\xc2j\x03'
    b'\x03\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x08h\x07h\x19eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'37a1f810db224abf977caafcd051a48f\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_23\x94ubK\x18h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x12K\x02K\x01eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x12h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x01h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x02h\xbdj\x17\x03\x00\x00h\xc2j\x19\x03\x00\x00ubh\xc2j\x17\x03\x00\x00ubh\xc2j\x1b'
    b'\x03\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x19h\th\x08eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'381928e6b9234db1967e1311d5521958\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_24\x94ubK\x19h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x12K\x11K\reh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x12h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\rh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x11h\xbdj/\x03\x00\x00h\xc2j1\x03\x00\x00ubh\xc2j/\x03\x00\x00ubh\xc2j3\x03\x00\x00ubh'
    b'\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x19h\x18h\x14eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'27c7303f0b8a46f3b90cdbb550a397d5\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_25\x94ubK\x1ah\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\rK\x02K\x12eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\rh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x12h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x02h\xbdjG\x03\x00\x00h\xc2jI\x03\x00\x00ubh\xc2jG\x03\x00\x00ubh\xc2jK\x03\x00\x00ubh'
    b'\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x14h\th\x19eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'bdca490a7e80457e8b38acf42159208c\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_26\x94ubK\x1bh\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x02K\rK\x08eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x02h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x08h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\rh\xbdj_\x03\x00\x00h\xc2ja\x03\x00\x00ubh\xc2j_\x03\x00\x00ubh\xc2jc\x03\x00\x00ubh'
    b'\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\th\x14h\x0feh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'e2fa62711283409f886d794303cf2e4b\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_27\x94ubK\x1ch\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x08K\rK\x0ceh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x08h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0ch\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\rh\xbdjw\x03\x00\x00h\xc2jy\x03\x00\x00ubh\xc2jw\x03\x00\x00ubh\xc2j{'
    b'\x03\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x0fh\x14h\x13eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'face7061799d43db8bada74bafadbd22\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_28\x94ubK\x1dh\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x07K\x02K\x08eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x07h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x08h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x02h\xbdj\x8f\x03\x00\x00h\xc2j\x91\x03\x00\x00ubh\xc2j\x8f\x03\x00\x00ubh\xc2j\x93'
    b'\x03\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x0eh\th\x0feh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'b6315a604c1c44098b7607dff92a987a\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_29\x94ubK\x1eh\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x13K\x04K\x03eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x13h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x03h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x04h\xbdj\xa7\x03\x00\x00h\xc2j\xa9\x03\x00\x00ubh\xc2j\xa7\x03\x00\x00ubh\xc2j\xab'
    b'\x03\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x1ah\x0bh\neh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'267eee196667486c92867d8511229b37\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_30\x94ubK\x1fh\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x13K\x0eK\x10eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x13h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x10h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0eh\xbdj\xbf\x03\x00\x00h\xc2j\xc1\x03\x00\x00ubh\xc2j\xbf\x03\x00\x00ubh\xc2j\xc3'
    b'\x03\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x1ah\x15h\x17eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'0397c3d810e44fa4bb94e1385ea99584\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_31\x94ubK h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x0eK\x13K\x03eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x0eh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x03h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x13h\xbdj\xd7\x03\x00\x00h\xc2j\xd9\x03\x00\x00ubh\xc2j\xd7\x03\x00\x00ubh\xc2j\xdb'
    b'\x03\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x15h\x1ah\neh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'c027669469754f7491f66ea6025139b9\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_32\x94ubK!h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x03K\tK\x0eeh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x03h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0eh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\th\xbdj\xef\x03\x00\x00h\xc2j\xf1\x03\x00\x00ubh\xc2j\xef\x03\x00\x00ubh\xc2j\xf3\x03'
    b'\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\nh\x10h\x15eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'7a3c1fefb7564452bb1feea81b46a262\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_33\x94ubK"h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\tK\x0bK\x0eeh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\th\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0eh\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x0bh\xbdj\x07\x04\x00\x00h\xc2j\t\x04\x00\x00ubh\xc2j\x07\x04\x00\x00ubh\xc2j\x0b\x04'
    b'\x00\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\x10h\x12h\x15eh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'713fa900e9864394939d5216962efeb2\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_34\x94ubK#h\xac)\x81\x94}\x94('
    b'h\x05h\x06h\xafh\x1ch\xb0]\x94(K\x06K\tK\x03eh\xb2h\xb4)\x81\x94}\x94('
    b'h\x05h\x06h\xb7h\xb9)\x81\x94}\x94(h\x05h\x06h\xbcK\x06h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\x03h\xbdh\xb9)\x81\x94}\x94('
    b'h\x05h\x06h\xbcK\th\xbdj\x1f\x04\x00\x00h\xc2j!\x04\x00\x00ubh\xc2j\x1f\x04\x00\x00ubh\xc2j#\x04\x00'
    b'\x00ubh\xc3K\x03h\xc4Nubh\xc5]\x94]\x94(K\x01K\x02K\x00eah\xc8h\xcb)\x81\x94}\x94(h\xce]\x94('
    b'h\rh\x10h\neh\xd0Nh\xd1h\xd4h\xd7K\x00\x85\x94h\xd9\x87\x94R\x94('
    b'K\x01K\x01K\x03\x86\x94h\xe1\x89C\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bh\xe6Nhe\x8c '
    b'4cd1556f7a544201be944c2ad50678f7\x94h\xe8Nubh\x98\x8c\x0e0x36329c240_35\x94ubuuub.')
poly = pickle.loads(poly_bytes)

plane = PlaneLinear(origin=(-1667.5098696403961, -953.9487618748584, 704.9462405738814),
                    normal=np.array([-0.55429341, 0.78847654, -0.2665775]),
                    xaxis=np.array([-0.36132633, -0.51648118, -0.77633142]),
                    yaxis=np.array([-0.74980137, -0.33399392, 0.57117945]))


def cut_polyhedron(polyhedron, plane):
    grp = AGroup(uuid="cut_example")
    a, b = polyhedron.plane_solid_split(plane)

    meshb = b.to_mesh()
    grp.add(meshb)
    grp.add(a.to_mesh())
    meshb.translate(-3 * plane.normal)
    return grp


if __name__ == "__main__":
    cut_polyhedron(poly, plane).dump("cut-example.json")
