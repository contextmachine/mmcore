import numpy as np

from mmcore.geom.shapes.base import OccLikeLoop


def bounds(outer=[[-241245.93137, -10699.151687, 0],
                  [-126531.615261, -6806.179119, 0],
                  [-143306.718695, -27002.024251, 0],
                  [-229852.594394, -39899.110783, 0]
                  ],

           # x =  np.array([(3829.7653620639830, -259.05847478062663, 0.0), (-18.839782761661354, 1.2743875736564974, 0.0), (152.61152519640734, 2853.5259909564056, 0.0), (4196.9531817063826, 3178.7879245086524, 0.0)]).tolist()
           inner=np.array([[-213065.707534, -35086.371292, 0],
                           [-208372.861742, -14759.249076, 0],
                           [-187527.064191, -14063.649196, 0],
                           [-186302.555933, -31077.040224, 0],
                           ]).tolist()):
    inner_border = OccLikeLoop(*inner)
    outer_border = OccLikeLoop(*outer)
    inner_border.width = 32000
    outer_border.width = 24000
    return inner_border, outer_border
