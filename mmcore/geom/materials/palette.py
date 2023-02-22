from mmcore.geom.materials import ColorRGB
from mmcore.gql.client import GQLMutation


def color_palette(colors, i=None, **kwargs):
    """
    >>> from mmcore.geom.materials import ColorRGB

    >>> colors = [
    ...    {"name":"Malachite","hex":"11e159","rgb":[17,225,89],"cmyk":[92,0,60,12]},
    ...    {"name":"Jade","hex":"21ad6f","rgb":[33,173,111],"cmyk":[81,0,36,32]},
    ...    {"name":"Viridian","hex":"29937a","rgb":[41,147,122],"cmyk":[72,0,17,42]},
    ...    {"name":"Caribbean Current","hex":"307885","rgb":[48,120,133],"cmyk":[64,10,0,48]},
    ...    {"name":"Marian blue","hex":"40439b","rgb":[64,67,155],"cmyk":[59,57,0,39]},
    ...    {"name":"Tekhelet","hex":"5718a7","rgb":[87,24,167],"cmyk":[48,86,0,35]},
    ...    {"name":"Eminence","hex":"6f3689","rgb":[111,54,137],"cmyk":[19,61,0,46]},
    ...    {"name":"Rose taupe","hex":"8e5e61","rgb":[142,94,97],"cmyk":[0,34,32,44]},
    ...    {"name":"Dark goldenrod","hex":"ad863a","rgb":[173,134,58],"cmyk":[0,23,66,32]},
    ...    {"name":"Old gold","hex":"ccad12","rgb":[204,173,18],"cmyk":[0,15,91,20]}
    ... ]

    >>> mutation = GQLMutation(temp_args=dict(attrs=("id", "name", "palette")),schema="presets",table="colors")

    >>> new_colors=[]
    >>> palette=12345
    >>> for k in colors:
    ...    col = ColorRGB(*k["rgb"])
    ...    dct = {"palette": palette, "name": k['name'], "hex": col.hex(), "decimal": col.decimal}
    ...    dct.update(col.__dict__)
    ...    col.__dict__ |= mt.run_query(variables=dct)
    ...    new_colors.append(col)
    >>> new_colors
    [ColorRGB(17, 225, 89),
     ColorRGB(33, 173, 111),
     ColorRGB(41, 147, 122),
     ColorRGB(48, 120, 133),
     ColorRGB(64, 67, 155),
     ColorRGB(87, 24, 167),
     ColorRGB(111, 54, 137),
     ColorRGB(142, 94, 97),
     ColorRGB(173, 134, 58),
     ColorRGB(204, 173, 18)]
    >>> col = new_colors[0]
    >>> col.id
    52
    >>> col.palette
    12345

    """
    mt = GQLMutation(temp_args=dict(attrs=("id", "name", "palette")),
                     schema="presets",
                     table="colors"
                     )

    palette = id(mt) if i is None else i
    nc = []
    for k in colors:
        col = ColorRGB(*k["rgb"])
        dct = {"palette": palette, "name": k['name'], "hex": col.hex(), "decimal": col.decimal}
        dct.update(col.__dict__)

        col.__dict__ |= mt.run_query(variable=dct)
        nc.append(col)
    return nc
