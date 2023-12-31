import numpy as np
from earcut import earcut

from mmcore.func import vectorize


def ecut(self) -> tuple:
    if self.holes is None:
        arguments = earcut.flatten([self.boundary])
    else:
        arguments = earcut.flatten([self.boundary] + self.holes)

    return arguments['vertices'], earcut.earcut(arguments['vertices'], arguments['holes'],
                                                arguments['dimensions']), arguments


def polygon_area(poly, cast_to_numpy=True) -> float:
    """
    Реализация формула площади Гаусса с использованием numpy. Данная реализация превосходит по скорости (примерно в 10x)
    альтернативы из пакетов shapely и compas, однако в oтличае от shapely не поддерживает сложные полигоны. Бенчмарки приведены ниже.

    Если полигон имеет вид [p0,p1,p2,..,pn], передать необходимо [p0,p1,p2,..,pn,p0].
    Это приведение не делается автоматически, в том числе для того,
    чтобы избежать проверки на то является ли poly массивом numpy или списком, а также просто ради здравого смысла.
    Вы также можете избежать создания нового numpy массива если зададите cast_to_numpy=False. Это в целом благоприятно
    влияет на производительность, особенно на больших наборах.

    Parameters
    ----------
    poly :

    Returns
    -------
    float value of area

    Benchmarks
    -------
    >>> import shapely
    >>> import compas.geometry as cg
    >>> import time
    >>> def bench(N):
    ...     dat1=[(-180606.0, 23079.0, 0.0), (-181468.0, 59713.0, 0.0), (-173710.0, 59713.0, 0.0), (-173710.0, 49369.0, 0.0),
    ...      (-177589.0, 49585.0, 0.0), (-177589.0, 40965.0, 0.0), (-168753.0, 40965.0, 0.0), (-168969.0, 48076.0, 0.0),
    ...      (-164012.0, 60791.0, 0.0), (-151082.0, 61222.0, 0.0), (-156254.0, 17907.0, 0.0),(-180606.0, 23079.0, 0.0)]
    ...     dat = np.array(dat1+[dat1[0]])
    ...     s = time.time()
    ...     for i in range(N):
    ...         pgn = polygon_area(dat, cast_to_numpy=False)
    ...     print(f'[mmcore](without casting) {N} items at: ', divmod(time.time() - s, 60))
    ...     s = time.time()
    ...     for i in range(N):
    ...         pgn = polygon_area(dat)
    ...     print(f'[mmcore](with casting) {N} items at: ', divmod(time.time() - s, 60))
    ...
    ...     s=time.time()
    ...     for i in range(N):
    ...         pgn = shapely.area(shapely.Polygon(dat1))
    ...
    ...     print(f'[shapely] {N} items at: ',divmod(time.time()-s,60))
    ...     s = time.time()
    ...     for i in range(N):
    ...         pgn = cg.Polygon(
    ...             dat1)
    ...         pgn.area
    ...     print(f'[compas] {N} items at: ', divmod(time.time() - s, 60))
    ...
    >>> bench(10_000)
[mmcore](without casting) 10000 items at:  (0.0, 0.02608180046081543)
[mmcore](with casting) 10000 items at:  (0.0, 0.019220829010009766)
[shapely] 10000 items at:  (0.0, 0.1269359588623047)
[compas] 10000 items at:  (0.0, 0.19669032096862793)
    >>> bench(100_000)
[mmcore](without casting) 100000 items at:  (0.0, 0.16399192810058594)
[mmcore](with casting) 100000 items at:  (0.0, 0.1786212921142578)
[shapely] 100000 items at:  (0.0, 1.268747091293335)
[compas] 100000 items at:  (0.0, 1.969149112701416)
    >>> bench(1_000_000)
[mmcore](without casting) 1000000 items at:  (0.0, 1.5401110649108887)
[mmcore](with casting) 1000000 items at:  (0.0, 1.7677040100097656)
[shapely] 1000000 items at:  (0.0, 12.567844152450562)
[compas] 1000000 items at:  (0.0, 21.297150135040283)
    """

    if cast_to_numpy:
        poly = np.array(poly)
    length = poly.shape[0] - 1
    return np.abs(np.dot(poly[:length, 1], poly[1:, 0]) - np.dot(poly[:length, 0], poly[1:, 1])) / 2


polygon_area_vectorized = np.vectorize(polygon_area,
                                       doc="polygon_area обернутая в np.vectorize, что делает ее более подходящей для работы с массивами.\n\n" + polygon_area.__doc__,
                                       otypes=[float],
                                       excluded=['cast_to_numpy'],
                                       signature='(i,n)->()')

@vectorize(signature='(i,j)->(k,j)')
def to_polygon_area(boundary: np.ndarray):
    """
    Добавляет первую точку контура в конец чтобы соответствовать требованиям polygon_area
    :param boundary:
    :type boundary:
    :return:
    :rtype:
    """
    return np.array([*boundary, boundary[0]],dtype=boundary.dtype)