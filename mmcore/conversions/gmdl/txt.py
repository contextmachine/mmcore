from geomdl import _exchange as exch
from geomdl._utilities import export
from geomdl.exceptions import GeomdlException


def read_file(file, **kwargs):
    binary = kwargs.get('binary', False)
    skip_lines = kwargs.get('skip_lines', 0)
    callback = kwargs.get('callback', None)
    try:
        for _ in range(skip_lines):
            next(file)
        content = fp.read() if callback is None else callback(fp)
        return content
    except IOError as e:
        raise GeomdlException("An error occurred during reading '{0}': {1}".format(file_name, e.args[-1]))
    except Exception as e:
        raise GeomdlException("An error occurred: {0}".format(str(e)))


def write_file(file, content, **kwargs):
    binary = kwargs.get('binary', False)
    callback = kwargs.get('callback', None)
    try:

        if callback is None:
            fp.write(content)
        else:
            callback(fp, content)
        return True
    except IOError as e:
        raise GeomdlException("An error occurred during writing '{0}': {1}".format(file_name, e.args[-1]))
    except Exception as e:
        raise GeomdlException("An error occurred: {0}".format(str(e)))


@export
def import_txt(file, two_dimensional=False, **kwargs):
    """ Reads control points from a text file and generates a 1-dimensional list of control points.
    The following code examples illustrate importing different types of text files for curves and surfaces:
    .. code-block:: python
        :linenos:
        # Import curve control points from a text file
        curve_ctrlpts = exchange.import_txt(file_name="control_points.txt")
        # Import surface control points from a text file (1-dimensional file)
        surf_ctrlpts = exchange.import_txt(file_name="control_points.txt")
        # Import surface control points from a text file (2-dimensional file)
        surf_ctrlpts, size_u, size_v = exchange.import_txt(file_name="control_points.txt", two_dimensional=True)
    If argument ``jinja2=True`` is set, then the input file is processed as a `Jinja2 <http://jinja.pocoo.org/>`_
    template. You can also use the following convenience template functions which correspond to the given mathematical
    equations:
    * ``sqrt(x)``:  :math:`\\sqrt{x}`
    * ``cubert(x)``: :math:`\\sqrt[3]{x}`
    * ``pow(x, y)``: :math:`x^{y}`
    You may set the file delimiters using the keyword arguments ``separator`` and ``col_separator``, respectively.
    ``separator`` is the delimiter between the coordinates of the control points. It could be comma
    ``1, 2, 3`` or space ``1 2 3`` or something else. ``col_separator`` is the delimiter between the control
    points and is only valid when ``two_dimensional`` is ``True``. Assuming that ``separator`` is set to space, then
    ``col_operator`` could be semi-colon ``1 2 3; 4 5 6`` or pipe ``1 2 3| 4 5 6`` or comma ``1 2 3, 4 5 6`` or
    something else.
    The defaults for ``separator`` and ``col_separator`` are *comma (,)* and *semi-colon (;)*, respectively.
    The following code examples illustrate the usage of the keyword arguments discussed above.
    .. code-block:: python
        :linenos:
        # Import curve control points from a text file delimited with space
        curve_ctrlpts = exchange.import_txt(file_name="control_points.txt", separator=" ")
        # Import surface control points from a text file (2-dimensional file) w/ space and comma delimiters
        surf_ctrlpts, size_u, size_v = exchange.import_txt(file_name="control_points.txt", two_dimensional=True,
                                                           separator=" ", col_separator=",")
    Please note that this function does not check whether the user set delimiters to the same value or not.
    :param file_name: file name of the text file
    :type file_name: str
    :param two_dimensional: type of the text file
    :type two_dimensional: bool
    :return: list of control points, if two_dimensional, then also returns size in u- and v-directions
    :rtype: list
    :raises GeomdlException: an error occurred reading the file
    """
    # Read file
    content = file.read()

    # Are we using a Jinja2 template?
    j2tmpl = kwargs.get('jinja2', False)
    if j2tmpl:
        content = exch.process_template(content)

    # File delimiters
    col_sep = kwargs.get('col_separator', ";")
    sep = kwargs.get('separator', ",")

    return exch.import_text_data(content, sep, col_sep, two_dimensional)


@export
def export_txt(obj, file, two_dimensional=False, **kwargs):
    """ Exports control points as a text file.
    For curves the output is always a list of control points. For surfaces, it is possible to generate a 2-dimensional
    control point output file using ``two_dimensional``.
    Please see :py:func:`.exchange.import_txt()` for detailed description of the keyword arguments.
    :param obj: a spline geometry object
    :type obj: abstract.SplineGeometry
    :param file_name: file name of the text file to be saved
    :type file_name: str
    :param two_dimensional: type of the text file (only works for Surface objects)
    :type two_dimensional: bool
    :raises GeomdlException: an error occurred writing the file
    """
    # Check if the user has set any control points
    if obj.ctrlpts is None or len(obj.ctrlpts) == 0:
        raise exch.GeomdlException("There are no control points to save!")

    # Check the usage of two_dimensional flag
    if obj.pdimension == 1 and two_dimensional:
        # Silently ignore two_dimensional flag
        two_dimensional = False

    # File delimiters
    col_sep = kwargs.get('col_separator', ";")
    sep = kwargs.get('separator', ",")

    content = exch.export_text_data(obj, sep, col_sep, two_dimensional)
    return write_file(file, content)
import rhino3dm._rhino3dm as rh
rh.Transform.Rotation()
rhino3dm.Transform.