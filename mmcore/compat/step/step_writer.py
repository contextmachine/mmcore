from __future__ import annotations

from collections import OrderedDict

from typing import Any

import numpy as np

from mmcore import __version__
from steputils import p21
from itertools import count
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface

from mmcore.geom._nurbs_eval import _tuple_to_nurbs, NURBSSurfaceTuple, NURBSCurveTuple
from mmcore.geom.nurbs_iso import extract_surface_boundaries, extract_surface_boundaries_tuple

import re


def parse_assignment(assignment_str):
    """
    Parses an assignment string and returns the function name and arguments as a tuple.

    Parameters:
        assignment_str (str): The assignment string to parse.

    Returns:
        tuple: A tuple containing the function name and a tuple of arguments.
    """
    # Remove any leading/trailing whitespace and the trailing semicolon
    assignment_str = assignment_str.strip().rstrip(';')

    # Regex to match the pattern: #number=FUNCTION_NAME(args)
    match = re.match(r'#\d+\s*=\s*([A-Z_]+)\s*\((.*)\)', assignment_str)
    if not match:
        raise ValueError("String does not match the expected format.")

    func_name = match.group(1)
    args_str = match.group(2)

    args = parse_arguments(args_str)

    return func_name, args


def parse_arguments(arg_str):
    """
    Recursively parses the argument string into nested tuples, handling quoted strings and numeric values.

    Parameters:
        arg_str (str): The argument string to parse.

    Returns:
        tuple: A tuple representing the parsed arguments.
    """
    args = []
    current = ''
    in_quote = False
    quote_char = ''

    i = 0
    while i < len(arg_str):
        char = arg_str[i]

        if in_quote:
            if char == quote_char:
                in_quote = False
                current += char
            else:
                current += char
            i += 1
            continue

        if char in ("'", '"'):
            in_quote = True
            quote_char = char
            current += char
            i += 1
            continue

        if char == '(':
            # Find the matching closing parenthesis
            count = 1
            i += 1
            start = i
            while i < len(arg_str) and count > 0:
                if arg_str[i] == '(':
                    count += 1
                elif arg_str[i] == ')':
                    count -= 1
                i += 1
            if count != 0:
                raise ValueError("Unbalanced parentheses in arguments.")
            # Recursively parse the substring inside the parentheses
            nested_str = arg_str[start:i - 1]
            nested = parse_arguments(nested_str)
            args.append(nested)
            continue
        elif char == ',':
            if current.strip():
                args.append(process_token(current.strip()))
                current = ''
            i += 1
            continue
        else:
            current += char
            i += 1

    if current.strip():
        args.append(process_token(current.strip()))

    return tuple(args)


def process_token(token):
    """
    Processes a single token, stripping quotes if present and converting to appropriate type.

    Parameters:
        token (str): The token to process.

    Returns:
        str, int, float, or tuple: The processed token.
    """
    # If the token starts and ends with quotes, strip them and return as string
    if (token.startswith("'") and token.endswith("'")) or (token.startswith('"') and token.endswith('"')):
        return token[1:-1]

    # Try to convert to integer
    try:
        return int(token)
    except ValueError:
        pass

    # Try to convert to float
    try:
        return float(token)
    except ValueError:
        pass

    # Otherwise, return as string
    return token


COMPLEX_ENTITY_INSTANCE='COMPLEX_ENTITY_INSTANCE'
CARTESIAN_POINT="CARTESIAN_POINT"
VERTEX_POINT="VERTEX_POINT"

ANY=p21.UnsetParameter('*')
UNSPECIFIED=p21.Enumeration('.UNSPECIFIED.')
FALSE=p21.Enumeration(".F.")
TRUE=p21.Enumeration(".T.")


def get_knot_multiplicities(knots):
    unique_knots = []
    multiplicities = []
    if not knots:
        return unique_knots, multiplicities
    last_knot = knots[0]
    count = 1
    for i in range(1, len(knots)):
        if abs(knots[i] - last_knot) < 1e-12:
            count += 1
        else:
            unique_knots.append(last_knot)
            multiplicities.append(count)
            last_knot = knots[i]
            count = 1
    unique_knots.append(last_knot)
    multiplicities.append(count)

    return unique_knots, multiplicities


class StepWriter:
    """
    Represents a writer for STEP files, which are used for exchanging digital information related to 3D models.

    This class provides methods to create and manipulate entities within a STEP file, allowing users to define complex 3D models with geometric and topological information. The class initializes with a default structure and provides various methods to add specific STEP entities. It maintains internal counters and references to ensure the uniqueness of STEP entities and provides functionality to output the final STEP file format.

    Attributes:
        step_file (p21.StepFile): The STEP file object being manipulated.
        tolerance (float): The tolerance value for geometric measurements.
        world_plane (tuple): Defines the world plane in 3D space.
        last_ref (p21.Reference): The last created reference for STEP entities.

    Methods:
        __init__(step_file: p21.StepFile = None, tolerance: float = 1e-7, world_plane: tuple = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.))):
            Initializes the StepWriter with default values and prepares the initial STEP file structure.
        write(fl: Any):
            Writes the current STEP file data to the given output.
        next_ref():
            Generates and returns the next unique reference for a STEP entity.
        add_nurbs_curve(curve: NURBSCurve, name: str = ''):
            Adds a NURBS curve entity to the STEP file.
        issteptype(obj):
            Checks if the given object is of a STEP type (Reference, SimpleEntityInstance, ComplexEntityInstance).
        typeof(ref: p21.Reference):
            Gets the STEP entity type of the given reference.
        add_edge_curve(start, end, geometry, same_sense: bool = True, name: str = ''):
            Adds an edge curve entity to the STEP file.
        add_vertex_point(pt: p21.Reference | Any, name: str = ''):
            Adds a vertex point entity to the STEP file.
        add_oriented_edge(edge, start: Any = ANY, end: Any = ANY, orientation: bool = True, name: str = ''):
            Adds an oriented edge entity to the STEP file.
        add_edge_loop(edges, name: str = ''):
            Adds an edge loop entity to the STEP file.
        add_face_bound(loop, orientation: bool = TRUE, name: str = ''):
            Adds a face bound entity to the STEP file.
        add_open_shell(faces, name: str = ''):
            Adds an open shell entity to the STEP file.
        add_advanced_face(loops, face_geometry, same_sense: bool = TRUE, name: str = ''):
            Adds an advanced face entity to the STEP file.
        add_shell_based_surface_model(shells, name: str = ''):
            Adds a shell-based surface model entity to the STEP file.
        add_manifold_surface_shape_representation(representations, context: Any = None, name: str = ''):
            Adds a manifold surface shape representation entity to the STEP file.
        add_units():
            Adds units of measurement to the STEP file.
        add_context3():
            Adds the default geometric representation context to the STEP file.
    """
    def __init__(self, step_file:p21.StepFile=None, tolerance=1e-5,world_plane= ((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.))):
        if step_file is None:
            step_file=p21.StepFile()
        self.step_file=step_file
        self.step_file.data=[p21.DataSection()]
        self.tolerance=tolerance
        self.step_file.header=p21.HeaderSection(entities=OrderedDict(
            {'FILE_DESCRIPTION': p21.entity('FILE_DESCRIPTION', (('',), '2;1')),

             'FILE_NAME':p21.entity('FILE_NAME', ('nc', '2024-11-15T02:15:59+03:00', ('Unspecified',), ('Unspecified',), f'mmcore@{__version__}',  'Unspecified', '')),
             'FILE_SCHEMA':p21.entity('FILE_SCHEMA',(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 3 1 4 }',),))

             }))

        self._counter=count()
        self._last_ref=self._counter.__next__()
        self.last_ref=p21.Reference(f'#{self._last_ref}')

        self._1 = self.add_entity(
            p21.entity('APPLICATION_CONTEXT', ('core data for automotive mechanical design processes',)))
        self._2 = self.add_entity(
            p21.entity('APPLICATION_PROTOCOL_DEFINITION', ('international standard', 'automotive_design', 2000, self._1)))
        self._3 = self.add_entity(p21.entity('PRODUCT_CONTEXT', ('', self._1, 'mechanical')))
        self._4 = self.add_entity(p21.entity('PRODUCT', ('Document', 'Document', p21.UnsetParameter('$'), (self._3))))
        self._5 = self.add_entity(p21.entity('PRODUCT_DEFINITION_FORMATION', ('', p21.UnsetParameter('$'), self._4)))
        self._6 = self.add_entity(p21.entity('PRODUCT_DEFINITION_CONTEXT', ('part definition', self._1, 'design')))
        self._7 = self.add_entity(p21.entity('PRODUCT_DEFINITION', ('design', p21.UnsetParameter('$'), self._5, self._6)))
        self._8 = self.add_entity(p21.entity('PRODUCT_DEFINITION_SHAPE', ('', p21.UnsetParameter('$'), self._7)))
        self._context3=None
        self._context3= self.add_context3()
        self.world_plane=self.add_plane(world_plane)
        # self.base_shape_representation=self.add_shape_representation((self.world_plane[-2],self.world_plane[-1]), self._context3,'Document')

    def write(self,fl):
        return self.step_file.write(fl)

    def next_ref(self):
        self._last_ref=        self._counter.__next__()
        self.last_ref=p21.Reference(f'#{self._last_ref}')
        return self.last_ref

    def add_b_spline_curve_with_knots(self, curve:NURBSCurve, name:str=''):
        unique_knots,mult= get_knot_multiplicities(curve.knots.tolist())
        return self.add_entity(p21.entity('B_SPLINE_CURVE_WITH_KNOTS',
                                          (
                                              name,
                                              int(curve.degree),
                                              [self.add_cartesian_point(pt) for pt in curve.control_points],
                                              UNSPECIFIED,
                                              FALSE,
                                              FALSE,
                                              mult,
                                              unique_knots,
                                              UNSPECIFIED
                                            )
                                          )
                               )
    def add_rational_b_spline_curve_with_knots(self, curve:NURBSCurve|NURBSCurveTuple, curve_form=UNSPECIFIED,closed_curve:bool=False,self_intersect:bool=False,name:str=''):

        weights=list(curve.weights) if not isinstance(curve.weights, list) else curve.weights

        knot_raw = curve.knots if isinstance(curve, NURBSCurve) else curve.knot
        unique_knots, mult = get_knot_multiplicities(list(knot_raw))
        return self.add_complex_entity(
            [
                p21.entity("BOUNDED_CURVE".upper(), ()),
                p21.entity(
                    "B_SPLINE_CURVE",
                    (
                        int(curve.degree),
                        [self.add_cartesian_point(pt) for pt in curve.control_points],
                        curve_form,
                        TRUE if closed_curve else FALSE,
                        TRUE if self_intersect else FALSE,
                    ),
                ),

                p21.entity("GEOMETRIC_REPRESENTATION_ITEM", ()),
                p21.entity("B_SPLINE_CURVE_WITH_KNOTS", (mult, unique_knots, UNSPECIFIED)),
                p21.entity("RATIONAL_B_SPLINE_CURVE", (weights,)),
                p21.entity("REPRESENTATION_ITEM", (name,)),
            ]
        )

    def issteptype(self, obj):
        return type(obj) in (p21.Reference, p21.SimpleEntityInstance, p21.ComplexEntityInstance)

    def typeof(self, ref:p21.Reference):
        item=self.step_file.data[0].instances[ref]
        if hasattr(item,'entity'):
            return item.entity.name
        elif hasattr(item,'entities'):
            return COMPLEX_ENTITY_INSTANCE
        else:
            raise ValueError(f'{type(ref)} is not STEP type.')
    def add_edge_curve(self, start, end, geometry, same_sense=True, name:str=''):
        if same_sense==True:
            same_sense=TRUE
        elif same_sense==False:
            same_sense=FALSE

        return self.add_entity(p21.entity('EDGE_CURVE',(name, self.add_vertex_point(start),self.add_vertex_point(end), geometry, same_sense)))

    def add_vertex_point(self, pt:p21.Reference|Any,name=''):
        if not self.issteptype(pt):
            return self.add_vertex_point(self.add_cartesian_point(pt))
        elif self.typeof(pt)==CARTESIAN_POINT:
            return self.add_entity(
                p21.entity(VERTEX_POINT, (name, pt)))
        elif self.typeof(pt)==VERTEX_POINT:
            return pt
        else:
            raise ValueError(f"{pt} not a valid VERTEX_POINT arg.")

    def add_oriented_edge(self, edge, start=ANY,end=ANY, orientation=True,name: str = ''):
        if orientation==True:
            orientation=TRUE
        elif orientation==False:
            orientation=FALSE

        return self.add_entity(p21.entity('ORIENTED_EDGE', (name, start, end, edge, orientation)))

    def add_edge_loop(self, edges, name:str=''):
        _edges=[]
        for edge in edges:
            if self.issteptype(edge):
                if self.typeof(edge)=='ORIENTED_EDGE':
                    _edges.append(edge)
                elif self.typeof(edge)== 'EDGE_CURVE':
                    _edges.append(self.add_oriented_edge(edge))
                else:
                    raise ValueError(f"{edge} must be reference to ORIENTED_EDGE or EDGE_CURVE")
            else:
                raise ValueError(f"{edge} must be reference to ORIENTED_EDGE or EDGE_CURVE")
        return self.add_entity(p21.entity(
            "EDGE_LOOP",(name, _edges)
        ))
    def add_face_bound(self, loop,orientation=TRUE,name:str=''):
        return self.add_entity(p21.entity(
            "FACE_BOUND", (name, loop, orientation)
        ))

    def add_open_shell(self, faces,name:str=''):
        return self.add_entity(p21.entity(
            "OPEN_SHELL", (name, faces)
        ))
    def add_closed_shell(self, faces,name:str=''):
        return self.add_entity(p21.entity(
            "CLOSED_SHELL", (name, faces)
        ))

    def add_advanced_face(self, loops, face_geometry, same_sense=TRUE,name:str=''):
        return self.add_entity(p21.entity(
            "ADVANCED_FACE", (name, loops, face_geometry, same_sense)
        ))
    def add_shell_based_surface_model(self, shells,name:str=''):
        return self.add_entity(p21.entity(
            "SHELL_BASED_SURFACE_MODEL", (name,shells)
        ))

    def add_manifold_surface_shape_representation(self, representations,context=None, name: str = ''):
        return self.add_entity(p21.entity(
            "MANIFOLD_SURFACE_SHAPE_REPRESENTATION", (name, representations, context if context is not None else self.add_context3()
                                                      )
        ))

    def add_units(self):
        # 54=(

        LENGTH_UNITS=self.add_complex_entity([p21.entity('LENGTH_UNIT',
                                 ()
                                 ),
                      p21.entity('NAMED_UNIT',
                                 (ANY,)
                                 ),
                      p21.entity(
                          'SI_UNIT',
                          (p21.Enumeration('.MILLI.'),p21.Enumeration('.METRE.'))
                        )
                      ])

        ANGLE_UNITS=self.add_complex_entity([
        p21.entity(
            'NAMED_UNIT', (ANY,)),
        p21.entity(
            'PLANE_ANGLE_UNIT',()),
        p21.entity('SI_UNIT',(p21.UnsetParameter('$'), p21.Enumeration('.RADIAN.')))
        ])

        p59=self.add_complex_entity(
            [p21.entity(
            'NAMED_UNIT', (ANY,)),
        p21.entity('SI_UNIT',(p21.UnsetParameter('$'),p21.Enumeration('.STERADIAN.'))),
        p21.entity('SOLID_ANGLE_UNIT', ())]
        )

        UNCERTAINTY_MEASURE_WITH_UNIT=self.add_entity(p21.entity('UNCERTAINTY_MEASURE_WITH_UNIT',(p21.entity('LENGTH_MEASURE',(self.tolerance,)),LENGTH_UNITS,'','maximum tolerance')))

        return (LENGTH_UNITS,ANGLE_UNITS,p59,UNCERTAINTY_MEASURE_WITH_UNIT)
    def add_context3(self):
        # 61=(

        if self._context3 is None:
            units = self.add_units()
            self._context3=self.add_complex_entity([p21.entity('GEOMETRIC_REPRESENTATION_CONTEXT', (3,)),
            p21.entity('GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT',( (units[-1],),)),
            p21.entity(
            'GLOBAL_UNIT_ASSIGNED_CONTEXT',((  units[0],units[1],units[2]),)),

            p21.entity('REPRESENTATION_CONTEXT',('ID1', '3D'))])
        return self._context3
    def add_axis_axis2_placement_3d(self, origin, x_axis,y_axis,name=''):
        return self.add_entity(p21.entity('AXIS2_PLACEMENT_3D', (name,origin,x_axis,y_axis)))

    def add_plane(self,plane):
        origin=self.add_cartesian_point(plane[0])
        xaxis=self.add_direction(plane[1])
        yaxis = self.add_direction(plane[2])

        xy=self.add_axis_axis2_placement_3d(origin,xaxis,yaxis)

        return (origin, xaxis,yaxis,xy)

    def add_shape_representation(self, items, context_of_items=None, name:str=''):
        return self.add_entity(p21.entity('SHAPE_REPRESENTATION', (name, items, context_of_items if context_of_items is not None else self.add_context3())))

    def add_shape_representation_relationship(self,		rep_1,rep_2, description:str='',name:str=''):
        return self.add_entity(p21.entity('SHAPE_REPRESENTATION_RELATIONSHIP', (name,description,rep_1,rep_2)))

    def add_cartesian_point(self, pt,name:str=''):
        if not (len(pt)==2 or len(pt)==3):
            raise ValueError(f"{self.__class__.__name__}.add_cartesian_point: cartesian point may be 2d or 3d. {len(pt)}d exist ({pt}).")
        return self.add_entity(p21.Entity(CARTESIAN_POINT, (name,tuple(pt))))
    def add_direction(self, v,name:str=''):
        if not (len(v)==2 or len(v)==3):
            raise ValueError(f"{self.__class__.__name__}.add_direction: direction may be 2d or 3d. {len(v)}d exist ({v}).")
        return self.add_entity(p21.Entity('DIRECTION', (name,tuple(v))))

    def add_surface_style(self, shell_based_surface_model,color=(0.5,0.5,0.5)):
        _492=self.add_manifold_surface_shape_representation((shell_based_surface_model,self.world_plane[-1]),self._context3)
        _493 = self.add_entity(p21.entity("SHAPE_DEFINITION_REPRESENTATION", (self._8, _492)))
        _494 = self.add_entity(p21.entity("PRODUCT_RELATED_PRODUCT_CATEGORY", ('part', p21.UnsetParameter('$'), (self._4,))))
        _501 = self.add_entity(p21.entity("COLOUR_RGB", ('', *color)))
        _502 = self.add_entity(p21.entity("FILL_AREA_STYLE_COLOUR", ('', _501)))
        _500 = self.add_entity(p21.entity("FILL_AREA_STYLE", ('', (_502,))))
        _499 = self.add_entity(p21.entity("SURFACE_STYLE_FILL_AREA", (_500,)))
        _498 = self.add_entity(p21.entity("SURFACE_SIDE_STYLE", ('', (_499,))))
        _497 = self.add_entity(p21.entity("SURFACE_STYLE_USAGE", (p21.Enumeration('.BOTH.'), _498)))
        _496 = self.add_entity(p21.entity("PRESENTATION_STYLE_ASSIGNMENT", ((_497,),)))
        _495 = self.add_entity(p21.entity("STYLED_ITEM", ('', (_496,), shell_based_surface_model)))
        _503 = self.add_entity(
            p21.entity("MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION", ('', (_495,), self._context3)))
        _504 = self.add_entity(p21.entity("PRESENTATION_LAYER_ASSIGNMENT", ('Default', '', (shell_based_surface_model,))))

    def add_entity(self, entity:p21.Entity):
        entity_id = self.next_ref()

        self.step_file.data[0].instances.update({entity_id:p21.SimpleEntityInstance(entity_id, entity)})
        return entity_id

    def add_complex_entity(self, entities:list[p21.Entity]):
        entity_id = self.next_ref()

        self.step_file.data[0].instances.update({entity_id:p21.complex_entity_instance(entity_id, entities)})
        return entity_id

    def add_bspline_surface(self, surf:NURBSSurface,color=(0.5,0.5,0.5), name:str=''):

        unique_knots_u, mult_u = get_knot_multiplicities(surf.knots_u.tolist())
        unique_knots_v, mult_v = get_knot_multiplicities(surf.knots_v.tolist())
        boundaries=extract_surface_boundaries(surf)

        shell= self.add_shell_based_surface_model(
                (self.add_open_shell(
                    (
                        self.add_advanced_face(
                        (
                        self.add_face_bound(
                        self.add_edge_loop([
                            self.add_oriented_edge(
                                self.add_edge_curve(
                                    self.add_cartesian_point(tuple(boundary.start())),
                                    self.add_cartesian_point(tuple(boundary.end())),
                                    self.add_b_spline_curve_with_knots(boundary)
                                )
                            ) for boundary in boundaries])
                        ),
                        ),
                        self.add_entity(
                            p21.entity('B_SPLINE_SURFACE_WITH_KNOTS',(
                                name,
                                int(surf.degree[0]),int(surf.degree[1]),
                                [[self.add_cartesian_point(pt) for pt in pts] for pts in surf.control_points],
                                UNSPECIFIED,FALSE,FALSE,FALSE, mult_u,mult_v,unique_knots_u,unique_knots_v,UNSPECIFIED
                                )
                            )
                        ),
                        FALSE,
                        name
                    ),
                ),),),name)
        self.add_surface_style(shell,color=color)
        return shell
    def add_rational_bspline_surface(self,surf,u_closed:bool=False,v_closed:bool=False,surface_form=UNSPECIFIED,self_intersect:bool=False,name=""):
        unique_knots_u, mult_u = get_knot_multiplicities(surf.knots_u.tolist())
        unique_knots_v, mult_v = get_knot_multiplicities(surf.knots_v.tolist())
        if isinstance(surf,NURBSSurfaceTuple):
            weights = surf.weights
            control_points = surf.control_points
        else:
            cptsw=np.array(surf.control_points_w)
            weights = cptsw[..., -1]
            control_points=cptsw[...,:-1]/weights[...,np.newaxis]

        return self.add_complex_entity(
            [
                p21.entity("BOUNDED_SURFACE", []),
                p21.entity(
                    "B_SPLINE_SURFACE",
                    (
                        int(surf.degree[0]),
                        int(surf.degree[1]),
                        [[self.add_cartesian_point(pt) for pt in pts] for pts in control_points],
                        surface_form,
                        TRUE if u_closed else FALSE,
                        TRUE if v_closed else FALSE,
                        TRUE if self_intersect else FALSE,
                    ),
                ),
                p21.entity("B_SPLINE_SURFACE_WITH_KNOTS", (mult_u, mult_v, unique_knots_u, unique_knots_v, UNSPECIFIED)),
                p21.entity("GEOMETRIC_REPRESENTATION_ITEM", []),
                p21.entity("RATIONAL_B_SPLINE_SURFACE", (weights.tolist(),)),
                p21.entity("REPRESENTATION_ITEM", [name]),
                p21.entity("SURFACE", [])
            ]
        )

    def add_nurbs_surface(self, surf:NURBSSurface|NURBSSurfaceTuple,color=(0.5,0.5,0.5), u_closed: bool = False,
                                 v_closed: bool = False,name:str=''):
        if isinstance(surf,NURBSSurfaceTuple):
            # surf=_tuple_to_nurbs(surf)
            boundaries = extract_surface_boundaries_tuple(surf)
        else:
            boundaries=extract_surface_boundaries(surf)

        shell= self.add_shell_based_surface_model(
                (self.add_open_shell(
                    (self.add_advanced_face(
                        (
                        self.add_face_bound(
                        self.add_edge_loop([
                            self.add_oriented_edge(
                                self.add_edge_curve(
                                    self.add_cartesian_point(tuple(boundary.start())),
                                    self.add_cartesian_point(tuple(boundary.end())),
                                    self.add_rational_b_spline_curve_with_knots(boundary, closed_curve=u_closed if i<2 else v_closed )
                                )
                            ) for i,boundary in enumerate(boundaries)])),
                        ),

                        self.add_rational_bspline_surface(surf,u_closed,v_closed),
                        TRUE,
                        name
                    ),
                ),),),name)
        self.add_surface_style(shell,color=color)
        return shell
    def add_nurbs_curve(self,curve:NURBSCurve):
        oriented_edge=self.add_oriented_edge(
            self.add_edge_curve(
                self.add_cartesian_point(tuple(curve.start())),
                self.add_cartesian_point(tuple(curve.end())),
                self.add_rational_bspline_surface(curve)

            ))
        return oriented_edge

    def add_brep(self, brep, color=(0.5, 0.5, 0.5), name: str = ''):
        """Export a BRep to STEP entities.

        Walks the BRep topology and emits the full STEP entity hierarchy:
        SHELL_BASED_SURFACE_MODEL → OPEN/CLOSED_SHELL → ADVANCED_FACE →
        FACE_BOUND → EDGE_LOOP → ORIENTED_EDGE → EDGE_CURVE.

        Each edge with geometry gets a RATIONAL_B_SPLINE_CURVE_WITH_KNOTS.
        Each face with a surface gets a RATIONAL_B_SPLINE_SURFACE.
        Faces without surface geometry are skipped.

        Parameters
        ----------
        brep : BRep
            The boundary representation to export.
        color : tuple
            RGB color for surface styling.
        name : str
            Name for the STEP shape representation.

        Returns
        -------
        p21.Reference
            Reference to the SHELL_BASED_SURFACE_MODEL entity.
        """
        from mmcore.geom._nurbs_knots import trim_curve

        # --- cache: map BRep entity IDs to STEP references ---
        # Avoids creating duplicate STEP entities for shared topology
        vertex_refs = {}   # brep vertex id → STEP VERTEX_POINT ref
        edge_refs = {}     # brep edge id → STEP EDGE_CURVE ref
        surface_refs = {}  # brep G_SRF id → STEP surface ref

        # --- helper: get or create a STEP vertex ---
        def _vertex(v_id):
            if v_id not in vertex_refs:
                v = brep.V[v_id]
                vertex_refs[v_id] = self.add_vertex_point(
                    self.add_cartesian_point(tuple(v.point))
                )
            return vertex_refs[v_id]

        # --- helper: get or create a STEP surface geometry ---
        def _surface_geom(surf_id):
            if surf_id not in surface_refs:
                srf = brep.G_SRF[surf_id]
                surface_refs[surf_id] = self.add_rational_bspline_surface(srf)
            return surface_refs[surf_id]

        # --- helper: get or create a STEP EDGE_CURVE ---
        def _edge_curve(edge_id):
            if edge_id not in edge_refs:
                edge = brep.E[edge_id]
                v_start_ref = _vertex(edge.v_start)
                v_end_ref = _vertex(edge.v_end)

                if edge.geom is not None:
                    # Trim the curve to the edge's parameter range
                    crv = brep.G_CRV[edge.geom]
                    t0, t1 = edge.param
                    trimmed = trim_curve(crv, min(t0, t1), max(t0, t1))
                    crv_ref = self.add_rational_b_spline_curve_with_knots(trimmed)
                else:
                    # No geometry — create a straight line between vertices
                    p0 = brep.V[edge.v_start].point
                    p1 = brep.V[edge.v_end].point
                    line_crv = NURBSCurveTuple(
                        order=2,
                        knot=np.array([0.0, 0.0, 1.0, 1.0]),
                        control_points=np.array([list(p0), list(p1)], dtype=float),
                        weights=np.array([1.0, 1.0]),
                    )
                    crv_ref = self.add_rational_b_spline_curve_with_knots(line_crv)

                edge_refs[edge_id] = self.add_entity(
                    p21.entity('EDGE_CURVE', (
                        '', v_start_ref, v_end_ref, crv_ref, TRUE
                    ))
                )
            return edge_refs[edge_id]

        # --- build STEP faces ---
        # Collect ALL faces with geometry into a single shell so that
        # importers (Rhino, SolidWorks) treat them as one joined object.
        all_step_faces = []
        for f_id, face in brep.F.items():
            if face.surf is None:
                continue  # skip wire/exterior faces

            # Surface geometry
            srf_ref = _surface_geom(face.surf)

            # Outer loop → FACE_OUTER_BOUND
            bounds = []
            if face.outer is not None:
                outer_edges = []
                for he_id in brep._loop_halfedges(face.outer):
                    he = brep.HE[he_id]
                    ec_ref = _edge_curve(he.edge)
                    oe_ref = self.add_oriented_edge(
                        ec_ref, orientation=he.orient
                    )
                    outer_edges.append(oe_ref)

                if outer_edges:
                    outer_loop_ref = self.add_edge_loop(outer_edges)
                    bounds.append(self.add_entity(
                        p21.entity('FACE_OUTER_BOUND', ('', outer_loop_ref, TRUE))
                    ))

            # Inner loops → FACE_BOUND
            for inner_loop_id in face.inners:
                inner_edges = []
                for he_id in brep._loop_halfedges(inner_loop_id):
                    he = brep.HE[he_id]
                    ec_ref = _edge_curve(he.edge)
                    oe_ref = self.add_oriented_edge(
                        ec_ref, orientation=he.orient
                    )
                    inner_edges.append(oe_ref)
                if inner_edges:
                    inner_loop_ref = self.add_edge_loop(inner_edges)
                    bounds.append(self.add_face_bound(inner_loop_ref))

            # ADVANCED_FACE
            af_ref = self.add_advanced_face(
                bounds, srf_ref, same_sense=TRUE, name=''
            )
            all_step_faces.append(af_ref)

        if not all_step_faces:
            return None

        # Single shell containing all faces
        shell_ref = self.add_open_shell(all_step_faces)

        # SHELL_BASED_SURFACE_MODEL
        model_ref = self.add_shell_based_surface_model((shell_ref,), name)
        self.add_surface_style(model_ref, color=color)
        return model_ref


if __name__ =="__main__":
    from pathlib import Path
    import sys
    sys.path.append(Path(__file__).parent.parent.parent.__str__())

    from mmcore._test_data import ssx

    we = StepWriter()
    ref1 = we.add_nurbs_surface(ssx[1][0], (0.8,0.8,0.8),'surface1')
    ref2 = we.add_nurbs_surface(ssx[1][1], (1.,1.0,0.),'surface2')
    with open('step-test1.step', 'w') as f:
        we.step_file.write(f)
