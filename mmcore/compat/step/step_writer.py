from __future__ import annotations

from collections import OrderedDict

from typing import Any
from mmcore import __version__
from steputils import p21
from itertools import count
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface

from mmcore.numeric.intersection.ssx.boundary_intersection import extract_surface_boundaries



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
        if abs(knots[i] - last_knot) < 1e-8:
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
    def __init__(self, step_file:p21.StepFile=None, tolerance=1e-7,world_plane= ((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.))):
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
        #self.base_shape_representation=self.add_shape_representation((self.world_plane[-2],self.world_plane[-1]), self._context3,'Document')

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
    def issteptype(self, obj):
        return type(obj) in (p21.Reference,p21.SimpleEntityInstance,p21.ComplexEntityInstance)


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

    def add_nurbs_surface(self, surf:NURBSSurface,color=(0.5,0.5,0.5), name:str=''):

        unique_knots_u, mult_u = get_knot_multiplicities(surf.knots_u.tolist())
        unique_knots_v, mult_v = get_knot_multiplicities(surf.knots_v.tolist())
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
                                    self.add_b_spline_curve_with_knots(boundary)
                                )
                            ) for boundary in boundaries])),
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
                        TRUE,
                        name
                    ),
                ),),),name)
        self.add_surface_style(shell,color=color)
        return shell
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

