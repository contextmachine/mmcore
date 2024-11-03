
import sys
import warnings

import numpy as np

from plotly import graph_objs
from dataclasses import dataclass


from mmcore.geom.curves.curve import Curve


class RenderConfigOverrides(TypedDict):
    figure_image_filename:str
    render_to_file:bool
    display_evalpts:bool
    display_trims:bool
    display_legend:bool
    display_axes:bool
    axes_equal:bool
    display_ctrlpts: bool
    display_bbox: bool
    figure_size:tuple[int,int]
    trim_size:int
    line_width:int
    dtype:type

@dataclass
class RenderColorsConfig:

    evalpts: str="black"
    trims:  str= "black"
    ctrlpts:  str= "black"
    display_bbox:  str= "aqua"



@dataclass
class RenderConfig:
    """
    """

    display_evalpts:bool=True
    display_trims:bool=True
    display_legend:bool=False
    display_axes:bool=False
    axes_equal:bool=True
    display_ctrlpts: bool = True
    display_bbox: bool = False
    figure_size:tuple[int,int]=(720, 720)
    trim_size:int=1
    line_width:int=2
    ctrlpts_offset:float=2.
    dtype:type=np.float64
    plot_bgcolor='white'

    



    def in_notebook(self):
        return 'ipykernel' in sys.modules

class BaseRenderer:
    def __init__(self, config:RenderConfig=None, **kwargs:RenderConfigOverrides):
        self._plots = []
        self.config=config if config is not None else RenderConfig()
        self.config.__dict__.update(kwargs)
    def render(self, **kwargs:RenderConfigOverrides)->graph_objs.Figure:
        raise NotImplemented()
    def update_config(self, **kwargs:RenderConfigOverrides):
        self.config.__dict__.update(kwargs)

    def add(self, ptsarr, plot_type, name="", color="black", idx=0,**kwargs):
        """ Adds points sets to the visualization instance for plotting.

        :param idx:
        :param density:
        :param ptsarr: control or evaluated points
        :type ptsarr: list, tuple
        :param plot_type: type of the plot, e.g. ctrlpts, evalpts, bbox, etc.
        :type plot_type: str
        :param name: name of the plot displayed on the legend
        :type name: str
        :param color: plot color
        :type color: str
        :param color: plot index
        :type color: int
        """
        # ptsarr can be a list, a tuple or an array
        if ptsarr is None or len(ptsarr) == 0:
            return
        # Add points, size, plot color and name on the legend
        plt_name = " ".join([str(n) for n in name]) if isinstance(name, (list, tuple)) else name
        elem = {'ptsarr': ptsarr, 'name': plt_name, 'color': color, 'type': plot_type, 'idx': idx,**kwargs}
        self._plots.append(elem)
    def clear(self):
        self._plots=[]
    def _prepare_color_config(self, color_config):
        if color_config is None:
            color_config = RenderColorsConfig()
        elif isinstance(color_config, dict):
            color_config = RenderColorsConfig(**color_config)
        elif isinstance(color_config, RenderColorsConfig):
            color_config = color_config
        else:
            color_config = RenderColorsConfig()
            warnings.warn('color config ignored')
        return color_config
    def setup(self, obj, color_config=None,idx=0, density=100):

        color_config=self._prepare_color_config(color_config)
        if hasattr(obj,'control_points') and self.config.display_ctrlpts:
            self.add(ptsarr=obj.control_points, name="", color=color_config.ctrlpts, plot_type='ctrlpts',idx=idx)
        if self.config.display_evalpts:
            self.add(ptsarr=obj.points(density), name="", color=color_config.evalpts, plot_type='evalpts',idx=idx)

        # Data requested by the visualization module
    def __call__(self, objs,color_config=None, density=100,*args, **kwargs):
        self.update_config(**kwargs)
        for i,obj in enumerate(objs):

            self.setup(obj, color_config=color_config,idx=i, density=100)

        return self.render(**kwargs)
    def add_marker(self, pts, color="blue",size=4,idx=None):
        self.add(ptsarr=pts, name="", color=color, plot_type='marker',idx=idx if idx is not None else len(self._plots),size=size)

    def add_curve(self, crv, color='black', idx=None):

        if hasattr(crv, 'control_points') and self.config.display_ctrlpts:
            self.add(ptsarr=crv.control_points, name="", color=color, plot_type='ctrlpts', idx=idx if idx is not None else len(self._plots))
        if self.config.display_evalpts:
            self.add(ptsarr=crv.points(), name="", color=color, plot_type='ctrlpts', idx=idx if idx is not None else len(self._plots))


class Renderer2D(BaseRenderer):
    """ Plotly visualization module for 2D curves. """
    def __init__(self, config=RenderConfig(),**kwargs):
        super().__init__(config, **kwargs)

    def setup(self, obj, color_config=None,idx=0, density=100):
        s,e=obj.interval()

        density*= int(e-s)
        color_config = self._prepare_color_config(color_config)
        if hasattr(obj, 'control_points') and self.config.display_ctrlpts:
            self.add(ptsarr=obj.control_points, name="", color=color_config.ctrlpts, plot_type='ctrlpts', idx=idx)
        if self.config.display_evalpts:
            self.add(ptsarr=obj.points(density), name="", color=color_config.evalpts, plot_type='evalpts', idx=idx)

    def render(self, **kwargs:RenderConfigOverrides):
        """ Plots the curve and the control points polygon. """
        self.update_config(**kwargs)
       

        # Initialize variables



        # Generate the figure
        fig = graph_objs.Figure( )


        for plot in self._plots[::-1]:
            pts = np.array(plot['ptsarr'],dtype=self.config.dtype)
            if plot['type'] == 'marker':
                fig.add_trace(graph_objs.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    name=plot['name'],
                    mode='markers',
                    marker=dict(
                        color=plot['color'],
                        size=plot['size'],
                    )
                ))

            # Plot control points
            if plot['type'] == 'ctrlpts' :
                fig.add_trace( graph_objs.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    name=plot['name'],
                    mode='lines+markers',
                    line=dict(
                        color=plot['color'],
                        width=int(self.config.line_width/2),
                        dash='dash'
                    )
                ))


            # Plot evaluated points
            if plot['type'] == 'evalpts' :
                fig.add_trace( graph_objs.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    name=plot['name'],
                    mode='lines',
                    line=dict(
                        color=plot['color'],
                        width=self.config.line_width
                    )
                ))


            # Plot bounding box
            if plot['type'] == 'bbox':
                fig.add_trace(graph_objs.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    name=plot['name'],
                    line=dict(
                        color=plot['color'],
                        width=self.config.line_width,
                        dash='dashdot',
                    )
                ))


            # Plot extras
            if plot['type'] == 'extras':
                fig.add_trace(graph_objs.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    name=plot['name'],
                    mode='markers',
                    marker=dict(
                        color=plot['color'][0],
                        size=plot['color'][1],
                        line=dict(
                            width=self.config.line_width
                        )
                    )
                ))

        plot_layout = dict(
            width=self.config.figure_size[0],
            height=self.config.figure_size[1],
            autosize=False,
            showlegend=self.config.display_legend,
            yaxis=dict(
                scaleanchor="x",
                showgrid=self.config.display_axes,
                showline=self.config.display_axes,
                zeroline=self.config.display_axes,
                showticklabels=self.config.display_axes,
            ),
            xaxis=dict(
                showgrid=self.config.display_axes,
                showline=self.config.display_axes,
                zeroline=self.config.display_axes,
                showticklabels=self.config.display_axes,
            ),
            plot_bgcolor=self.config.plot_bgcolor
        )

        fig.update_layout(plot_layout)
        return fig



class Curve3DRenderer(BaseRenderer):
    """ Plotly visualization module for 3D curves. """
    def __init__(self, config: RenderConfig=None, **kwargs):
        super().__init__(config, **kwargs)

    def render(self, **kwargs:RenderConfigOverrides):
        """ Plots the curve and the control points polygon. """

        self.update_config(**kwargs)



        fig = graph_objs.Figure()
        # Initialize variables


        for plot in self._plots:
            pts = np.array(plot['ptsarr'], dtype=self.config.dtype)

            # Try not to fail if the input is 2D
            if pts.shape[1] == 2:
                pts = np.c_[pts, np.zeros(pts.shape[0])]

            # Plot control points
            if plot['type'] == 'ctrlpts':
                fig.add_trace( graph_objs.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    name=plot['name'],
                    mode='lines+markers',
                    line=dict(
                        color=plot['color'],
                        width=self.config.line_width,
                        dash='dash'
                    ),
                    marker=dict(
                        color=plot['color'],
                        size=self.config.line_width * 2,
                    )
                ))


            # Plot evaluated points
            if plot['type'] == 'evalpts':
                fig.add_trace( graph_objs.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    name=plot['name'],
                    mode='lines',
                    line=dict(
                        color=plot['color'],
                        width=self.config.line_width
                    ),
                ))


            # Plot bounding box
            if plot['type'] == 'bbox' :
                fig.add_trace(
                    graph_objs.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    name=plot['name'],
                    mode='lines',
                    line=dict(
                        color=plot['color'],
                        width=self.config.line_width,
                        dash='dashdot',
                    ),
                ))


            # Plot extras
            if plot['type'] == 'extras':
                fig.add_trace(
                graph_objs.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    name=plot['name'],
                    mode='markers',
                    marker=dict(
                        color=plot['color'][0],
                        size=plot['color'][1],
                        line=dict(
                            width=self.config.line_width
                        )
                    )
                ))

        plot_layout = dict(
            width=self.config.figure_size[0],
            height=self.config.figure_size[1],
            autosize=False,
            showlegend=self.config.display_legend,
            scene=dict(
                xaxis=dict(
                    showgrid=self.config.display_axes,
                    showline=self.config.display_axes,
                    zeroline=self.config.display_axes,
                    showticklabels=self.config.display_axes,
                    title='',
                ),
                yaxis=dict(
                    showgrid=self.config.display_axes,
                    showline=self.config.display_axes,
                    zeroline=self.config.display_axes,
                    showticklabels=self.config.display_axes,
                    title='',
                ),
                zaxis=dict(
                    showgrid=self.config.display_axes,
                    showline=self.config.display_axes,
                    zeroline=self.config.display_axes,
                    showticklabels=self.config.display_axes,
                    title='',
                ),
            ),
        )
        if self.config.axes_equal:
            plot_layout['scene']['aspectmode'] = 'data'
        # Set aspect ratio

        # Generate the figure
        fig.update_layout(plot_layout)

        return fig


class SurfaceRenderer(BaseRenderer):
    """ Plotly visualization module for surfaces.

    Triangular mesh plot for the surface and wireframe plot for the control points grid.
    """
    def __init__(self, config: RenderConfig=None, **kwargs):
        super().__init__(config, **kwargs)


    def render(self, **kwargs):
        """ Plots the surface and the control points grid. """
        # Calling parent function
        self.update_config(**kwargs)

        # Initialize variables
        plot_data = []

        for plot in self._plots:
            # Plot control points
            if plot['type'] == 'ctrlpts' and self.config.display_ctrlpts:
                pts = np.array(plot['ptsarr'], dtype=self.config.dtype)
                pts[:, 2] += self.config.ctrlpts_offset
                figure = graph_objs.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    name=plot['name'],
                    mode='markers',
                    marker=dict(
                        color=plot['color'],
                        size=self.config.line_width * 2,
                    )
                )
                plot_data.append(figure)

            # Plot evaluated points
            if plot['type'] == 'evalpts' and self.config.display_evalpts:
                vertices = plot['ptsarr'][0]
                triangles = plot['ptsarr'][1]
                pts = [v.data for v in vertices]
                tri = [t.data for t in triangles]
                pts = np.array(pts, dtype=self.config.dtype)
                tri = np.array(tri, dtype=self.config.dtype)
                figure = graph_objs.Mesh3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    name=plot['name'],
                    i=tri[:, 0],
                    j=tri[:, 1],
                    k=tri[:, 2],
                    color=plot['color'],
                    opacity=0.75,
                )
                plot_data.append(figure)

            # Plot bounding box
            if plot['type'] == 'bbox' and self.config.display_bbox:
                pts = np.array(plot['ptsarr'], dtype=self.config.dtype)
                figure = graph_objs.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    name=plot['name'],
                    mode='lines',
                    line=dict(
                        color=plot['color'],
                        width=self.config.line_width,
                        dash='dashdot',
                    ),
                )
                plot_data.append(figure)

            # Plot trim curves
            if self.config.display_trims:
                if plot['type'] == 'trimcurve':
                    pts = np.array(plot['ptsarr'], dtype=self.config.dtype)
                    figure = graph_objs.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        name=plot['name'],
                        mode='markers',
                        marker=dict(
                            color=plot['color'],
                            size=self.config.trim_size * 2,
                        ),
                    )
                    plot_data.append(figure)

            # Plot extras
            if plot['type'] == 'extras':
                pts = np.array(plot['ptsarr'], dtype=self.config.dtype)
                figure = graph_objs.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    name=plot['name'],
                    mode='markers',
                    marker=dict(
                        color=plot['color'][0],
                        size=plot['color'][1],
                        line=dict(
                            width=self.config.line_width
                        )
                    )
                )
                plot_data.append(figure)

        plot_layout = dict(
            width=self.config.figure_size[0],
            height=self.config.figure_size[1],
            autosize=False,
            showlegend=self.config.display_legend,
            scene=dict(
                xaxis=dict(
                    showgrid=self.config.display_axes,
                    showline=self.config.display_axes,
                    zeroline=self.config.display_axes,
                    showticklabels=self.config.display_axes,
                    title='',
                ),
                yaxis=dict(
                    showgrid=self.config.display_axes,
                    showline=self.config.display_axes,
                    zeroline=self.config.display_axes,
                    showticklabels=self.config.display_axes,
                    title='',
                ),
                zaxis=dict(
                    showgrid=self.config.display_axes,
                    showline=self.config.display_axes,
                    zeroline=self.config.display_axes,
                    showticklabels=self.config.display_axes,
                    title='',
                ),
            ),
        )

        # Set aspect ratio
        if self.config.axes_equal:
            plot_layout['scene']['aspectmode'] = 'data'

        # Generate the figure
        fig = graph_objs.Figure(data=plot_data, layout=plot_layout)


        return fig
