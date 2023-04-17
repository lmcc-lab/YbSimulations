import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ion_model import PolarVector
import numpy as np
from typing import Union, List, Tuple, Dict, Iterable
from pprint import pprint


class VectorPlot:

    def __init__(self, axis_size=3) -> None:
        
        self.x_axis = PolarVector(axis_size, 0, np.pi/2)
        self.y_axis = PolarVector(axis_size, np.pi/2, np.pi/2)
        self.z_axis = PolarVector(axis_size, 0, 0)

        self.x_axis.translate(np.array([[-axis_size/2, -axis_size/2], [0, 0], [0, 0]]))
        self.y_axis.translate(np.array([[0, 0], [-axis_size/2, -axis_size/2], [0, 0]]))
        self.z_axis.translate(np.array([[0, 0], [0, 0], [-axis_size/2, -axis_size/2]]))

        self.vectors = [(self.x_axis, self.y_axis, self.z_axis), {}]
        self.colors = None
        self.graph_objects = []
        self.array_objects = []
        self.graph_objects_kwargs = []
        self.vector_kwargs = {}

    def add_vector(self, vectors: Dict[str, PolarVector], kwargs: Union[dict, Dict[str, dict]] = {}):
        if list(kwargs.keys()) != list(vectors.keys()):
            kwargs = {key: kwargs for key in vectors}
        self.vectors[1].update(vectors)
        self.vector_kwargs.update(kwargs)
    
    def align_vectors_with_experiment(self, theta=np.pi/2):
        for i, group in enumerate(self.vectors):
            if i == 0:
                for axis_vec in group:
                    axis_vec.rotate_about_x(theta)
            else:
                for vector in group.values():
                    vector.rotate_about_x(theta)
        for i, array in enumerate(self.array_objects):
            self.array_objects[i] = PolarVector.Rx(theta) @ array
    
    def rotate_about_y(self, theta=np.pi/2):
        for i, group in enumerate(self.vectors):
            if i == 0:
                for axis_vec in group:
                    axis_vec.rotate_about_y(theta)
            else:
                for vector in group.values():
                    vector.rotate_about_y(theta)
        for i, array in enumerate(self.array_objects):
            self.array_objects[i] = PolarVector.Ry(theta) @ array
    
    def rotate_about_z(self, theta=np.pi/2):
        for i, group in enumerate(self.vectors):
            if i == 0:
                for axis_vec in group:
                    axis_vec.rotate_about_z(theta)
            else:
                for vector in group.values():
                    vector.rotate_about_z(theta)
        for i, array in enumerate(self.array_objects):
            self.array_objects[i] = PolarVector.Rz(theta) @ array
    
    def add_arc_between_vecs(self, vec1: PolarVector, vec2: PolarVector, **kwargs):
        arc = PolarVector(0.5, vec1.theta, vec1.phi)
        angle = vec1.calculate_angle_between(vec2)
        angle = np.linspace(0, angle)
        x, y, z = np.zeros(len(angle)), np.zeros(len(angle)), np.zeros(len(angle))
        RotMat = vec1.arb_rot_mat(vec2, angle[1])
        for i, a in enumerate(angle):
            arc.rotate_with_rotation_matrix(RotMat)
            x[i], y[i], z[i] = arc.v[:, 1]
        self.add_line(x, y, z, **kwargs)


    def add_line(self, x, y, z, mode='lines', **kwargs):
        kwargs.update({"mode": mode})
        self.array_objects.append(np.array([x, y, z]))
        self.graph_objects_kwargs.append(kwargs)
    
    def reset_plot(self):
        self.graph_objects = []

    def prepare_plot(self):
        # prepare for plotting
        axis_labels = ["x", "y", "z"]
        axis_labels_head = ["x", "y", "z", "u", "v", "w"]
        x_vector = {label: point for label, point in zip(axis_labels, self.vectors[0][0].v)}
        y_vector = {label: point for label, point in zip(axis_labels, self.vectors[0][1].v)}
        z_vector = {label: point for label, point in zip(axis_labels, self.vectors[0][2].v)}

        x_vector["text"] = ["", "x"]
        y_vector["text"] = ["", "y"]
        z_vector["text"] = ["", "z"]
        x_vector["name"] = "x"
        y_vector["name"] = "y"
        z_vector["name"] = "z"

        other_vectors_lines = {name: {"line": {label: coordinates for label, coordinates in zip(axis_labels, list(vector.v))}, 
                                "head": {label: [vector.v[i%3][1], vector.v[i%3][0]] if label not in axis_labels_head[3:] else [vector.v[i%3][1] - vector.v[i%3][0]] for i, label in enumerate(axis_labels_head)}} 
                                for name, vector in self.vectors[1].items()}
        
        # Add x, y, z axes to graph_objects list
        self.graph_objects.append(go.Scatter3d(**x_vector, mode='lines+text', line={"color": "black"}))
        self.graph_objects.append(go.Scatter3d(**y_vector, mode='lines+text', line={"color": "black"}))
        self.graph_objects.append(go.Scatter3d(**z_vector, mode='lines+text', line={"color": "black"}))

        for name, vector in other_vectors_lines.items():
            plotted_line = go.Scatter3d(**vector['line'], mode='lines+text', name=name, **self.vector_kwargs.get(name, {}))
            plotted_head = go.Cone(**vector['head'], sizemode="absolute", sizeref=0.5, showscale=False)
            self.graph_objects.extend([plotted_line, plotted_head])
        
        for kw, array in zip(self.graph_objects_kwargs, self.array_objects):
            kw.update({"x": array[0], "y": array[1], "z": array[2]})
            self.graph_objects.append(go.Scatter3d(**kw))
    
    def show(self, **layout_kwargs):
        layout = go.Layout(**layout_kwargs)

        fig = go.Figure(data=self.graph_objects, layout=layout)
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        return fig
    
    # def plot_animation(self):

class Subplots:

    def __init__(self, **kwargs) -> None:
        self.fig = make_subplots(**kwargs)
    
    def update_subplot(self, row, col, graph_objects, xaxis_kwargs={}, yaxis_kwargs={}, update_scenes_kwargs={}, **kwargs):
        for ob in graph_objects:
            self.fig.add_trace(ob, row=row, col=col, **kwargs)
        self.fig.update_xaxes(row=row, col=col, **xaxis_kwargs)
        self.fig.update_yaxes(row=row, col=col, **yaxis_kwargs)
        self.fig.update_scenes(row=row, col=col, **update_scenes_kwargs)
                
    def layout(self, **kwargs):
        self.fig.update_layout(**kwargs)





    
