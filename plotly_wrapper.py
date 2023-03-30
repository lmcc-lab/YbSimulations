import plotly.graph_objs as go
from ion_model import PolarVector
import numpy as np
from typing import Union, List, Tuple


class VectorPlot:

    def __init__(self) -> None:
        
        self.x_axis = PolarVector(3, 0, np.pi/2)
        self.y_axis = PolarVector(3, np.pi/2, np.pi/2)
        self.z_axis = PolarVector(3, 0, 0)

        self.x_axis.translate(np.array([[-1.5, -1.5], [0, 0], [0, 0]]))
        self.y_axis.translate(np.array([[0, 0], [-1.5, -1.5], [0, 0]]))
        self.z_axis.translate(np.array([[0, 0], [0, 0], [-1.5, -1.5]]))

        self.vectors = [(self.x_axis, self.y_axis, self.z_axis)]

    def add_vector(self, vectors: Union[PolarVector, List[PolarVector], Tuple[PolarVector]]):

        if isinstance(vectors, tuple) or isinstance(vectors, list):
            self.vectors.extend(vectors)
        else:
            self.vectors.append(vectors)
    
    def align_vectors_with_experiment(self, theta=np.pi/2):
        for i, vector in enumerate(self.vectors):
            if i == 0:
                for axis_vec in vector:
                    axis_vec.rotate_about_x(theta)
            else:
                vector.rotate_about_x(theta)
        
    def plot(self):
        # prepare for plotting
        axis_labels = ["x", "y", "z"]
        axis_labels_head = ["x", "y", "z", "u", "v", "w"]
        x_vector = {label: point for label, point in zip(axis_labels, self.vectors[0][0].v)}
        y_vector = {label: point for label, point in zip(axis_labels, self.vectors[0][1].v)}
        z_vector = {label: point for label, point in zip(axis_labels, self.vectors[0][2].v)}

        x_vector["text"] = ["", "x"]
        y_vector["text"] = ["", "y"]
        z_vector["text"] = ["", "z"]

        other_vectors_lines = [{"line": {label: coordinates for label, coordinates in zip(axis_labels, vector.v)}, 
                                "head": {label: [vector.v[i%3][1]] for i, label in enumerate(axis_labels_head)}} 
                                for vector in self.vectors[1:]]
        
        x_axis = go.Scatter3d(**x_vector, mode='lines+text', line={"color": "black"})
        y_axis = go.Scatter3d(**y_vector, mode='lines+text', line={"color": "black"})
        z_axis = go.Scatter3d(**z_vector, mode='lines+text', line={"color": "black"})

        # create the line body
        vector_plot = [x_axis, y_axis, z_axis]
        for vector in other_vectors_lines:
            plotted_line = go.Scatter3d(**vector['line'], mode='lines', name='Vector')
            plotted_head = go.Cone(**vector['head'], sizemode="absolute", sizeref=0.5, name='Arrow')
            vector_plot.extend([plotted_line, plotted_head])
        
        layout = go.Layout(title='3D Vector with Cone Arrow')

        fig = go.Figure(data=vector_plot, layout=layout)
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

        return fig
        
