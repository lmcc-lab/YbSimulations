import magpylib as magpy
from magpylib.current import Loop
import numpy as np
import matplotlib.pyplot as plt
from ion_model import PolarVector
import plotly.graph_objects as go


def create_coil(num_loops, coil_diameter, loop_spacing, current=1):
    """
    
    coil_diameter: [mm]
    loop_spacing: [mm]
    """
    coil = magpy.Collection(style_label='coil1')
    for z in np.linspace(-loop_spacing * num_loops, loop_spacing * num_loops, num_loops):
        coil.add(Loop(current=current, diameter=coil_diameter, position=(0,0,z)))
    return coil


def add_uniform_field(vector: PolarVector, B_field: np.ndarray):
    vec = vector.v[:, 1].astype(np.float64)
    uf = np.zeros(B_field.shape) + vec
    return B_field + uf


def experimental_setup():
    coilx = create_coil(20, 100, 0.1)
    coily = coilx.copy(position=(0, 300, 0))
    coil_main = create_coil(20, 100, 0.1, current=1.9)
    coil_main.move((300, 0, 0))
    # coilx.copy(position=(-300, 0, 0))

    coilx.move((-300, 0, 0))

    coily.rotate_from_angax(90, 'x', start=0)
    coilx.rotate_from_angax(90, 'y', start=0)
    coil_main.rotate_from_angax(-90, 'y', start=0)

    coilz = create_coil(20, 300, 0.1, current=-1)
    coilz.move((0, 0, -150))

    all_coils = coilx + coily + coil_main + coilz #+ external_field
    return all_coils

def show_coils(coils, **kwargs):
    return magpy.show(*coils, **kwargs)

def show_fields(coils: magpy.Collection, uniform_field_vector: PolarVector = None, 
                spread: float = 100, num_streamtubes=4, num_sensors_per_plane = 15, stream_plot_kwargs = dict(sizeref = 0.3)):

    ts = np.linspace(-400, 400, num_sensors_per_plane)

    grid = np.array([[[(x,y,z) for x in ts] for y in ts] for z in ts])
    X, Y, Z = grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]

    if uniform_field_vector is None:
        uf = np.zeros(grid.shape)
    else:
        vec = uniform_field_vector.v[:, 1].astype(np.float64)
        uf = np.zeros(grid.shape) + vec

    B = magpy.getB(coils, grid) + uf
    
    print(np.argmin(np.abs(grid[:, :, :, 0])), np.argmin(np.abs(grid[:, :, :, 1])), np.argmin(np.abs(grid[:, :, :, 2])))

    Bx, By, Bz = B[:, :, :, 0], B[:, :, :, 1], B[:, :, :, 2]

    grad = np.gradient(B)
    dell = grad[0] + grad[1] + grad[2] + grad[3]

    dellx, delly, dellz = dell[:, :, :, 0], dell[:, :, :, 1], dell[:, :, :, 2]

    dell = dellx + delly + dellz

    minx, miny, minz = np.unravel_index(np.argmin(dell, axis=None), dell.shape)

    x, y, z, Bx, By, Bz = X.flatten(), Y.flatten(), Z.flatten(), Bx.flatten(), By.flatten(), Bz.flatten()

    x, y, z, Bx, By, Bz = x.tolist(), y.tolist(), z.tolist(), Bx.tolist(), By.tolist(), Bz.tolist()

    num_start = num_streamtubes
    start_x = np.linspace(minx-spread, minx+spread, num_start)
    start_y = np.linspace(miny-spread, miny+spread, num_start)
    start_z = np.linspace(minz-spread, minz+spread, num_start)

    SX, SY, SZ = np.meshgrid(start_x, start_y, start_z)
    start_x = SX.flatten()
    start_y = SY.flatten()
    start_z = SZ.flatten()


    data_plot = go.Streamtube(
        x=x, y=y, z=z, u=Bx, v=By, w=Bz,
        starts=dict(
            x = start_x,
            y = start_y,
            z = start_z),
        colorscale = 'jet',
        showscale = False,
        maxdisplayed = 700,
        name="B fields",
        **stream_plot_kwargs
        )

    return data_plot


if __name__ == "__main__":
    # external_field = create_coil(1, 1000, 0.1, current=-20)

    coilx = create_coil(20, 100, 0.1)
    coily = coilx.copy(position=(0, 300, 0))
    coil_main = create_coil(20, 100, 0.1, current=1.9)
    coil_main.move((-300, 0, 0))
    coilx.copy(position=(-300, 0, 0))
    
    coilx.move((300, 0, 0))
    
    coily.rotate_from_angax(90, 'x', start=0)
    coilx.rotate_from_angax(90, 'y', start=0)
    coil_main.rotate_from_angax(-90, 'y', start=0)
    
    coilz = create_coil(20, 300, 0.1)
    coilz.move((0, 0, -150))

    all_coils = coilx + coily + coil_main + coilz #+ external_field

    # helmholtz.rotate_from_angax(90, 'x', start=0)

    # create grid
    ts = np.linspace(-400, 400, 20)
    grid = np.array([[(x,0,z) for x in ts] for z in ts])

    uniform_vector = PolarVector(0.1, np.pi/2, np.pi/4)
    vec = uniform_vector.v[:, 1].astype(np.float64)
    uf = np.zeros(grid.shape) + vec

    # compute and plot field of coil1
    B = magpy.getB(all_coils, grid) + uf
    B *= 50
    B_vec = magpy.getB(all_coils, (0, 0, 0)) * 20000
    Bamp = np.linalg.norm(B, axis=2)
    Bamp /= np.amax(Bamp)
    
    fig = magpy.show(*all_coils, return_fig=True, backend="plotly")

    fig.show()    

    # # ax.quiver(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2], B[:, :, 0], B[:, :, 1], B[:, :, 2], color=Bamp)
    
    # sp = ax.streamplot(
    #     grid[:,:,0], grid[:, :, 1], grid[:,:,2], B[:,:,0], B[:, :, 1], B[:,:,2],
    #     density=2,
    #     color=Bamp,
    #     linewidth=np.sqrt(Bamp)*3,
    #     cmap='coolwarm',
    # )

    # plt.colorbar(sp.lines, ax=ax, label='[mT]')

    
    # # magpy.show(*helmholtz)
    # plt.show()