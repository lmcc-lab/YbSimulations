import magpylib as magpy
from magpylib.current import Loop
import numpy as np
import matplotlib.pyplot as plt
from ion_model import PolarVector


def create_coil(num_loops, coil_diameter, loop_spacing, current=1):
    """
    
    coil_diameter: [mm]
    loop_spacing: [mm]
    """
    coil = magpy.Collection(style_label='coil1')
    for z in np.linspace(-loop_spacing * num_loops, loop_spacing * num_loops, num_loops):
        coil.add(Loop(current=current, diameter=coil_diameter, position=(0,0,z)))
    return coil


def create_uniform_field(vector: np.ndarray, grid: np.ndarray):
    return grid + vector



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