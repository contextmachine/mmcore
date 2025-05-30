from ._classes import Mesh
import numpy as np
from typing import List, Tuple,NamedTuple
class FuseInfo(NamedTuple):
    position_counts:List[int]
    face_counts:List[int]
def fuse_meshes(meshes: List[Mesh]) -> Tuple[Mesh, FuseInfo]:
    """
    Fuse multiple meshes into a single mesh and return metadata for splitting back.
    Duplicated vertices are preserved; no deduplication is performed.

    Parameters
    ----------
    meshes : List[Mesh]
        List of meshes, each represented as a tuple (vertices, faces), where:
        - vertices: numpy.ndarray of shape (N, 3)
        - faces: numpy.ndarray of shape (M, 3), with integer indices into vertices.

    Returns
    -------
    fused_positions : numpy.ndarray
        Concatenated vertex array of shape (sum Ni, 3).
    fused_faces : numpy.ndarray
        Concatenated face array of shape (sum Mi, 3), with adjusted indices.
    info : FuseInfo
        A tuple containing:
        - position_counts: list of original vertex counts [N1, N2, ...].
        - face_counts: list of original face counts [M1, M2, ...].
    """
    all_position = []
    all_faces = []
    position_counts: List[int] = []
    face_counts: List[int] = []
    offset = 0
    
    for mesh in meshes:
        position, faces =  mesh['position'],  mesh['faces']
        n = position.shape[0]
        m = faces.shape[0]
        # Append vertices and record count
        all_position.append(position)
        position_counts.append(n)
        # Adjust and append faces, record count
        adjusted = faces + offset
        all_faces.append(adjusted)
        face_counts.append(m)
        offset += n

    fused_position = np.vstack(all_position) if all_position else np.zeros((0, 3), dtype=float)
    fused_faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=int)

    return Mesh(position=fused_position, faces=fused_faces), FuseInfo(position_counts, face_counts)

def split_mesh(fused: Mesh, info: FuseInfo) -> List[Mesh]:
    """
    Split a fused mesh back into the original list of meshes using provided metadata.

    Parameters
    ----------
    fused : Mesh
        A tuple (fused_vertices, fused_faces) as returned by fuse_meshes.
    info : FuseInfo
        Tuple containing (vertex_counts, face_counts) for each original mesh.

    Returns
    -------
    List[Mesh]
        Reconstructed list of meshes corresponding to the input order of fuse_meshes.
    """
    fused_positions, fused_faces = fused
    vertex_counts, face_counts = info

    meshes: List[Mesh] = []
    v_start = 0
    f_start = 0

    for v_count, f_count in zip(vertex_counts, face_counts):
        v_end = v_start + v_count
        f_end = f_start + f_count

        # Extract vertices slice
        verts = fused_positions[v_start:v_end]

        # Extract faces slice and rebase indices
        faces_slice = fused_faces[f_start:f_end]
        rebased = faces_slice - v_start

        meshes.append(Mesh(position=verts,faces=rebased))

        # Update offsets
        v_start = v_end
        f_start = f_end

    return meshes
