# Mesh Fusing

Unions multiple meshes into a single mesh (a process known as "mesh fusion"). This is especially
useful when preparing large 3D models for rendering in a web browser as it requires only a single
draw call.




User Guide:
-----------
The `union_mesh` function gives an ease to combine multiple 3D meshes into a single 3D mesh object, improving the
rendering process, particularly when viewing 3D models in a web browser.

A brief step-by-step guide on how to utilize this function would go like this:

1. **Prepare the Mesh data**: Accumulate a list of Mesh objects that you wish to combine into a single Mesh object.

```python

import pickle

# Load test data (a collection of 18140 Mesh objects)
with open("tests/data/test_union_gltf_data.pkl", 'rb') as f:
    meshes = pickle.load(f)

```

2. **Call the function**: Use the `union_mesh` function to combine the meshes. This function takes a list of Mesh
   objects and optional dictionary of additional attribute fields.

```python

from mmcore.geom.mesh.union import union_mesh

# Union the list of meshes into a single mesh  
single_mesh = union_mesh(meshes)


```

3. **Result**: The `union_mesh` function returns a single Mesh namedtuple object which includes combined attributes,
   indices and extras from the provided list of input Mesh objects.


4. **Convert to glTF**: Optionally, you may wish to convert the combined mesh into glTF format for further use, for
   instance rendering in a web browser-based 3D engine. You can utilize `create_union_mesh_node` to convert your mesh
   into a node, then `asscene` to create a full glTF Scene from the node.

```python

from mmcore.compat.gltf.convert import create_union_mesh_node, asscene

# Create a GLTF SceneNode from the fused mesh                          
scene_node = create_union_mesh_node(single_mesh, "mesh_node")
scene = asscene(scene_node)  # create GLTF Scene from SceneNode         


```

5. **Save as glTF file**: The resulting glTF Scene object can be serialized into a glTF file using a JSON dump
   operation.

```python

# Dump the scene as a glTF file
with open('single_mesh.gltf', 'w') as f:
    ujson.dump(scene.todict(), f)

This
generates
a
single
glTF
file
'single_mesh.gltf'
of
the
fused
mesh.


```

## Performance Considerations

The main function `union_mesh` under the hood calls `gen_indices_and_extras2` and `_combine_attributes`
functions. The `gen_indices_and_extras2` function traverses through each mesh's attributes resulting in
time complexity of O(m*n), where m is the number of meshes and n is the attribute count of a mesh.
The `_combine_attributes` function concatenates same attribute fields of all meshes, thus it has linear
time complexity of O(n), where n is the count of attribute names.

> Consequently, the overall time complexity of the union operation of meshes is O(m*n),
> where m is the number of meshes and n is the count of attribute names in a mesh.
