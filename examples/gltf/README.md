[source](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#schema-reference-accessor)



<html>
<colgroup>
<col>
<col>
<col>
<col>
</colgroup>
<thead>
<tr>
<th>Name</th>
<th>Accessor Type(s)</th>
<th>Component Type(s)</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><p><code>POSITION</code></p></td>
<td><p>VEC3</p></td>
<td><p><em>float</em></p></td>
<td><p>Unitless XYZ vertex positions</p></td>
</tr>
<tr>
<td><p><code>NORMAL</code></p></td>
<td><p>VEC3</p></td>
<td><p><em>float</em></p></td>
<td><p>Normalized XYZ vertex normals</p></td>
</tr>
<tr>
<td><p><code>TANGENT</code></p></td>
<td><p>VEC4</p></td>
<td><p><em>float</em></p></td>
<td><p>XYZW vertex tangents where the XYZ portion is normalized, and the W component is a sign value (-1 or +1) indicating handedness of the tangent basis</p></td>
</tr>
<tr>
<td><p><code>TEXCOORD_n</code></p></td>
<td><p>VEC2</p></td>
<td><p><em>float</em><br>
                                  <em>unsigned byte</em> normalized<br>
                                  <em>unsigned short</em> normalized</p></td>
<td><p>ST texture coordinates</p></td>
</tr>
<tr>
<td><p><code>COLOR_n</code></p></td>
<td><p>VEC3<br>
                VEC4</p></td>
<td><p><em>float</em><br>
                                  <em>unsigned byte</em> normalized<br>
                                  <em>unsigned short</em> normalized</p></td>
<td><p>RGB or RGBA vertex color linear multiplier</p></td>
</tr>
<tr>
<td><p><code>JOINTS_n</code></p></td>
<td><p>VEC4</p></td>
<td><p><em>unsigned byte</em>
                                  <em>unsigned short</em></p></td>
<td><p>See <a href="#skinned-mesh-attributes">Skinned Mesh Attributes</a></p></td>
</tr>
<tr>
<td><p><code>WEIGHTS_n</code></p></td>
<td><p>VEC4</p></td>
<td><p><em>float</em><br>
                                  <em>unsigned byte</em> normalized<br>
                                  <em>unsigned short</em> normalized</p></td>
<td><p>See <a href="#skinned-mesh-attributes">Skinned Mesh Attributes</a></p></td>
</tr>
</tbody>
</html>