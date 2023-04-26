from cmath import cos

from mmcore.geom.vectors import Vector


def computeTangents(self):

		index = self.index
		attributes = self.attributes

		# based on http://www.terathon.com/code/tangent.html
		# (per vertex tangents)

		if index is None or \
			attributes.position is None or \
			attributes.normal is None or \
			attributes.uv is None:

			print('THREE.BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)')
			return

		indices = index.array
		positions = attributes.position.array
		normals = attributes.normal.array
		uvs = attributes.uv.array

		nVertices = positions.length / 3

		if self.hasAttribute('tangent') is False:

			self.setAttribute('tangent', BufferAttribute(Float32Array(4 * nVertices), 4))

		tangents = self.getAttribute('tangent').array

		tan1 = []
		tan2 = []

		for i in range(nVertices):

			tan1.append(Vector3())
			tan2.append(Vector3())

		vA = Vector3()
		vB = Vector3()
		vC = Vector3()

		uvA = Vector2()
		uvB = Vector2()
		uvC = Vector2()

		sdir = Vector3()
		tdir = Vector3()

		def handleTriangle(a, b, c):

			vA.fromArray(positions, a * 3)
			vB.fromArray(positions, b * 3)
			vC.fromArray(positions, c * 3)

			uvA.fromArray(uvs, a * 2)
			uvB.fromArray(uvs, b * 2)
			uvC.fromArray(uvs, c * 2)

			vB.sub(vA)
			vC.sub(vA)

			uvB.sub(uvA)
			uvC.sub(uvA)

			r = 1.0 / (uvB.x * uvC.y - uvC.x * uvB.y)

			# silently ignore degenerate uv triangles having coincident or colinear vertices

			if not isFinite(r):
				return

			sdir.copy(vB).multiplyScalar(uvC.y).addScaledVector(vC, -uvB.y).multiplyScalar(r)
			tdir.copy(vC).multiplyScalar(uvB.x).addScaledVector(vB, -uvC.x).multiplyScalar(r)

			tan1[a].add(sdir)
			tan1[b].add(sdir)
			tan1[c].add(sdir)

			tan2[a].add(tdir)
			tan2[b].add(tdir)
			tan2[c].add(tdir)

		groups = self.groups

		if len(groups) == 0:

			groups = [{
				'start': 0,
				'count': len(indices)
			}]

		for i in range(len(groups)):

			group = groups[i]

			start = group['start']
			count = group['count']

			for j in range(start, start + count, 3):

				handleTriangle(
					indices[j + 0],
					indices[j + 1],
					indices[j + 2]
				)

		tmp = Vector()
		tmp2 = Vector()
		n = Vector()
		n2 = Vector()

		def handleVertex(v):

			n.fromArray(normals, v * 3)
			n2.copy(n)

			t = tan1[v]

			# Gram-Schmidt orthogonalize

			tmp.copy(t)
			tmp.sub(n.multiplyScalar(n.dot(t))).normalize()

			# Calculate handedness

			tmp2.crossVectors(n2, t)
			test = tmp2.dot(tan2[v])
			w = (test < 0.0) ? -1.0 : 1.0

			tangents[v * 4] = tmp.x
			tangents[v * 4 + 1] = tmp.y
			tangents[v * 4 + 2] = tmp.z
			tangents[v * 4 + 3] = w

		for i in range(len(groups)):

			group = groups[i]

			start = group['start']
			count = group['count']

			for j in range(start, start + count, 3):

				handleVertex(indices[j + 0])
				handleVertex(indices[j + 1])
				handleVertex(indices[j + 2])

class EdgesGeometry(BufferGeometry):

    def __init__(self, geometry = None, thresholdAngle = 1):

        super().__init__()

        self.type = 'EdgesGeometry'

        self.parameters = {
            'geometry': geometry,
            'thresholdAngle': thresholdAngle
        }

        if geometry is not None:

            precisionPoints = 4
            precision = pow(10, precisionPoints)
            thresholdDot = cos(mathutils.DEG2RAD * thresholdAngle)

            indexAttr = geometry.getIndex()
            positionAttr = geometry.getAttribute('position')
            indexCount = indexAttr.count if indexAttr else positionAttr.count

            indexArr = [0, 0, 0]
            vertKeys = ['a', 'b', 'c']
            hashes = [None] * 3

            edgeData = {}
            vertices = []
            for i in range(0, indexCount, 3):

                if indexAttr:

                    indexArr[0] = indexAttr.getX(i)
                    indexArr[1] = indexAttr.getX(i + 1)
                    indexArr[2] = indexAttr.getX(i + 2)

                else:

                    indexArr[0] = i
                    indexArr[1] = i + 1
                    indexArr[2] = i + 2

                a, b, c = _triangle
                a.fromBufferAttribute(positionAttr, indexArr[0])
                b.fromBufferAttribute(positionAttr, indexArr[1])
                c.fromBufferAttribute(positionAttr, indexArr[2])
                _triangle.getNormal(_normal)

                # create hashes for the edge from the vertices
                hashes[0] = f'{round(a.x * precision)},{round(a.y * precision)},{round(a.z * precision)}'
                hashes[1] = f'{round(b.x * precision)},{round(b.y * precision)},{round(b.z * precision)}'
                hashes[2] = f'{round(c.x * precision)},{round(c.y * precision)},{round(c.z * precision)}'

                # skip degenerate triangles
                if hashes[0] == hashes[1] or hashes[1] == hashes[2] or hashes[2] == hashes[0]:

                    continue

                # iterate over every edge
                for j in range(0, 3):

                    # get the first and next vertex making up the edge
                    jNext = (j + 1) % 3
                    vecHash0 = hashes[j]
                    vecHash1 = hashes[jNext]
                    v0 = _triangle[vertKeys[j]]
                    v1 = _triangle[vertKeys[jNext]]

                    hash = f'{vecHash0}_{vecHash1}'
                    reverseHash = f'{vecHash1}_{vecHash0}'

                    if reverseHash in edgeData and edgeData[reverseHash]:

                        # if we found a sibling edge add it into the vertex array if
                        # it meets the angle threshold and delete the edge from the map.
                        if _normal.dot(edgeData[reverseHash].normal) <= thresholdDot:

                            vertices.append(v0.x)
                            vertices.append(v0.y)
                            vertices.append(v0.z)
                            vertices.append(v1.x)
                            vertices.append(v1.y)
                            vertices.append(v1.z)

                        edgeData[reverseHash] = None

                    elif hash not in edgeData:

                        # if we've already got an edge here then skip adding a new one
                        edgeData[hash] = {

                            'index0': indexArr[j],
                            'index1': indexArr[jNext],
                            'normal': _normal.clone(),

                        }

            # iterate over all remaining, unmatched edges and add them to the vertex array
            for key in edgeData:

                if edgeData[key]:

                    index0 = edgeData[key]['index0']
                    index1 = edgeData[key]['index1']
                    _v0.fromBufferAttribute(positionAttr, index0)
                    _v1.fromBufferAttribute(positionAttr, index1)

                    vertices.append(_v0.x)
                    vertices.append(_v0.y)
                    vertices.append(_v0.z)
                    vertices.append(_v1.x)
                    vertices.append(_v1.y)
                    vertices.append(_v1.z)

            self.setAttribute('position', Float32BufferAttribute(vertices, 3))

    def copy(self, source):

        super().copy(source)

        self.parameters = dict(source.parameters)

        return self

export(EdgesGeometry)