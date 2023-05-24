const THREE = require("three")

function makeExtrusion(coords, holes, line_coords, color) {
    let pts = [];
    const material1 = new THREE.MeshLambertMaterial({color: color, wireframe: false});
    const spline = new THREE.LineCurve3(
        new THREE.Vector3(line_coords[0][0],
                          line_coords[0][1],
                          line_coords[0][2]),
        new THREE.Vector3(line_coords[1][0],
                          line_coords[1][1],
                          line_coords[1][2]
        )
    );
    coords.forEach((value, index, array) => {
        pts.push(new THREE.Vector2(value[0], value[1]))
    })
    const extrudeSettings1 = {
        steps: 1,
        depth: 1.0,
        bevelEnabled: false,
        extrudePath: spline
    };
    let shape1 = new THREE.Shape(pts);

    let pttt = [];
    holes.forEach((value1, index1, array1) => {

        let ptt = []
        value1.forEach((value2, index2, array2) => {
            ptt.push(new THREE.Vector2(value2[0], value2[1]))
        });
        pttt.push(new THREE.Shape(ptt));

    })
    pttt.forEach((value) => {
        shape1.holes.push(value)
    })
    const geometry1 = new THREE.ExtrudeGeometry(shape1, extrudeSettings1);
    return new THREE.Mesh(geometry1, material1);
}

function makeMany(cords, holes, axis, color) {
    let grp = new THREE.Group();
    axis.forEach((value) => {
        grp.add(makeExtrusion(cords, holes, value, color))
    })
    console.log(JSON.stringify(grp.toJSON()))
}
