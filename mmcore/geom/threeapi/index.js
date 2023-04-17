#!/usr/bin/env node
const THREE = require("three")
const pyodide = require("pyodide");
const {RhinoLoader} = require("three/addons/loaders/3DMLoader.js");
const rootObj=new THREE.Object3D()
pyodide.loadPyodide().then(function (pyod) {
    const mmcore_base= pyod.runPython(
        "type('Foo',(object,),{'__get__':lambda self, inst,own: inst.__dict__[self.name]})\n")
    mmcore_base.name = "aa"
    console.log(mmcore_base)
} )
const rhinoLoader = new RhinoLoader.Rhino3dmLoader()
function load_3dm(fname){
    rhinoLoader.load(fname, function (obj) {rootObj.attach(obj)

    },function (xhr) {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        function (err) {
            console.error('An error happened');
        })
}
function threeThing (){
    ThreeMFLoader.
    THREE.ExtrudeBufferGeometry()
}
const simple = new THREE.Object3D()
function three_json_dump(simple) {
    return JSON.stringify(simple.toJSON())

}

function load_from_file(fname) {
    loader.load(fname, function (obj) {
            // Add the loaded object to the scene
            return obj;
        },
        function (xhr) {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        function (err) {
            console.error('An error happened');
        }
    )

}
