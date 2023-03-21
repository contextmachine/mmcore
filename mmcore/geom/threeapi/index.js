#!/usr/bin/env node
const THREE = require("three")
const {pyodide} = require("../../../debugview/src/pyodide");
const {PyScriptApp} = require("../../../debugview/src/pyscript");
const mmcore_base = pyodide.pyimport("mmcore.baseitems")
console.log(mmcore_base)
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
