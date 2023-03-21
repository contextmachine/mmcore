#!/usr/bin/env node
const THREE = require("three")
const {pyodide} = require("pyodide");
const {Object3D, ObjectLoader} = require("three");

const scene = new THREE.Scene()

function three_json_dump() {
    const simple = new THREE.Object3D()
    var obj = simple.toJSON()
    return JSON.stringify(obj)

}

const loader = new THREE.ObjectLoader();

function load_from_file(fname) {
    loader.load(fname, function (obj) {
            // Add the loaded object to the scene
            scene.add(obj)
            console.log(obj)
        },
        function (xhr) {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        function (err) {
            console.error('An error happened');
        }
    )

}

const res = three_json_dump()
console.log(res)

async function main() {
    await load_from_file("http://192.168.205.1:8086/data/BufferGeometryObject.json");
}

c