import * as THREE from 'three';
THREE.Object3D.DEFAULT_UP=new THREE.Vector3(0,0,1)
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {Mesh} from "three";

import {Viewer,ViewerProps} from "./src/Viewer";
import {getValueType} from "three/addons/nodes/core/NodeUtils";
/*
const renderer = new THREE.WebGLRenderer( { antialias: true, logarithmicDepthBuffer: true } );
renderer.shadowMap.enabled = true;
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( window.innerWidth, window.innerHeight );
renderer.localClippingEnabled = true;
renderer.setClearColor( `#919191` );
*/
const viewer=new Viewer()
const renderer= viewer.initRenderer()
var response=await fetch("http://localhost:7711/cpts/clear",{ method:"GET", headers:{"Content-Type": "application/json"}} )


window.addEventListener( 'resize', onWindowResize );
document.body.appendChild( renderer.domElement );


const aspect=(window.innerHeight/window.innerWidth)

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera( -1/2,  1/2,aspect/2,-aspect/2);

scene.add(camera)
camera.position.set( 2, 2, 2 );
camera.rotateX(-20)
camera.rotateY(20)
camera.rotateZ(-220)
//const light=THREE.DirectionalLight(`#e4e4e4`, 1.)


scene.add( new THREE.AmbientLight( 0xffffff, 0.5 ) );

const dirLight = new THREE.DirectionalLight( 0xffffff, 3 );
dirLight.position.set( 5, 10, 7.5 );
dirLight.castShadow = true;
dirLight.shadow.camera.right = 2;
dirLight.shadow.camera.left = - 2;
dirLight.shadow.camera.top	= 2;
dirLight.shadow.camera.bottom = - 2;

dirLight.shadow.mapSize.width = 1024;
dirLight.shadow.mapSize.height = 1024;
scene.add( dirLight );


const controls = new OrbitControls( camera, renderer.domElement );

controls.zoomToCursor=true
controls.maxPolarAngle=Infinity
controls.update();


const material = new THREE.MeshBasicMaterial( { color: `#ffffff` } );
const lmaterial = new THREE.LineBasicMaterial( { color: `#ff4747` } );
const bb=new THREE.BoxGeometry(1,1,1)
const le =new THREE.LineSegments(bb, material)

//scene.add(le);


function setupIntersectionHandler(renderer, camera, scene) {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    renderer.domElement.addEventListener('mousedown', async event => {
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);


        const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1));
        const intersectionPoint = new THREE.Vector3();

        const intersectsPlane = raycaster.ray.intersectPlane(plane, intersectionPoint);

        if (intersectsPlane) {
            var response=await fetch("http://localhost:7711/cpts",{body:JSON.stringify({data:intersectionPoint}), method:"POST", headers:{"Content-Type": "application/json"}} )
            console.log(response)
            const sphere = new THREE.SphereGeometry(0.008); // Create sphere (point) at intersection.

            const m = new Mesh(sphere, material)
            m.position.set(intersectionPoint.x, intersectionPoint.y, intersectionPoint.z)

            console.log(sphere, m, intersectionPoint)
            console.log(camera)
            scene.add(m); // Add sphere to scene.
        }
    });
}

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );
    camera.setViewOffset( window.innerWidth, window.innerHeight)


}
setupIntersectionHandler(renderer, camera, scene)

function animate() {
    requestAnimationFrame( animate );

    renderer.render( scene, camera );
}

animate();




