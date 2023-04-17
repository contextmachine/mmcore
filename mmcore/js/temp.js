#! node
const THREE = require("three");
const vm = require("node:vm");
const net = require("node:net")
const fs = require("node:fs")


function createText( message, height ) {

	const canvas = document.createElement( 'canvas' );
	const context = canvas.getContext( '2d' );
	let metrics = null;
	const textHeight = 100;
	context.font = 'normal ' + textHeight + 'px Arial';
	metrics = context.measureText( message );
	const textWidth = metrics.width;
	canvas.width = textWidth;
	canvas.height = textHeight;
	context.font = 'normal ' + textHeight + 'px Arial';
	context.textAlign = 'center';
	context.textBaseline = 'middle';
	context.fillStyle = '#ffffff';
	context.fillText( message, textWidth / 2, textHeight / 2 );

	const texture = new THREE.Texture( canvas );
	texture.needsUpdate = true;

	const material = new THREE.MeshBasicMaterial( {
		color: 0xffffff,
		side: THREE.DoubleSide,
		map: texture,
		transparent: true,
	} );
	const geometry = new THREE.PlaneGeometry(
		( height * textWidth ) / textHeight,
		height
	);
	const plane = new THREE.Mesh( geometry, material );
	return plane;

}




const server = net.createServer((c) => {
  // 'connection' listener.

    console.log('client connected');
    c.on('end', () => {
        console.log('client disconnected');
    });

    c.on('data', (buff)=>{
        if (buff.toString()==="stop-serve"){
            c.end(()=>{
                fs.rm("/tmp/cxm.sock")
            })
        }

        const context={ THREE:THREE, group: new THREE.Group(),localResult:'initial' ,objLoader:new THREE.ObjectLoader()}
        vm.createContext(context)

        const vm_result= vm.runInContext(buff.toString(), context)(require);
        console.log(`vm: ${vm_result}, local: ${context.localResult}`)
        c.write(JSON.stringify(context.localResult));
    })

    c.pipe(c);
});

server.on('error', (err) => {
  console.log(err);
});
server.listen('/tmp/cxm.sock', () => {
     console.log('server bound');
     });

server.on("drop", (data)=>{
    fs.rmSync("/tmp/cxm.sock")
    console.log(data.localAddress)
    server.close((err)=>{
        fs.rmSync("/tmp/cxm.sock")
        console.log("closed")
        process.exit(1)


    })
});


