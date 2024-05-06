import * as THREE from "three";
function deepMerge(obj1, obj2) {
    if ((!(typeof(obj1)==="object" ) )||( !(typeof(obj2)==="object" ))){
        return obj2;
    }
    var res;
    const ks=new Set([...Object.keys(obj1), ...Object.keys( obj2)]);
    for (const k of ks) {
        var o1=Object.hasOwn(obj1, k)
        var o2=Object.hasOwn(obj2, k);
        var _exp1=`obj1.${k}`
        var _exp2=`obj2.${k}`
        var _exp3=`obj1.${k}=res;`
        if (o1 && o2){

            res=deepMerge(eval(_exp1),eval(_exp2))
            eval(_exp3)


        }else if (o1){



        } else if (o2){
            res=eval(_exp2)
            eval(_exp3)


        }
    }

    return obj1

}

export class ViewerProps extends Object{
    renderer={
        localClippingEnabled :true,
        shadowMap:{
            enabled:true
        }
    }
    clearColor=`#000`

    constructor(props) {
        super()
        props? deepMerge(this, props): null
    }
    set(props){
        props? deepMerge(this,props ): null
    }
}



export class Viewer extends Object {

    constructor(props) {
        super();

        this.viewerProps=new ViewerProps(props)


    }
    initRenderer(){
        const renderer= new THREE.WebGLRenderer();
        deepMerge(renderer, this.viewerProps.renderer);
        renderer.setPixelRatio( window.devicePixelRatio );
        renderer.setSize( window.innerWidth, window.innerHeight );
        renderer.setClearColor( this.viewerProps.clearColor );
        return renderer
    }
    setupScene(){


    }
    setupSystems(){

    }
}