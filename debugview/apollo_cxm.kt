fun add(a:Int, b:Int): Int{

    return  b+a;
}


fun main() {
    val args: Array<String> = arrayOf("1","2","3","4","5","6","7","8")
    var result:Int = 0
    for (arg in args){
        try {

            result+= arg.toInt()
        } catch (e: ArithmeticException) {
            println("Argument value (${arg}) is not Int")
            throw IllegalStateException(e)

        }

        // Working with result
    }
        


    println("Result:  $result")
}



