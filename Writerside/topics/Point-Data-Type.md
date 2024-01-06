# Vector (Point)

## Signature

### Vector Signature

|         | _scalar value_ |
|:-------:|:--------------:|
|  **x**  |    `float`     |
|  **y**  |    `float`     |          
|  **z**  |    `float`     |     
| .  .  . |    `float`     |         
|  **n**  |    `float`     |

### Vector Array Signature

Сигнатура массива векторов `(i, xy..n)` где i может иметь как 1 и более измерений.

|         |  **x**  |  **y**  |  **z**  | .  .  . |  **n**  |    
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|    
|   _0_   | `float` | `float` | `float` | `float` | `float` |    
|   _1_   | `float` | `float` | `float` | `float` | `float` |                  
| .  .  . | `float` | `float` | `float` | `float` | `float` |                 
|  _-1_   | `float` | `float` | `float` | `float` | `float` |        

## Examples

    Boundary, PolyLine, Curve
    Face, Grid, Surface, Sweep, Extruion
    Shell, Solid

- (10,3) - 1D массив трехмерных точек.

  `NdArray[Point]`, `Boundary`, `PolyLine`, `Curve`

  ```python
  # Square boundary 1 x 1

  np.array([[0.0,0.0,0.0],
            [1.0,0.0,0.0],
            [1.0,1.0,0.0],
            [0.0,1.0,0.0]
            ])

  ```

- (8, 10, 3) - 2D массив массивов трехмерных точек.

  ```python
  # Square boundary extrusion 1 x 1 x 5
  
  np.array([[[0.0,0.0,0.0],
             [1.0,0.0,0.0],
             [1.0,1.0,0.0],
             [0.0,1.0,0.0]],
           
            [[0.0,0.0,5.0],
             [1.0,0.0,5.0],
             [1.0,1.0,5.0],
             [0.0,1.0,5.0]]
           ])
  ```

- (6, 8, 10, 3) - 3D массив массивов трехмерных точек

  `NdArray[NdArray[NdArray[Points]]]` , `NdArray[NdArray[Curve]]`, `NdArray[Surface]`

