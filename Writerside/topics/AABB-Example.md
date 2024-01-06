# Intervals & AABB

The classic case is AABB - Axis Aligned Bounding Box.
In other cases Scalar Value Interval in 1D, Axis Aligned Rectangle in 2D and Axis Aligned Box in 3-ND.

## Signature

Удобное представление для AABB `(2, N)` 2 колонки `min, max` (количество этих свойств для данного типа неизменно
для 1d (интервал)
2d  (прямоугольник), 3d (бокс) и
так и Nd случая) и N строк, представляющих размерности.

|         | **min** | **max** |  
|---------|:-------:|---------|  
| **x**   | `float` | `float` |  
| **y**   | `float` | `float` | 
| **z**   | `float` | `float` | 
| .  .  . | `float` | `float` |   
| **n**   | `float` | `float` | 

## Examples

    Interval,  AABR,  AABB

Так для ограничивающего прямоугольника мы будем иметь 2 строки `x, y`, для ограничивающего бокса 3 `x, y, z` . При
необходимости это
определение легко расширяется до больших размерностей.

Это представление дает возможность использовать функцию `aabb` на любых размерностях без изменений, используя все
возможности `numpy`, а также сохранить объектную структуру.

