from enum import IntEnum

import numpy as np


class PointsOrder(IntEnum):
    COLLINEAR = -1
    CW = 0
    CCW = 1


def points_order(points, close=True) -> PointsOrder:
    if len(points) < 3:
        raise ValueError(f"At least 3 points expected! \n{points}")
    if close:
        points = np.concatenate([points, [points[0]]])

    determinant = sum(
        (points[i + 1][0] - points[i][0]) * (points[i + 1][1] + points[i][1]) for i in range(len(points) - 1)
    )
    if determinant > 0:
        return PointsOrder.CW
    elif determinant < 0:
        return PointsOrder.CCW
    else:
        return PointsOrder.COLLINEAR


# language=Glsl
"""
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // coordinates
    float w = 1.0/iResolution.y; // size of a pixel
    float x = fragCoord.x * w;
    float y = fragCoord.y * w;
    
    // function f(x) and naive distancefloa
    t f = 0.5 + 0.005/x + 0.01*x/(x-1.0) + 0.1*sin(x*20.0);
    float d = abs(f-y);
    
    // enable and disable the technique
    if( mod(iTime,2.0)>1.0 )
    {
        // derivative f'(x) and better distance estimation
        float fy = -0.005/(x*x) - 0.01/((x-1.0)*(x-1.0)) + 0.1*20.0*cos(20.0*x);
        fy = min( abs(fy), 40.0 );
        // distance estimation
        d /= sqrt( 1.0 + fy*fy );
    }
    else
    {
        d *= 0.707107;
    }
    
    // background
    vec3 col = vec3(0.15);

    // graph
    float thickness = iResolution.y/135.0;
    // graph thickness is 8 pixels at 1080p
    // cubic filtering is 2 pixels wide
    col = mix( col, vec3(0.0,0.7,1.0), smoothstep((0.5*thickness+2.0)*w,
                                                  (0.5*thickness+0.0)*w,
                                                   d) );
    fragColor = vec4(col,1.0);
}
"""
