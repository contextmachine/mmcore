# include "stdlib.h"

class NURBSSurfaceData{
    public:
    double* control_points;
    double* knots_u ;
    double* knots_v;
    int size_u;
    int size_v;
    int degree_u;
    int degree_v;
    NURBSSurfaceData()=default;
    NURBSSurfaceData(double* control_points, double* knots_u ,double* knots_v,int size_u,int size_v,int degree_u,int degree_v):control_points(control_points), knots_u(knots_u),knots_v(knots_v), size_u(size_u),size_v(size_v),degree_u(degree_u),degree_v(degree_v){};
    ~NURBSSurfaceData(){
       if (control_points!=NULL){
        free(control_points);
       } 
       if (knots_u!=NULL){
        free(knots_u);
       } 
        if (knots_v!=NULL){
        free(knots_v);
       } 
      
    }

};
    
