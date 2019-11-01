typedef struct __attribute__((packed)) _Object{

    float x;
    float y;
    float vx;
    float vy;
    float ax;
    float ay;
    float mass;

}Object;


__kernel void simulate(__global Object* objects, int size,float delta) {

    int i = get_global_id(0);
    
    objects[i].ax = 0;
    objects[i].ay = 0;

    for(int j = 0;j < size;j++){
        if(i==j)continue;
        if(objects[i].x == objects[j].x && objects[i].y == objects[j].y)continue;

        float xi = objects[i].x - objects[j].x;
        float yi = objects[i].y - objects[j].y;

        float l2 = pow((xi),2) + pow((yi),2);

        if(l2 < 1) continue;

        float mm = objects[i].mass * objects[j].mass;
        float a = mm / l2;

        float ax = a*xi*xi/l2;
        float ay = a*yi*yi/l2;
        if(xi < 0) ax*=-1;
        if(yi < 0) ay*=-1;

        objects[i].ax += -ax;
        objects[i].ay += -ay;
    }

    objects[i].x += objects[i].vx * delta;
    objects[i].y += objects[i].vy * delta;

    objects[i].vx += objects[i].ax * delta;
    objects[i].vy += objects[i].ay * delta;

    barrier(CLK_LOCAL_MEM_FENCE); /* Wait for others in work-group */
}