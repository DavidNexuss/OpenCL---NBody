#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <CL/cl.h>

#define OBJECT_LIST_SIZE 3000
#define KERNEL_FILE "src/main.cl"
#define MAX_SOURCE_SIZE 0x2000

#define DEBUG
//#define HEADLESS
#define SWIDTH 1920
#define SHEIGHT 1080
#define GPU

#define RANDMAX 300
#define RANDOM() rand() % RANDMAX

using namespace std;


GLFWwindow* window;
cl_context context;
cl_command_queue command_queue;
cl_kernel kernel;
struct Object{

    cl_float x;
    cl_float y;
    cl_float vx;
    cl_float vy;
    cl_float ax;
    cl_float ay;
    cl_float mass;

}__attribute__((packed));

struct ObjectList{

    Object *buffer;
    cl_mem mem_obj;

    bool writeBuffer(){
        return clEnqueueWriteBuffer(command_queue,mem_obj,CL_TRUE,0,OBJECT_LIST_SIZE * sizeof(Object),buffer,0,NULL,NULL);
    }

    bool readBuffer(){
	    return clEnqueueReadBuffer(command_queue, mem_obj, CL_TRUE,0,OBJECT_LIST_SIZE * sizeof(Object),buffer,0,NULL,NULL);
    }

    void printList(){

        readBuffer();
        for (size_t i = 0; i < OBJECT_LIST_SIZE; i++)
        {
           // cout << buffer[i].x << endl;
        }
        
    }
    ObjectList(){

        buffer = new Object[OBJECT_LIST_SIZE];
        for (size_t i = 0; i < OBJECT_LIST_SIZE; i++)
        {
            int x = RANDOM() - RANDMAX / 2;
            int y = RANDOM() - RANDMAX / 2;
            if((x*x + y*y) > (RANDMAX*RANDMAX / 4)){
                i--;
                cout << "out" << endl;
                continue;
            }
            buffer[i].x = x;
            buffer[i].y = y;

            buffer[i].mass = RANDOM();
        }
        
        cl_int ret = 0;
        mem_obj = clCreateBuffer(context,CL_MEM_READ_WRITE,OBJECT_LIST_SIZE * sizeof(Object),buffer,&ret);
        if(ret){
            fprintf(stderr,"Error in creating buffer %i",this);
            exit(1);
        }
    }

    ~ObjectList(){
        delete mem_obj;
        delete [] buffer;
    }
};

ObjectList* list;

void glfw_error_callback(int error,const char* description){

    fprintf(stderr,"Error: %s\n",description);
}
bool createGLFWindow(){

    if(!glfwInit()) return false;
    glfwSetErrorCallback(glfw_error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,0);
    
    window = glfwCreateWindow(SWIDTH,SHEIGHT,"N Body",NULL,NULL);
    if(!window) return false;

    glfwMakeContextCurrent(window);
    
    #ifdef DEBUG
        cout << "Window Created";
    #endif DEBUG
    return true;
}

bool createOpenGLContext(){ 
    glewExperimental=true;
    if(glewInit()){
        fprintf(stderr,"Error in initilizing glew");
        return false;
    }

    glViewport(0,0,SWIDTH,SHEIGHT);
    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    glOrtho(0,SWIDTH,0,SHEIGHT,0,1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    

    return true;
}
bool createOpenCLContext(){

    FILE *fp;
    fp = fopen(KERNEL_FILE,"r");
    if(!fp){
        fprintf(stderr,"Failed to load kernel\n");
        return false;
    }

    char* source_str = new char[MAX_SOURCE_SIZE];
    size_t source_size = fread(source_str,1,MAX_SOURCE_SIZE,fp);
    fclose(fp);

    cl_int ret = 0;

    cl_platform_id platform_id = 0;
    cl_device_id device_id = 0;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    ret = clGetPlatformIDs(1,&platform_id,&ret_num_platforms);
    ret = clGetDeviceIDs(platform_id,CL_DEVICE_TYPE_DEFAULT,1,&device_id,&ret_num_devices);

    context = clCreateContext(NULL,1,&device_id,NULL,NULL,&ret);
    command_queue = clCreateCommandQueue(context,device_id,0,&ret);

    cl_program program = clCreateProgramWithSource(context,1,(const char**)&source_str,(const size_t*)&source_size,&ret);
	ret = clBuildProgram(program,1,&device_id,NULL,NULL,NULL);
    
    if(ret){
		cout << "Build error " << ret << endl;
		return false;
	}

    kernel = clCreateKernel(program, "simulate",&ret);
    if(ret){
        cout << "Create kernel error " << ret << endl;
        return false;
    }
    delete [] source_str;

    return true;

}

void dispose(){
    glfwDestroyWindow(window);
    glfwTerminate();
    delete list;
}

void initSimulation(){

    list = new ObjectList();
    list->writeBuffer();
    #ifdef DEBUG
        cout << "List created" << endl;
        cout << "WriteBuffer operation " << endl;
    #endif
}

void cpustep(float delta){


    Object* objects = list->buffer;

    for (size_t i = 0; i < OBJECT_LIST_SIZE; i++)
    {
    
    objects[i].ax = 0;
    objects[i].ay = 0;

    for(int j = 0;j < OBJECT_LIST_SIZE;j++){
        if(i==j)continue;
        if(objects[i].x == objects[j].x && objects[i].y == objects[j].y)continue;

        float xi = objects[i].x - objects[j].x;
        float yi = objects[i].y - objects[j].y;

        float l2 = pow((xi),2) + pow((yi),2);

        if(l2 < 2) continue;

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
    }
    
}
double time_elapsed = 0.00001;
void step(){

	
	const int size = OBJECT_LIST_SIZE;
	const float delta = 1/800.0;
    time_elapsed += delta;
	cl_int ret = 0;

	//writeObjectBuffer();
	//set arguments
	ret = clSetKernelArg(kernel,0,sizeof(cl_mem),(void*)&list->mem_obj);
	ret = clSetKernelArg(kernel,1,sizeof(int),&size);
	ret = clSetKernelArg(kernel,2,sizeof(float),&delta);

	size_t global_item_size = OBJECT_LIST_SIZE;
	size_t local_item_size = OBJECT_LIST_SIZE / 10;
	ret = clEnqueueNDRangeKernel(command_queue,kernel,1,NULL,&global_item_size,&local_item_size,0,NULL,NULL);

}

void draw(){

    const int mul = 2; 
    for (size_t i = 0; i < OBJECT_LIST_SIZE; i++)
    {
        const int T = 3;
        GLfloat pointVertex[] = {list->buffer[i].x * mul + SWIDTH / 2,list->buffer[i].y * mul + SHEIGHT / 2};
        glColor3f( ((1920.0 - pointVertex[0])  / 1920.0) * (time_elapsed / T),
                   ((1080.0 - pointVertex[1]) / 1080.0) * (T / time_elapsed),
                   1 * (T / time_elapsed));
        glEnableClientState(GL_VERTEX_ARRAY);
        glPointSize(mul);
        glVertexPointer(2,GL_FLOAT,0,pointVertex);
        glDrawArrays(GL_POINTS,0,1);
        glDisableClientState(GL_VERTEX_ARRAY);
    }
    
}
int main(){

    srand(time(0));
    if(!createOpenCLContext()){ cout << "Error in creating OpenCL context"; exit(1);}

    #ifdef HEADLESS
    initSimulation();
    step();
    step();
    #else
    bool error; 
    if(!createGLFWindow()){ cout << "Error in creating GLFW window"; exit(1);}
    if(!createOpenGLContext()){ cout << "Error in creating OpenGL context"; exit(1);}

    glEnable(GL_PROGRAM_POINT_SIZE);
    initSimulation();
    list->printList();
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    do{
        
        //glClear(GL_COLOR_BUFFER_BIT);
       
        #ifdef GPU
        step();
        list->readBuffer(); 
        draw();
        #else
        cpustep(1/60.0);
        #endif
        glfwSwapBuffers(window);
        glfwPollEvents();

    } // Revisar que la tecla ESC fue presionada y cerrar la ventana
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
    glfwWindowShouldClose(window) == 0 );
    
    dispose();
    #endif
    
}