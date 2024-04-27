#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/sysinfo.h>
#include <stdbool.h>
#include <semaphore.h>
#include <sys/time.h>

#define NUMBER_OF_TASKS 3


/** Global variables */
cv::String faceCascadePath;
cv::CascadeClassifier faceCascade;
double overall_start_time, overall_stop_time;

typedef struct
{
    int threadId;
} ThreadArgs_t;

typedef struct
{
    int period;
    int burst_time;
    int count_for_period;
    struct sched_param priority_param;
    void *(*thread_handle)(void *);
    pthread_t thread;
    ThreadArgs_t thread_args;
    void *return_Value;
    pthread_attr_t attribute;
    int target_cpu;
} RmTask_t;

sem_t semaphore_10ms, semaphore_20ms;

double read_time(double *var)
{
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0)
    {
        perror("readTOD");
        return 0.0;
    }
    else
    {
        *var = ((double)(((double)tv.tv_sec * 1000) + (((double)tv.tv_usec) / 1000.0)));
    }
    return (*var);
}



void detectFaceOpenCVHaar(cv::CascadeClassifier faceCascade, cv::Mat &frameOpenCVHaar, int inHeight=300, int inWidth=0)
{
    int frameHeight = frameOpenCVHaar.rows;
    int frameWidth = frameOpenCVHaar.cols;
    if (!inWidth)
        inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float)inHeight;
    float scaleWidth = frameWidth / (float)inWidth;

    cv::Mat frameOpenCVHaarSmall, frameGray;
    cv::resize(frameOpenCVHaar, frameOpenCVHaarSmall, cv::Size(inWidth, inHeight));
    cv::cvtColor(frameOpenCVHaarSmall, frameGray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(frameGray, faces);

    for ( size_t i = 0; i < faces.size(); i++ )
    {
      int x1 = (int)(faces[i].x * scaleWidth);
      int y1 = (int)(faces[i].y * scaleHeight);
      int x2 = (int)((faces[i].x + faces[i].width) * scaleWidth);
      int y2 = (int)((faces[i].y + faces[i].height) * scaleHeight);
      rectangle(frameOpenCVHaar, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,255,0), (int)(frameHeight/150.0), 4);
    }
}


void *FaceDetectService(void *args)
{
    RmTask_t *task_parameters  = (RmTask_t *)args;
    
   faceCascadePath = "./haarcascade_frontalface_default.xml";
    if(!faceCascade.load(faceCascadePath))
    {
        printf("--(!)Error loading face cascade\n");
        return NULL;
    }

    cv::VideoCapture source;
    
    source.open(0, cv::CAP_V4L);

    cv::Mat frame;

    double tt_opencvHaar = 0;
    double fpsOpencvHaar = 0;

    while (true)
    {
        source >> frame;
        if (frame.empty())
            break;

        double t = cv::getTickCount();
        detectFaceOpenCVHaar(faceCascade, frame);
        tt_opencvHaar = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        fpsOpencvHaar = 1/tt_opencvHaar;

        cv::putText(frame, cv::format("OpenCV HAAR ; FPS = %.2f",fpsOpencvHaar), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(0, 0, 255), 4);

        cv::imshow("OpenCV - HAAR Face Detection", frame);

        int k = cv::waitKey(5);
        if(k == 27)
        {
            cv::destroyAllWindows();
            break;
        }
    }
    return NULL;
}


void *ServoActuatorService(void *args)
{
    RmTask_t *task_parameters  = (RmTask_t *)args;
    
    struct sched_param schedule_param;
    int policy;
    double execution_complete_time_for_a_loop;
    double execution_start_time_for_a_loop;

    while (1)
    {
        read_time(&execution_start_time_for_a_loop);
        // sem_wait(&semaphore_10ms);
   
        printf("Actuation Service will execute\n");

        pthread_getschedparam(pthread_self(), &policy, &schedule_param);
        read_time(&execution_complete_time_for_a_loop);
        
        // printf("Thread10 | priority = %d | time stamp(arrival) %lf msec | CPU burst time : %lf \n", schedule_param.sched_priority, (execution_start_time_for_a_loop - overall_start_time), (execution_complete_time_for_a_loop - execution_start_time_for_a_loop));
        sleep(1);
    }

    return NULL;
}


void *ServoShootService(void *args)
{
    RmTask_t *task_parameters  = (RmTask_t *)args;
    
    struct sched_param schedule_param;
    int policy;
    double execution_complete_time_for_a_loop;
    double execution_start_time_for_a_loop;

    while (1)
    {
        read_time(&execution_start_time_for_a_loop);
        // sem_wait(&semaphore_10ms);
   
        printf("Shooting service will execute\n");

        pthread_getschedparam(pthread_self(), &policy, &schedule_param);
        read_time(&execution_complete_time_for_a_loop);
        
        // printf("Thread10 | priority = %d | time stamp(arrival) %lf msec | CPU burst time : %lf \n", schedule_param.sched_priority, (execution_start_time_for_a_loop - overall_start_time), (execution_complete_time_for_a_loop - execution_start_time_for_a_loop));
        sleep(1);

    }

    return NULL;
}



void print_scheduler(void)
{
    int schedType;
    schedType = sched_getscheduler(getpid());
    switch (schedType)
    {
    case SCHED_FIFO:
        printf("Pthread Policy is SCHED_FIFO\n");
        break;
    case SCHED_OTHER:
        printf("Pthread Policy is SCHED_OTHER\n");
        break;
    case SCHED_RR:
        printf("Pthread Policy is SCHED_OTHER\n");
        break;
    default:
        printf("Pthread Policy is UNKNOWN\n");
    }
}


int main( int argc, const char** argv )
{


    pthread_t threads[NUMBER_OF_TASKS];
    cpu_set_t threadcpu;

    int rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    int rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    RmTask_t tasks[NUMBER_OF_TASKS] = {
        {.period = 20,     // ms
         .burst_time = 10, // ms
         .priority_param = {rt_max_prio},
         .thread_handle = &FaceDetectService,
         .thread = threads[0],
         .thread_args = {0},
         .target_cpu = 2},

        {.period = 50,
         .burst_time = 20,
         .priority_param = {rt_max_prio - 1},
         .thread_handle = &ServoActuatorService,
         .thread = threads[1],
         .thread_args = {1},
         .target_cpu = 0},

        {.period = 50,
         .burst_time = 20,
         .priority_param = {rt_max_prio - 2},
         .thread_handle = &ServoShootService,
         .thread = threads[2],
         .thread_args = {2},
         .target_cpu = 0}


    };
    //Initialize Semaphore
    sem_init(&semaphore_10ms, false, 1);
    sem_init(&semaphore_20ms, false, 1);

    pthread_attr_t attribute_flags_for_main; // for schedular type, priority
    struct sched_param main_priority_param;

    printf("This system has %d processors configured and %d processors available.\n", get_nprocs_conf(), get_nprocs());

    printf("Before adjustments to scheduling policy:\n");
    print_scheduler();

    CPU_ZERO(&threadcpu); // clear all the cpus in cpuset


    main_priority_param.sched_priority = rt_max_prio;
    for (int i = 0; i < NUMBER_OF_TASKS; i++)
    {
        CPU_SET(tasks[i].target_cpu, &threadcpu);
        
        // initialize attributes
        pthread_attr_init(&tasks[i].attribute);
        
        pthread_attr_setinheritsched(&tasks[i].attribute, PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setschedpolicy(&tasks[i].attribute, SCHED_FIFO);
        pthread_attr_setschedparam(&tasks[i].attribute, &tasks[i].priority_param);
        pthread_attr_setaffinity_np(&tasks[i].attribute, sizeof(cpu_set_t), &threadcpu);

        CPU_ZERO(&threadcpu);
    }

    pthread_attr_init(&attribute_flags_for_main);

    pthread_attr_setinheritsched(&attribute_flags_for_main, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attribute_flags_for_main, SCHED_FIFO);
    pthread_attr_setaffinity_np(&attribute_flags_for_main, sizeof(cpu_set_t), &threadcpu);

    // Main thread is already created we have to modify the priority and scheduling scheme
    int status_setting_schedular = sched_setscheduler(getpid(), SCHED_FIFO, &main_priority_param);
    if (status_setting_schedular)
    {
        printf("ERROR; sched_setscheduler rc is %d\n", status_setting_schedular);
        perror(NULL);
        exit(-1);
    }

    printf("After adjustments to scheduling policy:\n");
    print_scheduler();


    read_time(&overall_start_time);

    for (int i = 0; i < NUMBER_OF_TASKS; i++)
    {
        // Create a thread
        // First paramter is thread which we want to create
        // Second parameter is the flags that we want to give it to
        // third parameter is the routine we want to give
        // Fourth parameter is the value
        printf("Setting thread %d to core %d\n", i, tasks[i].target_cpu);
        
        

        if (pthread_create(&tasks[i].thread, &tasks[i].attribute, tasks[i].thread_handle, &tasks[i]) != 0)
        {
            perror("Create_Fail");
        }

        
    }


    printf("Test Conducted over %lf msec\n", (double)(overall_stop_time - overall_start_time));

    for (int i = 0; i < NUMBER_OF_TASKS; i++)
    {
        pthread_join(tasks[i].thread, &tasks[i].return_Value);
    }

    if (pthread_attr_destroy(&tasks[0].attribute) != 0)
        perror("attr destroy");
    if (pthread_attr_destroy(&tasks[1].attribute) != 0)
        perror("attr destroy");

    sem_destroy(&semaphore_10ms);
    sem_destroy(&semaphore_20ms);  

    
}