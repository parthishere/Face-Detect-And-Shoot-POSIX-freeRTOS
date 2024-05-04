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
#include <mqueue.h>

#define FRAME_HEIGHT 320
#define FRAME_WIDTH 240

#define NANOSEC_PER_SEC 1000000000

#define OVERALL_DEADLINE 190
#define FACE_DETECTION_DEADLINE 100
#define SERVO_ACTUATION_DEADLINE 50
#define SERVO_SHOOT_DEADLINE 40

#define NUMBER_OF_TASKS 3

#define CUSTOM_MQ_NAME "/send_receive_mq"

struct mq_attr mq_attr;
mqd_t message_queue_instance;

/** Global variables */
cv::String faceCascadePath;
cv::CascadeClassifier faceCascade;
double overall_start_time, overall_stop_time;

double wcet_servo_actuation;
double wcet_servo_shoot;
double wcet_face_recognition;

static bool execution_complete = false;

#ifdef IS_RPI

#define NUM_GPIO 32

#define SERVO1_PIN 4
#define SERVO2_PIN 23
#define LASER_PIN 18

#define MIN_WIDTH 1000
#define MAX_WIDTH 2000
#define SERVO_RANGE 180

#include <pigpio.h>

void change_servo_degree(int output_pin, uint8_t degree)
{
    if (degree < 0)
    {
        degree = 0;
    }
    else if (degree > SERVO_RANGE)
    {
        degree = SERVO_RANGE;
    }

    int pwmWidth = MIN_WIDTH + (degree * (MAX_WIDTH - MIN_WIDTH) / SERVO_RANGE);

    gpioServo(output_pin, pwmWidth);
}

#endif

typedef struct
{
    int threadId;
} ThreadArgs_t;

typedef struct
{
    int period;
    int burst_time;
    struct sched_param priority_param;
    void *(*thread_handle)(void *);
    pthread_t thread;
    ThreadArgs_t thread_args;
    void *return_Value;
    pthread_attr_t attribute;
    int target_cpu;
} RmTask_t;

typedef struct
{
    int x1;
    int y1;
    int x2;
    int y2;
} Points_t;

sem_t semaphore_face_detect, semaphore_servo_actuator, semaphore_servo_shoot;

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

void delay_ns(int ns)
{

    double residual;
    struct timeval current_time_val;
    struct timespec delay_time = {0, ns}; // delay for 33.33 msec, 30 Hz
    struct timespec remaining_time;
    int rc;

    gettimeofday(&current_time_val, (struct timezone *)0);

    rc = nanosleep(&delay_time, &remaining_time);

    if (rc == EINTR)
    {
        residual = remaining_time.tv_sec + ((double)remaining_time.tv_nsec / (double)NANOSEC_PER_SEC);

        if (residual > 0.0)
            printf("residual=%lf, sec=%d, nsec=%d\n", residual, (int)remaining_time.tv_sec, (int)remaining_time.tv_nsec);
    }
    else if (rc < 0)
    {
        perror("delay_ns nanosleep");
        exit(-1);
    }
}

Points_t detectFaceOpenCVLBP(cv::CascadeClassifier faceCascade, cv::Mat &frameGray, int inHeight = 300, int inWidth = 0)
{
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(frameGray, faces);

    if (!faces.empty())
    {
        int x1 = faces[0].x;
        int y1 = faces[0].y;
        int x2 = faces[0].x + faces[0].width;
        int y2 = faces[0].y + faces[0].height;
        cv::rectangle(frameGray, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        return {x1, y1, x2, y2};
    }
    return {0, 0, 0, 0};
}

void *FaceDetectService(void *args)
{
    RmTask_t *task_parameters = (RmTask_t *)args;

    struct sched_param schedule_param;
    int policy, cpucore;
    pthread_t thread;
    cpu_set_t cpuset;

    thread = pthread_self();
    cpucore = sched_getcpu();

    pthread_getschedparam(pthread_self(), &policy, &schedule_param);
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

    std::string faceCascadePath = "./lbpcascade_frontalface.xml";
    cv::CascadeClassifier faceCascade;

    if (!faceCascade.load(faceCascadePath))
    {
        printf("--(!)Error loading face cascade\n");
        return NULL;
    }

    cv::VideoCapture source;
    source.open(0, cv::CAP_V4L);

    cv::Mat frame, frameGray;
    cv::Size frameSize(FRAME_HEIGHT, FRAME_WIDTH);

    double start_ms;
    double end_ms;
    double fps, execute_ms;

    while (true)
    {

        //sem_wait(&semaphore_face_detect);
        read_time(&start_ms);

        source >> frame;
        if (frame.empty())
            break;
        cv::imshow("Original Frame", frame);
        cv::resize(frame, frame, frameSize);
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

        Points_t face_points = detectFaceOpenCVLBP(faceCascade, frameGray);

        cv::imshow("OpenCV - LBP Face Detection", frameGray);

        if (face_points.x1 != 0 && face_points.x2 != 0)
        {
            // Allocate memory for the Points_t structure
            Points_t *points_buffer_ptr = (Points_t *)malloc(sizeof(Points_t));

            // Copy the face_points data into the allocated memory
            memcpy(points_buffer_ptr, &face_points, sizeof(Points_t));

            printf("sender - Message to send | X1 = %d X2 = %d Y1 = %d Y2 = %d | Sending message of size = %lu\n",
                   points_buffer_ptr->x1, points_buffer_ptr->x2, points_buffer_ptr->y1, points_buffer_ptr->y2, sizeof(Points_t));

            // Send the message containing the Points_t structure
            if (mq_send(message_queue_instance, (const char *)points_buffer_ptr, sizeof(Points_t), 0) == -1)
            {
                perror("mq_send");
                free(points_buffer_ptr);
            }
            else
            {
                printf("sender - message ptr %p successfully sent\n", points_buffer_ptr);
            }
        }
        else
        {
#ifdef IS_RPI
            gpioWrite(LASER_PIN, 0);
#endif
        }

        read_time(&end_ms);
        execute_ms = (end_ms - start_ms);
        fps = 1000 / execute_ms;

        if (wcet_face_recognition < execute_ms)
        {
            wcet_face_recognition = execute_ms;
        }

        printf("FPS: %.2f execution time : %f, current time from start 1s 2ms, execution time for one frame 10ms\n", fps, execute_ms);

        int k = cv::waitKey(5);
        if (k == 27)
        {
            execution_complete = true;
            // set flag
            break;
        }
        
    }

#ifdef IS_RPI
    gpioTerminate();
#endif

    cv::destroyAllWindows();
    return NULL;
}

void *ServoActuatorService(void *args)
{
    RmTask_t *task_parameters = (RmTask_t *)args;

    struct sched_param schedule_param;
    int policy, cpucore;
    pthread_t thread;
    cpu_set_t cpuset;
    double execution_complete_time_for_a_loop;
    double execution_start_time_for_a_loop;

    thread = pthread_self();
    cpucore = sched_getcpu();

    pthread_getschedparam(pthread_self(), &policy, &schedule_param);
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

    int center_x, center_y;
    int degree = 0;

#ifdef IS_RPI

    std::cout << "Starting Thread 1 now! Press CTRL+C to exit" << std::endl;

    while (!execution_complete)
    {

        Points_t received_points;
        ssize_t received_size = mq_receive(message_queue_instance, (char *)&received_points, sizeof(Points_t), NULL);

        if (received_size == -1)
        {
            perror("mq_receive");
        }

        else
        {
            read_time(&execution_start_time_for_a_loop);
            printf("receiver - Received message | X1 = %d X2 = %d Y1 = %d Y2 = %d | Received message of size = %ld\n",
                   received_points.x1, received_points.x2, received_points.y1, received_points.y2, received_size);

            center_x = (received_points.x1 + received_points.x2) / 2;
            center_y = (received_points.y1 + received_points.y2) / 2;
            degree = ((degree + 10) % 180);

            change_servo_degree(SERVO1_PIN, degree);
            change_servo_degree(SERVO2_PIN, degree);

            
            read_time(&execution_complete_time_for_a_loop);

            double execution_time = execution_complete_time_for_a_loop - execution_start_time_for_a_loop;
            

            printf("Execution time for Servo Actuation service is : %f ms\n", execution_time);

            if (wcet_servo_actuation < execution_time)
            {
                wcet_servo_actuation = execution_time;
            }

            sem_post(&semaphore_servo_shoot);
        }
    }

    gpioTerminate();

#endif

    return NULL;
}

void *ServoShootService(void *args)
{
    RmTask_t *task_parameters = (RmTask_t *)args;

    double execution_complete_time_for_a_loop;
    double execution_start_time_for_a_loop;

    struct sched_param schedule_param;
    int policy, cpucore;
    pthread_t thread;
    cpu_set_t cpuset;

    thread = pthread_self();
    cpucore = sched_getcpu();

    pthread_getschedparam(pthread_self(), &policy, &schedule_param);
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

#ifdef IS_RPI

    std::cout << "Starting Thread 2 now! Press CTRL+C to exit" << std::endl;

    int degree = 0;
    while (!execution_complete)
    {

        sem_wait(&semaphore_servo_shoot);
        read_time(&execution_start_time_for_a_loop);
        printf("Shoot !!!! \n\r");
        // change_servo_degree(SERVO1, degree);
        

        gpioWrite(LASER_PIN, 1);

        read_time(&execution_complete_time_for_a_loop);

        double execution_time = execution_complete_time_for_a_loop - execution_start_time_for_a_loop;

        printf("Execution time for Servo Shoot service is : %f ms\n", execution_time);

        if (wcet_servo_shoot < execution_time)
        {
            wcet_servo_shoot = execution_time;
        }

        sem_post(&semaphore_face_detect);
    }
    gpioTerminate();

#endif

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

int main(int argc, const char **argv)
{

#ifdef IS_RPI

    std::cout << "Raspberry PI " << std::endl;

    if (gpioInitialise() < 0)
        return -1;
    
    gpioSetMode(LASER_PIN, PI_OUTPUT);

#else

    std::cout << "Linux System " << std::endl;

#endif
    pthread_t threads[NUMBER_OF_TASKS];
    cpu_set_t threadcpu;

    /* setup common message q attributes */
    mq_attr.mq_maxmsg = 10;
    mq_attr.mq_msgsize = sizeof(Points_t);
    mq_attr.mq_flags = 0;

    mq_unlink(CUSTOM_MQ_NAME); // Unlink if the previous message queue exists

    message_queue_instance = mq_open(CUSTOM_MQ_NAME, O_CREAT | O_RDWR, S_IRWXU, &mq_attr);
    if (message_queue_instance == (mqd_t)(-1))
    {
        perror("mq_open");
    }

    int rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    int rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    RmTask_t tasks[NUMBER_OF_TASKS] = {
        // {.period = 20,     // ms
        //  .burst_time = 10, // ms
        //  .priority_param = {rt_max_prio},
        //  .thread_handle = &FaceDetectService,
        //  .thread = threads[0],
        //  .thread_args = {0},
        //  .target_cpu = 2},
        {20, 10, {rt_max_prio}, &FaceDetectService, threads[0], {0}, NULL, tasks[0].attribute, 1},
        {50, 20, {rt_max_prio - 1}, &ServoActuatorService, threads[1], {1}, NULL, tasks[1].attribute, 0},
        {50, 20, {rt_max_prio - 2}, &ServoShootService, threads[2], {2}, NULL, tasks[2].attribute, 0}

    };
    // Initialize Semaphore
    sem_init(&semaphore_face_detect, false, 1);
    sem_init(&semaphore_servo_actuator, false, 1);
    sem_init(&semaphore_servo_shoot, false, 1);

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
    if (pthread_attr_destroy(&tasks[2].attribute) != 0)
        perror("attr destroy");

    sem_destroy(&semaphore_face_detect);
    sem_destroy(&semaphore_servo_actuator);
    sem_destroy(&semaphore_servo_shoot);
}
