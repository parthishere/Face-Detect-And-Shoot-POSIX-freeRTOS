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
#include <math.h>
#include <signal.h>
#include <atomic>

/*
* Example 9
* C: 35 4 3
* T: 50 100 100
* D: 50 50 50
*
* Task 0, WCET=35, Period=50, Utility Sum = 0.700000
* Task 1, WCET=4, Period=100, Utility Sum = 0.740000
* Task 2, WCET=3, Period=100, Utility Sum = 0.770000
*
* Total Utility Sum = 0.770000
* LUB = 0.779763
* RM LUB: Feasible
* Completion time feasibility: Feasible
* Scheduling point feasibility: Feasible
* Deadline monotonic: Feasible
*
* (Period)
* Total utility in EDF: 0.770000 Which is less than 1.0
* EDF on Period: Feasible
* Total utility in LLF: 0.770000 Which is less than 1.0
* LLF on Period: Feasible

* (Deadline)
* Total utility in EDF: 0.840000 Which is less than 1.0
* EDF on Deadline: Feasible
* Total utility in LLF: 0.840000 Which is less than 1.0
* LLF on Deadline: Feasible
*
*/

#define FRAME_HEIGHT 320
#define FRAME_WIDTH 240

#define NANOSEC_PER_SEC 1000000000

#define OVERALL_DEADLINE 150
#define FACE_DETECTION_DEADLINE 50
#define SERVO_ACTUATION_DEADLINE 50
#define SERVO_SHOOT_DEADLINE 50

#define NUMBER_OF_TASKS 3

#define CUSTOM_MQ_NAME "/send_receive_mq"

struct mq_attr mq_attr;
mqd_t message_queue_instance;

/** Global variables */
cv::String faceCascadePath;
cv::CascadeClassifier faceCascade;
double overall_start_time, overall_stop_time;
double face_recognition_start_ms;
double face_recognition_end_ms;

double wcet_servo_actuation;
double wcet_servo_shoot;
double wcet_face_recognition;
double wcet_overall;

int overall_deadline_miss;
int face_detection_deadline_miss;
int servo_actuation_deadline_miss;
int servo_shoot_deadline_miss;

int starting_count = 0;

volatile bool exit_flag = false;
std::atomic<bool> stop_timer = false;

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



void *Sequencer(void *threadp)
{
    struct timeval current_time_val;
    struct timespec delay_time = {0, 50000000}; // delay for 33.33 msec, 30 Hz
    struct timespec remaining_time;
    double current_time;
    double residual;
    int rc, delay_cnt = 0;
    unsigned long long seqCnt = 0;

    gettimeofday(&current_time_val, (struct timezone *)0);
    syslog(LOG_CRIT, "Sequencer thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec - start_time_val.tv_sec), (int)current_time_val.tv_usec / USEC_PER_MSEC);
    printf("Sequencer thread @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec - start_time_val.tv_sec), (int)current_time_val.tv_usec / USEC_PER_MSEC);

    do
    {
        delay_cnt = 0;
        residual = 0.0;

        gettimeofday(&current_time_val, (struct timezone *)0);
        syslog(LOG_CRIT, "Sequencer thread prior to delay @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec - start_time_val.tv_sec), (int)current_time_val.tv_usec / USEC_PER_MSEC);

        delay_ns(50000000);

        seqCnt++;
        gettimeofday(&current_time_val, (struct timezone *)0);
        syslog(LOG_CRIT, "Sequencer cycle %llu @ sec=%d, msec=%d\n", seqCnt, (int)(current_time_val.tv_sec - start_time_val.tv_sec), (int)current_time_val.tv_usec / USEC_PER_MSEC);

        
        syslog(LOG_CRIT, "Task 1 (Frame Sampler thread) Released \n");
        sem_post(&semaphore_face_detect); // Frame Sampler thread
    
        syslog(LOG_CRIT, "Task 2 (Servo Actuation) Released \n");
        sem_post(&semaphore_servo_actuator); // Time-stamp with Image Analysis thread
    
        syslog(LOG_CRIT, "Task 3 (Servo Shoot) Released \n");
        sem_post(&semaphore_servo_shoot); // Difference Image Proc thread
        
        gettimeofday(&current_time_val, NULL);
        syslog(LOG_CRIT, "Sequencer release all sub-services @ sec=%d, msec=%d\n", (int)(current_time_val.tv_sec - start_time_val.tv_sec), (int)current_time_val.tv_usec / USEC_PER_MSEC);

    } while (!!exit_flag);

    sem_post(&semaphore_face_detect);
    sem_post(&semaphore_servo_actuator);
    sem_post(&semaphore_servo_shoot);
   
    
    pthread_exit((void *)0);
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

    double fps, execute_ms;

    while (!exit_flag)
    {
        sem_wait(&semaphore_face_detect);
        read_time(&face_recognition_start_ms);

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

            // Send the message containing the Points_t structure
            if (mq_send(message_queue_instance, (const char *)points_buffer_ptr, sizeof(Points_t), 0) == -1)
            {
                perror("mq_send");
                free(points_buffer_ptr);
            }
        }
        else
        {
#ifdef IS_RPI
            gpioWrite(LASER_PIN, 0);
#endif
        }

        read_time(&face_recognition_end_ms);
        execute_ms = (face_recognition_end_ms - face_recognition_start_ms);
        fps = 1000 / execute_ms;

        if (wcet_face_recognition < execute_ms && starting_count > 5)
        {
            wcet_face_recognition = execute_ms;
        }
        if (execute_ms > FACE_DETECTION_DEADLINE && starting_count > 5)
        {
            face_detection_deadline_miss++;
        }

        // printf("| FPS                             | %.2f       |\n", fps);
        // printf("| Execution Time                  | %.2f ms    |\n\n", execute_ms);

        int k = cv::waitKey(5);
        if (k == 27)
        {
            Points_t *points_buffer_ptr = (Points_t *)malloc(sizeof(Points_t));
            memcpy(points_buffer_ptr, &face_points, sizeof(Points_t));

            exit_flag = true;

            sem_post(&semaphore_face_detect);
            mq_send(message_queue_instance, (const char *)points_buffer_ptr, sizeof(Points_t), 0);
            sem_post(&semaphore_servo_shoot);
            // set flag
            break;
        }

        if (starting_count < 9)
        {
            starting_count++;
        }
    }
    printf("Face Detection service ended !\n\r");

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
    double angle_pan = 0;
    double angle_tilt = 0;
    int angle_pan_int;
    int angle_tilt_int;

#ifdef IS_RPI

    std::cout << "Starting Thread 1 now! Press CTRL+C to exit" << std::endl;

    do
    {

        Points_t received_points;
        ssize_t received_size = mq_receive(message_queue_instance, (char *)&received_points, sizeof(Points_t), NULL);

        if (received_size == -1)
        {
            if (errno == EINTR)
            {
                // Interrupted by signal, check exit_flag and continue if not set
                continue;
            }
            perror("mq_receive");
        }

        else
        {
            read_time(&execution_start_time_for_a_loop);
            // printf("receiver - Received message | X1 = %d X2 = %d Y1 = %d Y2 = %d | Received message of size = %ld\n",
            //        received_points.x1, received_points.x2, received_points.y1, received_points.y2, received_size);

            center_x = (received_points.x1 + received_points.x2) / 2;
            center_y = (received_points.y1 + received_points.y2) / 2;

            if (center_x > 160)
            {
                angle_pan = atan(((320.0 - center_x) / 160.0)) * (180.0 / M_PI);
            }
            else
            {
                angle_pan = atan(((160.0 - center_x) / 160.0)) * (180.0 / M_PI);
                angle_pan = 50 + angle_pan;
            }

            angle_tilt = atan(((240.0 - center_y) / 160.0)) * (180.0 / M_PI);

            angle_pan_int = (int)angle_pan;
            angle_tilt_int = (int)angle_tilt;

            // printf("Angle pan %d Angle tilt %d\n\r", angle_pan_int, angle_tilt_int);

            change_servo_degree(SERVO1_PIN, angle_pan_int);
            change_servo_degree(SERVO2_PIN, angle_tilt_int);

            read_time(&execution_complete_time_for_a_loop);

            double execution_time = execution_complete_time_for_a_loop - execution_start_time_for_a_loop;

            // printf("| Execution time for Servo Actuation      | %.2f ms    |\n\n", execution_time);

            if (wcet_servo_actuation < execution_time && starting_count > 5)
            {
                wcet_servo_actuation = execution_time;
            }
            if (execution_time > SERVO_ACTUATION_DEADLINE && starting_count > 5)
            {
                servo_actuation_deadline_miss++;
            }


        }
    } while (!exit_flag);
    printf("Servo Actuation service ended !\n\r");

#endif

    return NULL;
}

void *ServoShootService(void *args)
{
    RmTask_t *task_parameters = (RmTask_t *)args;

    double execution_complete_time_for_a_servo_shoot;
    double execution_start_time_for_a_servo_shoot;

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
    do
    {
        

        sem_wait(&semaphore_servo_shoot);

        read_time(&execution_start_time_for_a_servo_shoot);

        gpioWrite(LASER_PIN, 1);

        read_time(&execution_complete_time_for_a_servo_shoot);

        double execution_time = execution_complete_time_for_a_servo_shoot - execution_start_time_for_a_servo_shoot;

        double overall_response_time = execution_complete_time_for_a_servo_shoot - face_recognition_start_ms;

        // printf("| Execution time for Servo Shoot          | %.2f ms    |\n", execution_time);
        // printf("| Overall response time                   | %.2f ms    |\n\n", overall_response_time);

        if (wcet_servo_shoot < execution_time && starting_count > 5)
        {
            wcet_servo_shoot = execution_time;
        }
        if (execution_time > SERVO_SHOOT_DEADLINE && starting_count > 5)
        {
            servo_shoot_deadline_miss++;
        }
        if (overall_response_time > OVERALL_DEADLINE && starting_count > 5)
        {
            overall_deadline_miss++;
        }
        if (overall_response_time > wcet_overall && starting_count > 5 && !exit_flag)
        {
            wcet_overall = overall_response_time;
        }

        sem_post(&semaphore_face_detect);
    } while (!exit_flag);
    printf("Servo shoot service ended !\n\r");

    gpioWrite(LASER_PIN, 0);

#endif

    return NULL;
}

void printFinalTable()
{
    printf("+--------------------------------------------------+------------+\n");
    printf("| Metric                                          | Value       |\n");
    printf("+--------------------------------------------------+------------+\n");
    printf("| Overall Deadline Miss Count                     | %11d |\n", overall_deadline_miss);
    printf("| Face Detection Deadline Miss Count              | %11d |\n", face_detection_deadline_miss);
    printf("| Servo Actuation Deadline Miss Count             | %11d |\n", servo_actuation_deadline_miss);
    printf("| Servo Shoot Deadline Miss Count                 | %11d |\n", servo_shoot_deadline_miss);
    printf("| Face Recognition Worst-Case Execution Time      | %8.2f ms |\n", wcet_face_recognition);
    printf("| Servo Actuation Worst-Case Execution Time       | %8.2f ms |\n", wcet_servo_actuation);
    printf("| Servo Shoot Worst-Case Execution Time           | %8.2f ms |\n", wcet_servo_shoot);
    printf("| OverAll Response Worst-Case Execution Time      | %8.2f ms |\n", wcet_overall);
    printf("+--------------------------------------------------+------------+\n");
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

    // Cleanup actions
    mq_close(message_queue_instance);
    mq_unlink(CUSTOM_MQ_NAME);

#ifdef IS_RPI
    gpioTerminate();
#endif

    if (pthread_attr_destroy(&tasks[0].attribute) != 0)
        perror("attr destroy");
    if (pthread_attr_destroy(&tasks[1].attribute) != 0)
        perror("attr destroy");
    if (pthread_attr_destroy(&tasks[2].attribute) != 0)
        perror("attr destroy");

    sem_destroy(&semaphore_face_detect);
    sem_destroy(&semaphore_servo_actuator);
    sem_destroy(&semaphore_servo_shoot);

    printFinalTable();

#ifdef IS_RPI
    gpioTerminate();
#endif
}
