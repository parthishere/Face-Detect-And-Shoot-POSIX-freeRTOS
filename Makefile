INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS= 
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS) 
LDFLAGS= -Wl,-Map,output.map
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt -lpthread

HFILES= 
CFILES= facedetect.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

# Name of binary
MAIN = facedetect

ifeq ($(IS_JETSON_NANO),1)
    CFLAGS += -DIS_JETSON_NANO
endif

run: $(MAIN)
	@sudo ./$(MAIN) $(ARGS)

all: clean $(MAIN) run
	@echo Program Compiled

clean:
	-rm -f *.o *.d
	-rm -f $(MAIN) 

$(MAIN): facedetect.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
