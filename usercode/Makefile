LIB=libuser.so
CC=gcc
CCFLAGS=-fPIC -DDOUBLE_PRECISION
LDFLAGS=-shared

$(LIB): random.o irregularWaves.o uclib.o
	$(CC) $(CCFLAGS) $^ -o $(LIB) $(LDFLAGS)

%.o: %.c
	$(CC) $(CCFLAGS) -c $?

clean:
	rm *.o *.mod $(LIB)

