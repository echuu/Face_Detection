# objdump -S test3 > test3.sym

HDRS = 
OBJS = boost.o
LIB = -lpthread -lrt -llua -ldl
CFLAGS = -g

all: boost

server1: $(OBJS)
	gcc $(CFLAGS) $(OBJS) -o boost $(LIB)

boost.o: boost.c $(HDRS)
	gcc $(CFLAGS) -c boost.c -o boost.o

clean:
	rm -f $(OBJS) target
