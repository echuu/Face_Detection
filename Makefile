# objdump -S test3 > test3.sym

INC_PATH = /home/eric/lua-5.3.4/src
LUAJIT_PATH = /home/eric/torch/install/include
HDRS = 
OBJS = boost.o
LIB_PATH = /home/eric/torch/install/lib
LIB = -lpthread -lrt -ldl -lm -lluajit
CFLAGS = -g -I$(LUAJIT_PATH)

all: target

target: $(OBJS)
	gcc $(CFLAGS) $(OBJS) -o target -L$(LIB_PATH) $(LIB)

boost.o: boost.c $(HDRS)
	gcc $(CFLAGS) -c boost.c -o boost.o

clean:
	rm -f $(OBJS) target
