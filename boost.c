#include <stdio.h>
#include <math.h>
#include <tgmath.h>
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"


/* the Lua interpreter */
lua_State* L;

int luaFunc(int x, int y)
{
	int sum = 0;

	// lua function name -- testFunction.lua
	printf("before getglobal\n");
	lua_getglobal(L, "testFunction");

	// 1st arg
	lua_pushnumber(L, x);

	// 2nd arg
	lua_pushnumber(L, y);

	// call the function w/ 2 args, return 1 result
	printf("before lua_call()\n");
	lua_call(L, 2, 1);
	printf("after lua_call()\n");

	// get result, store in sum
	sum = (int)lua_tointeger(L, -1);
	lua_pop(L, 1);

	return sum;

}

int lua_adaboost(int iters)
{

	/* lua function name -- findWC.lua */
	return 0;


}


int main(int argc, char *argv[])
{
	int sum = 0;

	L = luaL_newstate();
	luaL_openlibs(L);
	luaL_dofile(L, "testFunction.lua");
	sum = luaFunc(10, 15);
	printf("The result is %d \n", sum);

	lua_close(L);

	printf("press enter to exit");

	getchar();

	return 0;


}