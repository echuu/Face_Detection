#include <stdio.h>
#include <math.h>
#include <tgmath.h>
#include "lualib.h"
#include "lauxlib.h"
#include "luajit.h"


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

	int success = 0;

	/* lua function name -- findWC.lua */
	lua_getglobal(L, "findWC");

	// push first argument
	lua_pushnumber(L, iters);

	// call the functino with 2 args, 1 return result
	lua_call(L, 1, 1);
	success = (int) lua_tointeger(L, -1);

	return success;


}


int main(int argc, char *argv[])
{
	int T = 10;
	int success = 0;


	L = luaL_newstate();
	
	luaL_openlibs(L);

	printf("before call to findWC()\n");

	//luaL_dofile(L, "findWC.lua");
	//success = lua_adaboost(T);

	luaL_dofile(L, "testFunction.lua");
	success = luaFunc(10, 15);

	if (success) {
		printf("AdaBoost Success\n");
	}

	lua_close(L);

	printf("press enter to exit");

	getchar();

	return 0;
}