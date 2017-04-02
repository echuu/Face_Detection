#include <stdio.h>
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"


/* the Lua interpreter */
lua_State* L;

int luaFunc(int x, int y)
{
	int sum;

	// lua function name -- testFunction.lua
	lua_getglobal(L, "testFunction");

	// 1st arg
	lua_pushnumber(L, x);

	// 2nd arg
	lua_pushnumber(L, y);

	// call the function w/ 2 args, return 1 result
	lua_call(L, 2, 1);

	// get result, store in sum
	sum = (int)lua_tointger(L, -1);
	lua_pop(L, 1);

	return sum;

}

int main(int argc, char *argv[])
{
	int sum;

	L = lua_open();
	luaL_dofile(L, "add.lua");
	sum = luaadd(10, 15);
	printf("The sum is %d \n", sum);

	lua_close(L);

	printf("press enter to exit");

	getchar();

	return 0;


}