#pragma once
#include <stdio.h>

#if defined(__GNUC__)
#include "gcc_core.h"

#elif defined(__clang__)
#include "clang_core.h"

#elif defined (_WIN32) || defined (_WIN64)
#include "msvc_core.h"

#endif


// Demonstrates functionality of __FUNCTION__, __FUNCDNAME__, and __FUNCSIG__ macros
void exampleFunction()
{
    printf("Function name: %s\n", __FUNCTION__);
    printf("Decorated function name: %s\n", __FUNCDNAME__);
    printf("Function signature: %s\n", __FUNCSIG__);
// Sample Output
// -------------------------------------------------
// Function name: exampleFunction
// Decorated function name: ?exampleFunction@@YAXXZ
// Function signature: void __cdecl exampleFunction(void)
}

// VC: default: __cdecl; windows API __stdcall
// __stdcall: 参数从右向左压入堆栈；被调用函数自己恢复堆栈；函数名前导下划线后面跟着@和参数的大小 _@16func
// __cdecl: 右向左；调用者恢复堆栈，手动清栈；函数名前加前导下划线
// __fastcall: CPU寄存器传参,第一个和第二个DWORD 参数通过ecxhe edx，后面右向左压栈；被调用函数清栈
// __thiscall: 不能显示指定；C++类成员函数的缺省调用方式。参数个数确定类似于__stdcall，不定类似于__cdecl