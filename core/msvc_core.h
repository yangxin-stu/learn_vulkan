#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#define log(x, ...) {std::cout << x << ": " << __VA_ARGS__;}
#define str(x) (#x)
// Predefined macros, depending on C/C++, compilation target, chosen compiler options
// 1. standard predifined macors
// __cplusplus, __DATE__, __TIME__, __FILE__, __LINE__
// __STDC__ , c compiler
// __STDCPP_THREADS__ c++ multi thread

// 2. MS
// __CHAR_UNSIGNED
// __CLR_VER
// __COUNTER__ when used increases by 1
// __cplusplus_cli, __cplusplus_winrt
// _DEBUG /LDd, /MDd, /MTd is set
// _OPENMP, _MT
// _WIN32 _WIN64
// _MSC_VER
void example() {
    printf("%s\n", __func__);
    printf("Function name: %s\n", __FUNCTION__);
    printf("Decorated function name: %s\n", __FUNCDNAME__);
    printf("Function signature: %s\n", __FUNCSIG__);
    // Function name: exampleFunction
    // Decorated function name: ?exampleFunction@@YAXXZ
    // Function signature: void __cdecl exampleFunction(void)
}

class ExampleClass {
    int m_nID;
public:
    // initialize object with a read-only unique ID
    ExampleClass(int nID): m_nID(nID) {}
    int GetID(void) { return m_nID; }
    // ExampleClass el {__COUNTER__};
};

// #pragma, __pragma, _Pragma  machine-specific or operating system-specific
// Some pragma directives provide the same functionality as compiler options
// eached in source code, it overrides the behavior specified by the compiler option

#pragma message("the #pragma way")
_Pragma ("message(\"the _Pragma way\")")

#define MY_ASSERT(BOOL_EXPRESSION) \
do { \
    _Pragma("warning(suppress: 4127)") /* C4127 conditional expression is constant */ \
    if (!(BOOL_EXPRESSION)) { \
        printf("MY_ASSERT FAILED: \"" #BOOL_EXPRESSION "\" on %s(%d)", __FILE__, __LINE__); \
        exit(-1); \
    } \
} while (0)






