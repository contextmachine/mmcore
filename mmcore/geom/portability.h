/* portability.h — bring in alloca() on most compilers/OSes */

#if defined(_MSC_VER)         /* Microsoft Visual C++ */
  #include <malloc.h>         /* declares _alloca() */
  #define alloca _alloca

#elif defined(__MINGW32__)     /* MinGW/gcc on Windows */
  #include <malloc.h>         /* _alloca, sometimes alloca() */
  /* capability: _alloca is available */

#elif defined(__GNUC__)        /* gcc or clang on Unix */
  /* gcc/clang provide __builtin_alloca; no header needed */
  #define alloca __builtin_alloca

#else                          /* fallback for other Unix-ish */
  #ifdef HAVE_ALLOCA_H
    #include <alloca.h>       /* POSIX-ish declaration */
  #else
    /* As a last resort, declare it yourself */
    void *alloca(size_t);
  #endif
#endif
