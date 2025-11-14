#pragma once

#ifndef EigenOpt_SIMPLEX_DBG
#ifdef EigenOpt_SIMPLEX_DEBUG_ON
#include <iostream>
#define EigenOpt_SIMPLEX_DBG(x) {std::cout << "[DBG] " << x << std::endl;}
#else
#define EigenOpt_SIMPLEX_DBG(x) ;
#endif
#endif
