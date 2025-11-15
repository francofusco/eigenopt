#pragma once

// EigenOpt_QUADPROG_HIGHVIS_MSG() is meant for temporary debug statements.
// They can be helpful to pinpoint an issue, but they should later be removed
// from the code.
#define EigenOpt_QUADPROG_HIGHVIS_MSG(x) {std::cout << "\033[33m[DBG] " << x << "\033[m" << std::endl;}


#ifdef EigenOpt_QUADPROG_DEBUG_ON
  #include <iostream>

  // EigenOpt_QUADPROG_DBG() is meant for "long-term" debug statements, i.e.,
  // messages that will be left in the code even after testing.
  #ifndef EigenOpt_QUADPROG_SILENCE_DBG
    #define EigenOpt_QUADPROG_DBG(x) {std::cout << "[DBG] " << x << std::endl;}
  #else
    #define EigenOpt_QUADPROG_DBG(x) ;
  #endif

  namespace EigenOpt {
  namespace quadratic_programming {
  namespace internal {
    template<class T>
    std::string vec2str(const std::vector<T>& v) {
      std::stringstream ss;
      for(const auto& e : v)
        ss << " " << e;
      return ss.str();
    }
  } // namespace internal
  } // namespace quadratic_programming
  } // namespace EigenOpt

  #ifdef EigenOpt_QUADPROG_BREAKPOINTS_ON
    namespace EigenOpt {
    namespace quadratic_programming {
    namespace internal {
      void breakpoint() {
        std::string foo;
        std::cout << "Press enter to proceed ";
        std::cin >> foo;
      }
    } // namespace internal
    } // namespace quadratic_programming
    } // namespace EigenOpt
    #define EigenOpt_QUADPROG_BREAK EigenOpt::quadratic_programming::internal::breakpoint();
  #else
    #define EigenOpt_QUADPROG_BREAK ;
  #endif
#else
  #define EigenOpt_QUADPROG_DBG(x) ;
  #define EigenOpt_QUADPROG_BREAK ;
#endif
