#pragma once


#ifdef EigenOpt_QUADPROG_DEBUG_ON
  #include <iostream>
  #define EigenOpt_QUADPROG_DBG(x) {std::cout << "[DBG] " << x << std::endl;}

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
    #define EigenOpt_QUADPROG_BREAK EigenOpt::::quadratic_programming::internal::breakpoint();
  #else
    #define EigenOpt_QUADPROG_BREAK ;
  #endif
#else
  #define EigenOpt_QUADPROG_DBG(x) ;
  #define EigenOpt_QUADPROG_BREAK ;
#endif
