// Shared PASS/FAIL reporting for the DirectLUSolvers test suites.
//
// Every test binary used to carry its own copy of `int g_failures` plus a
// `check()` that printed a differently-padded line. This centralizes both so a
// new test costs an include instead of a copy-paste, and so all suites produce
// the same, greppable output format.
//
// Usage:
//   #include "testing/Check.h"
//   using lu_testing::check;
//   ...
//   int main() { ...; return lu_testing::summarize("SupernodalLU"); }

#ifndef DIRECTLUSOLVERS_TEST_TESTING_CHECK_H
#define DIRECTLUSOLVERS_TEST_TESTING_CHECK_H

#include <chrono>
#include <cstdio>
#include <string>

namespace lu_testing {

using Clock = std::chrono::steady_clock;

// Process-wide failure counter. Function-local static so this stays header-only
// with no ODR trouble across translation units.
inline int& failureCount() {
  static int failures = 0;
  return failures;
}

// Width of the test-name column. Wide enough for the longest existing name.
constexpr int kNameWidth = 46;

// Report a check whose evidence is a number (a residual, an agreement, a ratio).
// `value` is printed whether or not the check passed -- a passing check's margin
// is just as informative as a failing one's.
inline bool check(bool ok, const char* name, double value) {
  std::printf("  [%s] %-*s %.3e\n", ok ? "PASS" : "FAIL", kNameWidth, name, value);
  if (!ok) ++failureCount();
  return ok;
}

inline bool check(bool ok, const std::string& name, double value) {
  return check(ok, name.c_str(), value);
}

// Report a check that is purely boolean (no meaningful numeric evidence).
inline bool checkTrue(bool ok, const char* name) {
  std::printf("  [%s] %-*s %s\n", ok ? "PASS" : "FAIL", kNameWidth, name, ok ? "ok" : "failed");
  if (!ok) ++failureCount();
  return ok;
}

inline bool checkTrue(bool ok, const std::string& name) { return checkTrue(ok, name.c_str()); }

// Record a failure that is not a check (a matrix failed to load, a solver threw).
inline void fail(const std::string& message) {
  std::printf("  [FAIL] %s\n", message.c_str());
  ++failureCount();
}

// Informational line, styled like a check but never counted.
inline void note(const std::string& message) { std::printf("        %s\n", message.c_str()); }

inline double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Final verdict + process exit code. Call as `return lu_testing::summarize(...)`.
inline int summarize(const char* suiteName) {
  const int failures = failureCount();
  std::printf("\n%s: %s (%d failure%s)\n", suiteName,
              failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED", failures,
              failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}

}  // namespace lu_testing

#endif  // DIRECTLUSOLVERS_TEST_TESTING_CHECK_H
