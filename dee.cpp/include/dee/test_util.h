#pragma once
// dee.cpp - minimal self-contained test harness (no external deps).
#include <cstdio>

namespace dee {
namespace test {

struct Summary {
    int passed = 0;
    int failed = 0;
};

inline Summary& summary() {
    static Summary s;
    return s;
}

inline void record(bool ok, const char* expr, const char* file, int line) {
    if (ok) {
        summary().passed++;
    } else {
        summary().failed++;
        std::printf("FAIL: %s\n  at %s:%d\n", expr, file, line);
    }
}

inline int report(const char* suite_name) {
    std::printf("\n==== %s ====\n", suite_name);
    std::printf("passed=%d failed=%d\n", summary().passed, summary().failed);
    return summary().failed == 0 ? 0 : 1;
}

}  // namespace test
}  // namespace dee

#define DEE_CHECK(cond) ::dee::test::record((cond), #cond, __FILE__, __LINE__)
#define DEE_CHECK_EQ(a, b) \
    ::dee::test::record((a) == (b), #a " == " #b, __FILE__, __LINE__)
#define DEE_CHECK_NE(a, b) \
    ::dee::test::record((a) != (b), #a " != " #b, __FILE__, __LINE__)
