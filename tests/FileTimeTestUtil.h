#ifndef KYTY_TESTS_FILE_TIME_TEST_UTIL_H_
#define KYTY_TESTS_FILE_TIME_TEST_UTIL_H_

#include <chrono>
#include <cstdint>
#include <filesystem>

// Sets a host file's write time to `unix_ns` nanoseconds since the Unix epoch.
inline void SetUnixWriteTime(const std::filesystem::path& path, int64_t unix_ns) {
	const std::chrono::sys_time<std::chrono::nanoseconds> time {std::chrono::nanoseconds(unix_ns)};
#ifdef _MSVC_STL_VERSION // no file_clock::from_sys
	// MSVC's file clock starts in 1601. Keeping a nanosecond duration during the
	// epoch conversion overflows int64_t before a modern Unix time can be added.
	// Convert to its native 100 ns units first; exact 100 ns fixtures stay exact.
	const auto file_time = std::chrono::clock_cast<std::chrono::file_clock>(
	    std::chrono::floor<std::filesystem::file_time_type::duration>(time));
#else
	const auto file_time = std::chrono::file_clock::from_sys(time);
#endif
	std::filesystem::last_write_time(
	    path, std::chrono::floor<std::filesystem::file_time_type::duration>(file_time));
}

#endif // KYTY_TESTS_FILE_TIME_TEST_UTIL_H_
