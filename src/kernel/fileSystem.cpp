#include "kernel/fileSystem.h"

#include "common/archive.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "kernel/memory.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/network.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <random>
#include <string_view>
#include <system_error>
#include <vector>

namespace Libs::LibKernel::FileSystem {

LIB_NAME("libkernel", "libkernel");

constexpr int      DESCRIPTOR_MIN = 3;
constexpr uint64_t DIR_BLOCK_SIZE = 512;

enum class SpecialFile {
	None,
	Random,
};

class MountPoints {
public:
	struct MountPair {
		std::filesystem::path dir;
		std::string           point;
		std::shared_ptr<Common::ArchiveReader> archive;
	};

	MountPoints() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	virtual ~MountPoints() = default;

	KYTY_CLASS_NO_COPY(MountPoints);

	void Mount(const std::filesystem::path& folder, const std::string& point);
	void Umount(const std::string& folder_or_point);

	[[nodiscard]] std::filesystem::path ResolvePath(const std::string& mounted_name);
	[[nodiscard]] bool                  IsMountPoint(const std::string& point);

private:
	std::vector<MountPair> m_mount_pairs;
	Common::Mutex          m_mutex;
};

struct File {
	Common::File                        f;
	std::string                         name;
	std::filesystem::path               real_name;
	std::atomic_bool                    opened;
	std::atomic_bool                    directory;
	std::atomic_bool                    readable;
	std::atomic_bool                    writable;
	std::atomic_bool                    append;
	std::atomic_bool                    sync_writes;
	std::atomic_bool                    close_on_exec;
	SpecialFile                         special;
	Common::Mutex                       mutex;
	std::vector<uint8_t>                dirents;
	uint64_t                            dents_offset;
};

class FileDescriptors {
public:
	FileDescriptors() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	virtual ~FileDescriptors() = default;

	KYTY_CLASS_NO_COPY(FileDescriptors);

	int   CreateDescriptor();
	void  DeleteDescriptor(int d);
	File* GetFile(int d);
	File* GetFile(const std::filesystem::path& real_name);
	void  CloseAll();

private:
	std::vector<File*> m_files;
	Common::Mutex      m_mutex;
};

static MountPoints*     g_mount_points = nullptr;
static FileDescriptors* g_files        = nullptr;

static bool IsRandomDevice(const std::string& path) {
	return path == "/dev/random" || path == "/dev/urandom";
}

static void FillRandomBuffer(void* buf, size_t nbytes) {
	auto*              out = reinterpret_cast<uint8_t*>(buf);
	std::random_device random;

	for (size_t i = 0; i < nbytes; i++) {
		out[i] = static_cast<uint8_t>(random());
	}
}

// Without times (directories, devices) the current time is reported.
static void SetStatTimes(FileStat* stat, std::optional<Common::File::Times> times) {
	if (!times) {
		const int64_t now =
		    std::chrono::nanoseconds(std::chrono::system_clock::now().time_since_epoch()).count();
		times = {now, now};
	}
	constexpr int64_t NANOS_PER_SEC = 1000000000;
	stat->st_atim     = {times->access_ns / NANOS_PER_SEC, times->access_ns % NANOS_PER_SEC};
	stat->st_mtim     = {times->write_ns / NANOS_PER_SEC, times->write_ns % NANOS_PER_SEC};
	stat->st_ctim     = stat->st_atim;
	stat->st_birthtim = stat->st_mtim;
}

// Directories report a 64 KiB st_blksize; games size one whole-directory getdents from it.
static void SetStatSize(FileStat* stat, uint64_t size, bool directory) {
	stat->st_size    = static_cast<int64_t>(size);
	stat->st_blksize = directory ? 0x10000 : 512;
	stat->st_blocks  = static_cast<int64_t>((size + 511) / 512);
}

static uint64_t AlignDown(uint64_t value, uint64_t alignment) {
	return (alignment != 0 ? value & ~(alignment - 1) : value);
}

static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
	return (alignment != 0 ? (value + alignment - 1) & ~(alignment - 1) : value);
}

static std::vector<uint8_t> PackDirents(const std::vector<Common::File::DirEntry>& entries) {
	std::vector<uint8_t> dirents;
	uint64_t             offset             = 0;
	uint64_t             last_reclen_offset = 0;
	for (const auto& entry: entries) {
		const auto& name = entry.name;
		EXIT_NOT_IMPLEMENTED(name.size() > 255);
		const auto reclen = AlignUp(8 + name.size() + 1, 4);
		if (offset + reclen > dirents.size()) {
			if (!dirents.empty()) {
				auto* last_reclen =
				    reinterpret_cast<uint16_t*>(dirents.data() + last_reclen_offset);
				*last_reclen += static_cast<uint16_t>(dirents.size() - offset);
			}
			offset = dirents.size();
			dirents.resize(offset + DIR_BLOCK_SIZE);
		}
		*reinterpret_cast<uint32_t*>(dirents.data() + offset) =
		    static_cast<uint32_t>(std::hash<std::string_view> {}(name));
		*reinterpret_cast<uint16_t*>(dirents.data() + offset + 4) = static_cast<uint16_t>(reclen);
		dirents[offset + 6]                                       = entry.is_file ? 8 : 4;
		dirents[offset + 7] = static_cast<uint8_t>(name.size());
		memcpy(dirents.data() + offset + 8, name.c_str(), name.size() + 1);
		last_reclen_offset = offset + 4;
		offset += reclen;
	}
	if (!dirents.empty()) {
		auto* last_reclen = reinterpret_cast<uint16_t*>(dirents.data() + last_reclen_offset);
		*last_reclen += static_cast<uint16_t>(dirents.size() - offset);
	}
	return dirents;
}

static uint64_t ReadDirectory(File* file, void* buf, uint64_t count) {
	const auto remaining =
	    file->dents_offset < file->dirents.size() ? file->dirents.size() - file->dents_offset : 0;
	const auto bytes = std::min(count, remaining);
	if (bytes != 0) {
		Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buf), bytes);
		memcpy(buf, file->dirents.data() + file->dents_offset, bytes);
		file->dents_offset += bytes;
	}
	return bytes;
}

static int MissingPathError(const std::filesystem::path& real_name);
static int MissingNodeError(const std::filesystem::path& real_name);
static int CreationParentError(const std::filesystem::path& real_name);

static int PosixToKernel(int posix_errno) {
	return KERNEL_ERROR_UNKNOWN + posix_errno;
}

int FileDescriptors::CreateDescriptor() {
	Common::LockGuard lock(m_mutex);

	auto* file        = new File {};
	file->opened      = false;
	file->directory   = false;
	file->readable    = false;
	file->writable    = false;
	file->append      = false;
	file->sync_writes = false;
	file->special     = SpecialFile::None;

	int files_num = static_cast<int>(m_files.size());
	for (int index = 0; index < files_num; index++) {
		if (m_files[index] == nullptr) {
			m_files[index] = file;
			return index + DESCRIPTOR_MIN;
		}
	}

	m_files.push_back(file);
	return static_cast<int>(m_files.size()) + DESCRIPTOR_MIN - 1;
}

void FileDescriptors::DeleteDescriptor(int d) {
	Common::LockGuard lock(m_mutex);

	auto index = static_cast<size_t>(d - DESCRIPTOR_MIN);

	EXIT_IF(index >= m_files.size());
	EXIT_IF(m_files[index] == nullptr);
	EXIT_IF(m_files[index]->opened);

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	// Close host files opened before a failed descriptor setup.
	m_files[index]->f.Close();
#endif

	delete m_files[index];
	m_files[index] = nullptr;
}

File* FileDescriptors::GetFile(int d) {
	Common::LockGuard lock(m_mutex);

	auto index = static_cast<size_t>(d - DESCRIPTOR_MIN);

	if (index >= m_files.size()) {
		return nullptr;
	}

	return m_files[index];
}

File* FileDescriptors::GetFile(const std::filesystem::path& real_name) {
	Common::LockGuard lock(m_mutex);

	for (auto* f: m_files) {
		if (f != nullptr && f->real_name == real_name) {
			return f;
		}
	}

	return nullptr;
}

void FileDescriptors::CloseAll() {
	Common::LockGuard lock(m_mutex);

	for (auto& f: m_files) {
		if (f != nullptr) {
			if (f->opened) {
				f->f.Close();
			}
			delete f;
			f = nullptr;
		}
	}
}

void MountPoints::Mount(const std::filesystem::path& folder, const std::string& point) {
	Common::LockGuard lock(m_mutex);

	auto point_str  = Common::FixDirectorySlash(point);

	Umount(point_str);

	MountPair p;
	p.dir   = folder;
	p.point = point_str;
	p.archive = Common::OpenArchive(folder);

	m_mount_pairs.push_back(std::move(p));
}

void MountPoints::Umount(const std::string& folder_or_point) {
	Common::LockGuard lock(m_mutex);

	auto folder_or_point_str = Common::FixDirectorySlash(folder_or_point);

	const auto it = std::find_if(
	    m_mount_pairs.begin(), m_mount_pairs.end(), [&folder_or_point_str](const MountPair& p) {
		    return Common::FixDirectorySlash(Common::PathToGenericString(p.dir)) == folder_or_point_str ||
		           p.point == folder_or_point_str;
	    });
	if (it != m_mount_pairs.end()) {
		m_mount_pairs.erase(it);
	}
}

// Guests join mount points and names with extra slashes (e.g. "/addcont0//file").
static std::string NormalizeGuestPath(const std::string& mounted_name) {
	std::string normalized = Common::FixFilenameSlash(mounted_name);
	normalized.erase(std::unique(normalized.begin(), normalized.end(),
	                             [](char a, char b) { return a == '/' && b == '/'; }),
	                 normalized.end());
	return normalized;
}

bool MountPoints::IsMountPoint(const std::string& point) {
	Common::LockGuard lock(m_mutex);

	return std::ranges::any_of(m_mount_pairs,
	                           [&point](const MountPair& p) { return p.point == point; });
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// Resolve guest paths case-insensitively on case-sensitive hosts.
static std::filesystem::path ResolvePathIgnoringCase(const std::filesystem::path& path) {
	std::error_code ec;
	if (std::filesystem::exists(path, ec)) {
		return path;
	}

	// Preserve unmatched components for the caller's ENOENT path.
	std::filesystem::path resolved =
	    path.has_root_path() ? path.root_path() : std::filesystem::path(".");
	bool matched = true;

	for (const auto& component: path.relative_path()) {
		if (component.empty()) {
			continue;
		}

		auto candidate = resolved / component;
		if (!matched || std::filesystem::exists(candidate, ec)) {
			resolved = std::move(candidate);
			continue;
		}

		bool found = false;
		for (std::filesystem::directory_iterator entry(resolved, ec), end; entry != end;
		     entry.increment(ec)) {
			if (ec) {
				break;
			}
			if (Common::EqualNoCase(entry->path().filename().string(), component.string())) {
				resolved = entry->path();
				found    = true;
				break;
			}
		}

		if (!found) {
			resolved = std::move(candidate);
			matched  = false;
		}
	}

	return resolved;
}
#else
static bool HasWindowsForbiddenFilenameCharacter(const std::string& relative_path) {
	return relative_path.find_first_of("<>:\"|?*") != std::string::npos;
}
#endif

std::filesystem::path MountPoints::ResolvePath(const std::string& mounted_name) {
	Common::LockGuard lock(m_mutex);

	const std::string normalized = NormalizeGuestPath(mounted_name);

	// Match the entire guest path so a mount root works with or without a trailing slash.
	const auto mounted_path = Common::FixDirectorySlash(normalized);
	const auto it           = std::find_if(
	    m_mount_pairs.begin(), m_mount_pairs.end(),
	    [&mounted_path](const MountPair& p) { return mounted_path.starts_with(p.point); });
	if (it != m_mount_pairs.end()) {
		const auto& p = *it;
		auto rel_path = Common::RemoveFirst(normalized, p.point.size());
		while (rel_path.starts_with('/')) {
			rel_path = Common::RemoveFirst(rel_path, 1);
		}

		const auto native_rel_path = Common::PathFromUtf8(rel_path);
		if (Common::IsArchivePath(p.dir)) {
			return p.dir / native_rel_path;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (HasWindowsForbiddenFilenameCharacter(rel_path)) {
			::printf("FileSystem: Windows-incompatible guest filename: %s\n", mounted_name.c_str());
		}
		return p.dir / native_rel_path;
#else
		return ResolvePathIgnoringCase(p.dir / native_rel_path);
#endif
	}

	// Unmounted guest paths must not fall through to the host filesystem.
	return {};
}

void Initialize() {
	g_mount_points = new MountPoints;
	g_files        = new FileDescriptors;
}

void EmergencyShutdown() {
	if (g_files != nullptr) {
		g_files->CloseAll();
	}
}

void Shutdown() {
	EmergencyShutdown();
	delete g_files;
	delete g_mount_points;
	g_files        = nullptr;
	g_mount_points = nullptr;
}

void Mount(const std::filesystem::path& folder, const std::string& point) {

	g_mount_points->Mount(folder, point);
}

void Umount(const std::string& folder_or_point) {

	g_mount_points->Umount(folder_or_point);
}

std::filesystem::path GetRealFilename(const std::string& mounted_file_name) {

	return g_mount_points->ResolvePath(mounted_file_name);
}

bool IsOpenDescriptor(int d) {
	const auto* file = g_files->GetFile(d);
	return file != nullptr && file->opened;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int KYTY_SYSV_ABI KernelOpen(const char* path, int flags, uint16_t mode) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto flags_u = static_cast<uint32_t>(flags);

	LOGF("\t path = %s\n"
	     "\t flags = %08" PRIx32 "\n"
	     "\t mode = %04" PRIx16 "\n",
	     path, flags_u, mode);

	bool append    = (flags_u & 0x0008u) != 0;
	bool fsync     = (flags_u & 0x0080u) != 0;
	bool sync      = (flags_u & 0x0080u) != 0;
	bool creat     = (flags_u & 0x0200u) != 0;
	bool trunc     = (flags_u & 0x0400u) != 0;
	bool excl      = (flags_u & 0x0800u) != 0;
	bool dsync     = (flags_u & 0x1000u) != 0;
	bool direct    = (flags_u & 0x00010000u) != 0;
	bool directory = (flags_u & 0x00020000u) != 0;

	if (direct) {
		LOGF("\t O_DIRECT ignored on host file backend\n");
	}

	flags_u &= 0x3u;

	Common::File::Mode rw_mode = Common::File::Mode::Read;

	switch (flags_u) {
		case 0: rw_mode = Common::File::Mode::Read; break;
		case 1: rw_mode = Common::File::Mode::Write; break;
		case 2: rw_mode = Common::File::Mode::ReadWrite; break;
		default: EXIT("invalid flag_u: %u\n", flags_u);
	}

	EXIT_NOT_IMPLEMENTED(directory && rw_mode != Common::File::Mode::Read);
	EXIT_NOT_IMPLEMENTED(directory && (trunc || creat));

	int   descriptor = g_files->CreateDescriptor();
	auto* file       = g_files->GetFile(descriptor);

	EXIT_IF(file == nullptr || file->opened || file->directory);

	file->name          = path;
	file->readable      = rw_mode != Common::File::Mode::Write;
	file->writable      = rw_mode != Common::File::Mode::Read;
	file->append        = append;
	file->sync_writes   = fsync || sync || dsync;
	file->close_on_exec = (static_cast<uint32_t>(flags) & 0x00100000u) != 0;

	if (!directory && IsRandomDevice(file->name)) {
		file->real_name = file->name;
		file->special   = SpecialFile::Random;
		file->opened    = true;

		LOGF_COLOR(Log::Color::Green, "\tOpen device: %s, [ok]\n",
		           Common::PathToString(file->real_name).c_str());

		return descriptor;
	}

	file->real_name = g_mount_points->ResolvePath(file->name);
	if (file->real_name.empty()) {
		g_files->DeleteDescriptor(descriptor);
		return KERNEL_ERROR_ENOENT;
	}
	if (Common::IsArchivePath(file->real_name) &&
	    (rw_mode != Common::File::Mode::Read || trunc || creat)) {
		g_files->DeleteDescriptor(descriptor);
		return KERNEL_ERROR_EROFS;
	}

	if (trunc && rw_mode == Common::File::Mode::Read) {
		g_files->DeleteDescriptor(descriptor);
		return KERNEL_ERROR_EACCES;
	}

	const auto info = Common::File::GetInfo(file->real_name);
	const bool dir_exist = info && !info->is_file;
	const bool file_exist = info && info->is_file;

	// A missing path opened without O_CREAT is ENOENT, or ENOTDIR below a non-directory.
	if (!creat && !dir_exist && !file_exist) {
		// DeleteDescriptor frees `file`; classify the path first.
		const int error = MissingNodeError(file->real_name);
		g_files->DeleteDescriptor(descriptor);
		return error;
	}
	// O_CREAT creates only the last component (POSIX); its parent must be a directory.
	if (creat && !dir_exist && !file_exist) {
		if (const int error = CreationParentError(file->real_name); error != OK) {
			g_files->DeleteDescriptor(descriptor);
			return error;
		}
	}

	if (directory || dir_exist) {
		if (!dir_exist) {
			g_files->DeleteDescriptor(descriptor);
			return KERNEL_ERROR_ENOTDIR;
		}

		EXIT_NOT_IMPLEMENTED(!directory && rw_mode != Common::File::Mode::Read);
		EXIT_NOT_IMPLEMENTED(!directory && (trunc || creat));

		const auto entries = Common::File::GetDirEntries(file->real_name);
		file->dirents      = PackDirents(entries);
		file->dents_offset = 0;
		file->directory    = true;

		LOGF_COLOR(Log::Color::Green, "\tOpen dir: %s, entries = %" PRIu32 ", [ok]\n",
		           Common::PathToString(file->real_name).c_str(),
		           static_cast<uint32_t>(entries.size()));

		for (const auto& f: entries) {
			LOGF("\t\t%s %s\n", f.is_file ? "[file]" : "[dir ]", f.name.c_str());
		}
	} else {
		bool result = false;

		if (excl && creat && file_exist) {
			g_files->DeleteDescriptor(descriptor);
			return KERNEL_ERROR_EEXIST;
		}

		if (creat && (!file_exist || trunc)) {
			result = file->f.Create(file->real_name);

			LOGF_COLOR(result ? Log::Color::Green : Log::Color::Red, "\tCreate: %s, %s\n",
			           Common::PathToString(file->real_name).c_str(), result ? "[ok]" : "[fail]");
		} else {
			result = file->f.Open(file->real_name, rw_mode);

			LOGF_COLOR(result ? Log::Color::Green : Log::Color::Red, "\tOpen: %s, %s\n",
			           Common::PathToString(file->real_name).c_str(), result ? "[ok]" : "[fail]");
		}

		if (result && trunc) {
			result = file->f.Truncate(0);
		}

		if (result && append) {
			file->f.Seek(file->f.Size());
		}

		if (!result || file->f.IsInvalid()) {
			g_files->DeleteDescriptor(descriptor);
			return KERNEL_ERROR_EACCES;
		}
	}

	file->opened = true;
	return descriptor;
}

int KYTY_SYSV_ABI KernelClose(int d) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (::Libs::Network::Net::IsSocket(d)) {
		const auto result = ::Libs::Network::Net::SocketClose(d);
		return (result == OK ? OK : PosixToKernel(::Libs::Network::NetToPosix(result)));
	}

	auto* file = g_files->GetFile(d);

	if (file == nullptr) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_IF(!file->opened);

	if (!file->directory && file->special == SpecialFile::None) {
		file->f.Close();
	}

	file->opened = false;

	LOGF("\tClose: %s\n", Common::PathToString(file->real_name).c_str());

	g_files->DeleteDescriptor(d);

	return OK;
}

int KYTY_SYSV_ABI KernelFcntl(int d, int command, int arg) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	auto* file = g_files->GetFile(d);
	if (file == nullptr || !file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	switch (command) {
		case 1: return file->close_on_exec.load(std::memory_order_relaxed); // F_GETFD
		case 2: // F_SETFD, FD_CLOEXEC
			file->close_on_exec.store((arg & 1) != 0, std::memory_order_relaxed);
			return OK;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int64_t KYTY_SYSV_ABI KernelRead(int d, void* buf, size_t nbytes) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (::Libs::Network::Net::IsSocket(d)) {
		const auto result = ::Libs::Network::Net::Recv(d, buf, nbytes, 0);
		return (result >= 0 ? result : PosixToKernel(*Posix::GetErrorAddr()));
	}

	auto* file = g_files->GetFile(d);

	if (file == nullptr) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_IF(!file->opened);

	if (nbytes > INT_MAX) {
		return KERNEL_ERROR_EINVAL;
	}
	if (file->directory) {
		Common::LockGuard lock(file->mutex);
		const auto        bytes = ReadDirectory(file, buf, nbytes);
		LOGF("\tRead %" PRIu64 " directory bytes from: %s\n", bytes,
		     Common::PathToString(file->real_name).c_str());
		return static_cast<int64_t>(bytes);
	}

	if (file->special == SpecialFile::Random) {
		FillRandomBuffer(buf, nbytes);

		LOGF("\tRead %" PRIu64 " random bytes from: %s\n", static_cast<uint64_t>(nbytes),
		     Common::PathToString(file->real_name).c_str());

		return static_cast<int64_t>(nbytes);
	}

	file->mutex.Lock();

	bool       is_invalid = file->f.IsInvalid();
	const auto pos        = file->f.Tell();
	const auto file_size  = file->f.Size();
	const auto remaining  = pos < file_size ? file_size - pos : 0;
	Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buf),
	                         std::min<uint64_t>(nbytes, remaining));
	uint32_t bytes_read = 0;
	file->f.Read(buf, static_cast<uint32_t>(nbytes), &bytes_read);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileReadBytes, bytes_read);

	file->mutex.Unlock();

	if (is_invalid) {
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tRead %u bytes from: %s\n", bytes_read, Common::PathToString(file->real_name).c_str());

	return bytes_read;
}

// Adapted from Jetsku/KytyPS5 01d7cd0a38dfc964cfca5cd583c7b8fdc99d8681 (GPL-2.0).
// Host file I/O cannot enter the user-mode GPU fault handler when it encounters a watched
// guest page. After a short write, copy a bounded readable guest span through that handler,
// then resume at the exact unwritten byte. Never bypass the GPU tracker by reading backing.
static void WriteFromGuest(Common::File& file, const void* data, uint32_t size,
                           uint32_t* bytes_written) {
	constexpr uint32_t Chunk = 1u << 20u;
	thread_local std::vector<uint8_t> bounce;
	const auto base = reinterpret_cast<uintptr_t>(data);
	if (size != 0 && size - 1u > UINTPTR_MAX - base) {
		if (bytes_written != nullptr) *bytes_written = 0;
		return;
	}
	uint32_t total = 0;
	while (total < size) {
		const auto* source = reinterpret_cast<const uint8_t*>(base + total);
		uint32_t written = 0;
		file.Write(source, size - total, &written);
		total += written;
		if (total == size) break;

		const auto address = base + total;
		source = reinterpret_cast<const uint8_t*>(address);
		Memory::VirtualQueryInfo mapping {};
		if (Memory::KernelVirtualQuery(source, 0, &mapping, sizeof(mapping)) != OK ||
		    !mapping.is_committed || (mapping.protection & 1) == 0 ||
		    mapping.start > address || mapping.end <= address) {
			break;
		}
		const auto count = static_cast<uint32_t>(
		    std::min<uint64_t>({size - total, Chunk, mapping.end - address}));
		bounce.resize(count);
		std::memcpy(bounce.data(), source, count);
		written = 0;
		file.Write(bounce.data(), count, &written);
		total += written;
		if (written < count) break;
	}
	if (bytes_written != nullptr) *bytes_written = total;
}

int64_t KYTY_SYSV_ABI KernelWrite(int d, const void* buf, size_t nbytes) {
	PRINT_NAME();

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (d == 1 || d == 2) {
		auto*      out      = (d == 1 ? stdout : stderr);
		const auto written  = std::fwrite(buf, 1, nbytes, out);
		const auto flush_ok = std::fflush(out) == 0;

		if (written != nbytes || !flush_ok) {
			return KERNEL_ERROR_EIO;
		}

		return static_cast<int64_t>(written);
	}

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	if (::Libs::Network::Net::IsSocket(d)) {
		const auto result = ::Libs::Network::Net::Send(d, buf, nbytes, 0);
		return (result >= 0 ? result : PosixToKernel(*Posix::GetErrorAddr()));
	}

	auto* file = g_files->GetFile(d);

	if (file == nullptr || !file->opened || !file->writable) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_NOT_IMPLEMENTED(file->directory);
	EXIT_NOT_IMPLEMENTED(file->special != SpecialFile::None);

	EXIT_IF(!file->opened);

	EXIT_NOT_IMPLEMENTED(nbytes > UINT_MAX);

	file->mutex.Lock();

	bool     is_invalid    = file->f.IsInvalid();
	uint32_t bytes_written = 0;
	if (file->append) {
		file->f.Seek(file->f.Size());
	}
	WriteFromGuest(file->f, buf, static_cast<uint32_t>(nbytes), &bytes_written);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileWriteBytes, bytes_written);
	if (file->sync_writes) {
		file->f.Flush();
	}

	file->mutex.Unlock();

	if (is_invalid) {
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tWrite %u bytes to: %s\n", bytes_written, Common::PathToString(file->real_name).c_str());

	return bytes_written;
}

int64_t KYTY_SYSV_ABI KernelPread(int d, void* buf, size_t nbytes, int64_t offset) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (offset < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	auto* file = g_files->GetFile(d);

	if (file == nullptr) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_NOT_IMPLEMENTED(file->directory);

	EXIT_IF(!file->opened);

	EXIT_NOT_IMPLEMENTED(nbytes > UINT_MAX);

	if (file->special == SpecialFile::Random) {
		FillRandomBuffer(buf, nbytes);

		LOGF("\tRead %" PRIu64 " random bytes (pos = %" PRId64 ") from: %s\n",
		     static_cast<uint64_t>(nbytes), offset, Common::PathToString(file->real_name).c_str());

		return static_cast<int64_t>(nbytes);
	}

	file->mutex.Lock();

	bool       is_invalid = file->f.IsInvalid();
	auto       pos        = file->f.Tell();
	const auto file_size  = file->f.Size();
	const auto remaining =
	    static_cast<uint64_t>(offset) < file_size ? file_size - static_cast<uint64_t>(offset) : 0;
	Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buf),
	                         std::min<uint64_t>(nbytes, remaining));
	uint32_t bytes_read = 0;
	file->f.Seek(offset);
	file->f.Read(buf, static_cast<uint32_t>(nbytes), &bytes_read);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileReadBytes, bytes_read);
	file->f.Seek(pos);

	file->mutex.Unlock();

	if (is_invalid) {
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tRead %u bytes (pos = %" PRId64 ") from: %s\n", bytes_read, offset,
	     Common::PathToString(file->real_name).c_str());

	return bytes_read;
}

static int ValidateIovecs(const KernelIovec* iov, int iovcnt, int64_t offset,
                          std::vector<KernelIovec>* buffers, size_t* total) {
	constexpr int MaxIov = 1024;
	if (iovcnt < 0 || iovcnt > MaxIov || offset < 0) {
		return KERNEL_ERROR_EINVAL;
	}
	if (iov == nullptr && iovcnt != 0) {
		return KERNEL_ERROR_EFAULT;
	}

	if (iovcnt != 0) {
		buffers->assign(iov, iov + iovcnt);
	}
	*total = 0;
	for (const auto& buffer: *buffers) {
		if (buffer.iov_len > INT_MAX - *total) {
			return KERNEL_ERROR_EINVAL;
		}
		if (buffer.iov_base == nullptr && buffer.iov_len != 0) {
			return KERNEL_ERROR_EFAULT;
		}
		*total += buffer.iov_len;
	}
	return OK;
}

int64_t KYTY_SYSV_ABI KernelPreadv(int d, const KernelIovec* iov, int iovcnt, int64_t offset) {
	PRINT_NAME();

	std::vector<KernelIovec> buffers;
	size_t                   total = 0;
	const int                error = ValidateIovecs(iov, iovcnt, offset, &buffers, &total);
	if (error != OK) {
		return error;
	}

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}
	if (::Libs::Network::Net::IsSocket(d)) {
		return KERNEL_ERROR_ESPIPE;
	}
	auto* file = g_files->GetFile(d);
	if (file == nullptr || !file->opened || !file->readable) {
		return KERNEL_ERROR_EBADF;
	}

	Common::LockGuard lock(file->mutex);
	if (file->directory) {
		return KERNEL_ERROR_EISDIR;
	}
	if (total == 0) {
		return 0;
	}
	if (file->special == SpecialFile::Random) {
		for (const auto& buffer: buffers) {
			if (buffer.iov_len != 0) {
				Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buffer.iov_base),
				                         buffer.iov_len);
				FillRandomBuffer(buffer.iov_base, buffer.iov_len);
			}
		}
		return static_cast<int64_t>(total);
	}
	if (file->f.IsInvalid()) {
		return KERNEL_ERROR_EIO;
	}

	const auto position  = file->f.Tell();
	const auto file_size = file->f.Size();
	if (!file->f.Seek(static_cast<uint64_t>(offset))) {
		return KERNEL_ERROR_EIO;
	}
	auto    remaining  = static_cast<uint64_t>(offset) < file_size ? file_size - offset : 0;
	int64_t bytes_read = 0;
	for (const auto& buffer: buffers) {
		if (remaining == 0) {
			break;
		}
		if (buffer.iov_len == 0) {
			continue;
		}
		const auto count = static_cast<uint32_t>(std::min<uint64_t>(buffer.iov_len, remaining));
		Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buffer.iov_base), count);
		uint32_t bytes = 0;
		file->f.Read(buffer.iov_base, count, &bytes);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileReadBytes, bytes);
		bytes_read += bytes;
		remaining -= bytes;
		if (bytes < count) {
			break;
		}
	}
	if (!file->f.Seek(position)) {
		return KERNEL_ERROR_EIO;
	}
	LOGF("\tReadv %" PRId64 " bytes (pos = %" PRId64 ", iovcnt = %d) from: %s\n", bytes_read,
	     offset, iovcnt, Common::PathToString(file->real_name).c_str());
	return bytes_read;
}

int64_t KYTY_SYSV_ABI KernelPwrite(int d, const void* buf, size_t nbytes, int64_t offset) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (offset < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	auto* file = g_files->GetFile(d);

	if (file == nullptr || !file->opened || !file->writable) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_NOT_IMPLEMENTED(file->directory);
	EXIT_NOT_IMPLEMENTED(file->special != SpecialFile::None);

	EXIT_IF(!file->opened);

	EXIT_NOT_IMPLEMENTED(nbytes > UINT_MAX);

	file->mutex.Lock();

	bool     is_invalid    = file->f.IsInvalid();
	auto     pos           = file->f.Tell();
	uint32_t bytes_written = 0;
	file->f.Seek(file->append ? file->f.Size() : static_cast<uint64_t>(offset));
	WriteFromGuest(file->f, buf, static_cast<uint32_t>(nbytes), &bytes_written);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileWriteBytes, bytes_written);
	if (file->sync_writes) {
		file->f.Flush();
	}
	file->f.Seek(pos);

	file->mutex.Unlock();

	if (is_invalid) {
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tWrite %u bytes (pos = %" PRId64 ") to: %s\n", bytes_written, offset,
	     Common::PathToString(file->real_name).c_str());

	return bytes_written;
}

static int64_t Writev(int d, const KernelIovec* iov, int iovcnt, std::optional<int64_t> offset) {
	std::vector<KernelIovec> buffers;
	size_t                   total = 0;
	const int error = ValidateIovecs(iov, iovcnt, offset.value_or(0), &buffers, &total);
	if (error != OK) {
		return error;
	}

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}
	if (::Libs::Network::Net::IsSocket(d)) {
		return KERNEL_ERROR_ESPIPE;
	}
	auto* file = g_files->GetFile(d);
	if (file == nullptr || !file->opened || !file->writable) {
		return KERNEL_ERROR_EBADF;
	}
	Common::LockGuard lock(file->mutex);
	if (file->directory) {
		return KERNEL_ERROR_EISDIR;
	}
	if (file->special != SpecialFile::None) {
		return KERNEL_ERROR_EINVAL;
	}
	if (total == 0) {
		return 0;
	}
	if (file->f.IsInvalid()) {
		return KERNEL_ERROR_EIO;
	}

	const auto position = file->f.Tell();
	auto       target   = position;
	if (file->append) {
		target = file->f.Size();
	} else if (offset) {
		target = static_cast<uint64_t>(*offset);
	}
	if (target > static_cast<uint64_t>(INT64_MAX) - total) {
		return KERNEL_ERROR_EFBIG;
	}
	if ((offset || file->append) && !file->f.Seek(target)) {
		return KERNEL_ERROR_EIO;
	}

	int64_t bytes_written = 0;
	for (const auto& buffer: buffers) {
		if (buffer.iov_len == 0) {
			continue;
		}
		uint32_t bytes = 0;
		WriteFromGuest(file->f, buffer.iov_base, static_cast<uint32_t>(buffer.iov_len), &bytes);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileWriteBytes, bytes);
		bytes_written += bytes;
		if (bytes < buffer.iov_len) {
			break;
		}
	}
	const bool flushed  = !file->sync_writes || file->f.Flush();
	const bool restored = !offset || file->f.Seek(position);
	if (!flushed || !restored) {
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tWritev %" PRId64 " bytes (pos = %" PRIu64 ", iovcnt = %d) to: %s\n", bytes_written,
	     target, iovcnt, Common::PathToString(file->real_name).c_str());
	return bytes_written;
}

int64_t KYTY_SYSV_ABI KernelWritev(int d, const KernelIovec* iov, int iovcnt) {
	PRINT_NAME();
	return Writev(d, iov, iovcnt, std::nullopt);
}

int64_t KYTY_SYSV_ABI KernelPwritev(int d, const KernelIovec* iov, int iovcnt, int64_t offset) {
	PRINT_NAME();
	return Writev(d, iov, iovcnt, offset);
}

int64_t KYTY_SYSV_ABI KernelLseek(int d, int64_t offset, int whence) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	auto* file = g_files->GetFile(d);

	if (file == nullptr) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_IF(!file->opened);

	if (file->special != SpecialFile::None) {
		return KERNEL_ERROR_ESPIPE;
	}

	Common::LockGuard lock(file->mutex);

	if (!file->directory && file->f.IsInvalid()) {
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	uint64_t base = 0;
	switch (whence) {
		case 0: break;
		case 1: base = file->directory ? file->dents_offset : file->f.Tell(); break;
		case 2: base = file->directory ? file->dirents.size() : file->f.Size(); break;
		default: return KERNEL_ERROR_EINVAL;
	}

	uint64_t position = 0;
	if (offset >= 0) {
		if (base > static_cast<uint64_t>(INT64_MAX - offset)) {
			return KERNEL_ERROR_EOVERFLOW;
		}
		position = base + static_cast<uint64_t>(offset);
	} else {
		const auto distance = uint64_t {0} - static_cast<uint64_t>(offset);
		if (base < distance) {
			return KERNEL_ERROR_EINVAL;
		}
		position = base - distance;
		if (position > INT64_MAX) {
			return KERNEL_ERROR_EOVERFLOW;
		}
	}

	if (file->directory) {
		file->dents_offset = position;
	} else {
		file->f.Seek(position);
		EXIT_IF(file->f.Tell() != position);
	}

	LOGF("\tLseek (pos = %" PRIu64 ") to: %s\n", position,
	     Common::PathToString(file->real_name).c_str());

	return static_cast<int64_t>(position);
}

int KYTY_SYSV_ABI KernelStat(const char* path, FileStat* sb) {
	PRINT_NAME();

	if (path == nullptr || sb == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	LOGF("\t KernelStat: %s\n", path);

	// The guest root holds only mount points and has no host directory.
	if (std::strcmp(path, "/") == 0) {
		*sb = {.st_mode = 0040777u};
		return OK;
	}

	return StatHostPath(g_mount_points->ResolvePath(path), sb);
}

int StatHostPath(const std::filesystem::path& real_file_name, FileStat* sb) {
	const auto info = Common::File::GetInfo(real_file_name);
	if (!info) {
		LOGF("\t file not found\n");
		return MissingNodeError(real_file_name);
	}

	const bool is_dir = !info->is_file;

	FileStat stat {};
	stat.st_mode = 0000777u | (is_dir ? 0040000u : 0100000u);
	const uint64_t size =
	    is_dir ? PackDirents(Common::File::GetDirEntries(real_file_name)).size() : info->size;
	SetStatSize(&stat, size, is_dir);
	SetStatTimes(&stat, is_dir ? std::nullopt : Common::File::GetTimes(real_file_name));
	*sb = stat;

	return OK;
}

int KYTY_SYSV_ABI KernelFstat(int d, FileStat* sb) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (sb == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	auto* file = g_files->GetFile(d);

	if (file == nullptr) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_IF(!file->opened);

	LOGF("\tKernelFstat: %s\n", Common::PathToString(file->real_name).c_str());

	FileStat stat {};
	stat.st_mode = 0000777u | (file->directory ? 0040000u : 0100000u);

	std::optional<Common::File::Times> times;

	if (file->special != SpecialFile::None) {
		stat.st_mode = 0020644u;
		SetStatSize(&stat, 0, false);
	} else if (!file->directory) {
		file->mutex.Lock();

		bool is_invalid = file->f.IsInvalid();
		auto size       = file->f.Size();
		times           = file->f.GetTimes();

		file->mutex.Unlock();

		if (is_invalid) {
			LOGF("\tfile is invalid\n");
			return KERNEL_ERROR_EIO;
		}

		SetStatSize(&stat, size, false);
	} else {
		SetStatSize(&stat, file->dirents.size(), true);
	}

	SetStatTimes(&stat, times);
	*sb = stat;

	return OK;
}

static int TruncatePathError(const std::error_code& error) {
	switch (static_cast<std::errc>(error.default_error_condition().value())) {
		case std::errc::no_such_file_or_directory: return KERNEL_ERROR_ENOENT;
		case std::errc::not_a_directory: return KERNEL_ERROR_ENOTDIR;
		case std::errc::filename_too_long: return KERNEL_ERROR_ENAMETOOLONG;
		case std::errc::permission_denied: return KERNEL_ERROR_EACCES;
		case std::errc::read_only_file_system: return KERNEL_ERROR_EROFS;
		case std::errc::file_too_large: return KERNEL_ERROR_EFBIG;
		case std::errc::invalid_argument: return KERNEL_ERROR_EINVAL;
		case std::errc::no_space_on_device: return KERNEL_ERROR_ENOSPC;
		default: return KERNEL_ERROR_EIO;
	}
}

// Windows reports a missing path for "file/child", hiding the guest's ENOTDIR.
// Missing ancestors may also set error while reporting file_type::not_found.
static int MissingPathError(const std::filesystem::path& real_name) {
	for (auto parent = real_name.parent_path(); !parent.empty() && parent != parent.root_path();
	     parent = parent.parent_path()) {
		std::error_code error;
		const auto status = std::filesystem::status(parent, error);
		if (error) {
			if (error != std::errc::no_such_file_or_directory && error != std::errc::not_a_directory) {
				return TruncatePathError(error);
			}
		} else if (std::filesystem::exists(status)) {
			return std::filesystem::is_directory(status) ? KERNEL_ERROR_ENOENT
			                                             : KERNEL_ERROR_ENOTDIR;
		}
		// The walk is lexical; it never expands symlink targets or repeats a root.
		if (parent == parent.parent_path()) {
			break;
		}
	}
	return KERNEL_ERROR_ENOENT;
}

// A guest path that does not exist: ENOENT, or ENOTDIR when a prefix component is a
// non-directory. Unmounted (empty) and read-only archive paths keep ENOENT.
static int MissingNodeError(const std::filesystem::path& real_name) {
	if (real_name.empty() || Common::IsArchivePath(real_name)) {
		return KERNEL_ERROR_ENOENT;
	}
	return MissingPathError(real_name);
}

// The last component is about to be created: its parent must exist as a directory.
static int CreationParentError(const std::filesystem::path& real_name) {
	if (real_name.empty()) {
		return KERNEL_ERROR_ENOENT;
	}
	if (Common::IsArchivePath(real_name)) {
		return OK; // Archive mutations are rejected with EROFS by the callers.
	}
	const auto parent = real_name.parent_path();
	if (parent.empty()) {
		return OK;
	}
	std::error_code error;
	const auto      status = std::filesystem::status(parent, error);
	if (!error && std::filesystem::is_directory(status)) {
		return OK;
	}
	if (!error && std::filesystem::exists(status)) {
		return KERNEL_ERROR_ENOTDIR;
	}
	return MissingPathError(real_name);
}

int KYTY_SYSV_ABI KernelTruncate(const char* path, int64_t length) {
	PRINT_NAME();
	if (path == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}
	if (path[0] != '/' || length < 0) {
		return KERNEL_ERROR_EINVAL;
	}
	size_t component_length = 0;
	for (size_t i = 0; path[i] != '\0'; ++i) {
		if (i >= 1023) {
			return KERNEL_ERROR_ENAMETOOLONG;
		}
		if (path[i] == '/') {
			component_length = 0;
		} else if (++component_length > 255) {
			return KERNEL_ERROR_ENAMETOOLONG;
		}
	}
	if (IsRandomDevice(path)) {
		return OK;
	}
	const auto real_name = g_mount_points->ResolvePath(path);
	if (real_name.empty()) {
		return KERNEL_ERROR_ENOENT;
	}
	if (Common::IsArchivePath(real_name)) {
		return KERNEL_ERROR_EROFS;
	}

	std::error_code error;
	const auto      status = std::filesystem::status(real_name, error);
	if (!error) {
		if (!std::filesystem::exists(status)) {
			return MissingPathError(real_name);
		}
		if (std::filesystem::is_directory(status)) {
			return KERNEL_ERROR_EISDIR;
		}
		if (std::filesystem::is_regular_file(status)) {
			std::filesystem::resize_file(real_name, static_cast<uint64_t>(length), error);
		}
	}
	if (error) {
		if (error == std::errc::no_such_file_or_directory) {
			return MissingPathError(real_name);
		}
		return TruncatePathError(error);
	}
	LOGF("\tTruncate (size = %" PRId64 ") file: %s\n", length,
	     Common::PathToString(real_name).c_str());
	return OK;
}

int KYTY_SYSV_ABI KernelFtruncate(int d, int64_t length) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	if (length < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	if (::Libs::Network::Net::IsSocket(d)) {
		return KERNEL_ERROR_EINVAL;
	}

	auto* file = g_files->GetFile(d);

	if (file == nullptr || !file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	if (!file->writable) {
		return KERNEL_ERROR_EBADF;
	}

	if (file->directory || file->special != SpecialFile::None) {
		return KERNEL_ERROR_EINVAL;
	}

	Common::LockGuard lock(file->mutex);

	if (file->f.IsInvalid() || !file->f.Truncate(static_cast<uint64_t>(length))) {
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tFtruncate (size = %" PRId64 ") file: %s\n", length,
	     Common::PathToString(file->real_name).c_str());

	return OK;
}

int KYTY_SYSV_ABI KernelFsync(int d) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	auto* file = g_files->GetFile(d);
	if (file == nullptr || !file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	// Directories, devices and read-only descriptors have no written data to flush
	if (!file->writable || file->special != SpecialFile::None) {
		return OK;
	}

	Common::LockGuard lock(file->mutex);
	if (!file->f.Flush()) {
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tFsync: %s\n", Common::PathToString(file->real_name).c_str());
	return OK;
}

int KYTY_SYSV_ABI KernelUnlink(const char* path) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto real_file_name = g_mount_points->ResolvePath(path);
	if (Common::IsArchivePath(real_file_name)) {
		return KERNEL_ERROR_EROFS;
	}

	const auto info = Common::File::GetInfo(real_file_name);
	if (!info) {
		return MissingNodeError(real_file_name);
	}

	if (!info->is_file) {
		return KERNEL_ERROR_EPERM;
	}

	auto* open_file = g_files->GetFile(real_file_name);
	bool  ok        = (open_file != nullptr && open_file->opened && !open_file->directory
	                       ? open_file->f.Unlink()
	                       : Common::File::DeleteFile(real_file_name));

	if (!ok) {
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tKernelUnlink: %s\n", path);

	return OK;
}

int KYTY_SYSV_ABI KernelRename(const char* from, const char* to) {
	PRINT_NAME();

	if (from == nullptr || to == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto from_path = std::string(from);
	auto to_path   = std::string(to);
	auto real_from = g_mount_points->ResolvePath(from_path);
	auto real_to   = g_mount_points->ResolvePath(to_path);
	if (Common::IsArchivePath(real_from) || Common::IsArchivePath(real_to)) {
		return KERNEL_ERROR_EROFS;
	}
	// After the archive check (EROFS keeps priority) and before any host change: the same
	// rules as KernelRmdir. Terminal "." / ".." forms are EINVAL (they could alias a mount
	// root that lexical matching misses); a mount root is never moved or replaced (EBUSY),
	// so g_mount_points can never point at a moved host directory.
	for (const auto* guest : {&from_path, &to_path}) {
		const auto normalized = Common::FixDirectorySlash(NormalizeGuestPath(*guest));
		if (normalized.ends_with("/./") || normalized.ends_with("/../")) {
			return KERNEL_ERROR_EINVAL;
		}
		if (g_mount_points->IsMountPoint(normalized)) {
			return KERNEL_ERROR_EBUSY;
		}
	}

	if (real_from.empty() || real_to.empty()) {
		return KERNEL_ERROR_ENOENT;
	}
	const auto from_info = Common::File::GetInfo(real_from);
	if (!from_info) {
		return MissingNodeError(real_from);
	}
	// POSIX: a directory cannot move below itself (EINVAL), whether or not the target exists.
	if (!from_info->is_file) {
		std::error_code from_error;
		std::error_code to_error;
		const auto      from_full = std::filesystem::weakly_canonical(real_from, from_error);
		const auto      to_full   = std::filesystem::weakly_canonical(real_to, to_error);
		if (!from_error && !to_error) {
			const auto inside = to_full.lexically_relative(from_full);
			if (!inside.empty() && inside != "." && *inside.begin() != "..") {
				return KERNEL_ERROR_EINVAL;
			}
		}
	}
	// The destination's parent must already be a directory (POSIX; nothing is created).
	if (const int error = CreationParentError(real_to); error != OK) {
		return error;
	}
	const auto to_info = Common::File::GetInfo(real_to);
	if (to_info) {
		std::error_code same_error;
		if (std::filesystem::equivalent(real_from, real_to, same_error)) {
			return OK; // POSIX: renaming a file onto itself does nothing.
		}
		if (!from_info->is_file && to_info->is_file) {
			return KERNEL_ERROR_ENOTDIR;
		}
		if (from_info->is_file && !to_info->is_file) {
			return KERNEL_ERROR_EISDIR;
		}
		if (!to_info->is_file) {
			// Directory over directory: a non-empty target is never replaced. An empty one is
			// left to the native rename, which replaces it atomically on POSIX hosts and fails
			// (EACCES) on Windows, which has no atomic directory replace. Never delete first.
			std::error_code error;
			if (!std::filesystem::is_empty(real_to, error) || error) {
				return error ? KERNEL_ERROR_EIO : KERNEL_ERROR_ENOTEMPTY;
			}
		}
	}
	// Files are replaced atomically; an existing destination survives a failed rename.
	switch (Common::File::RenameReplacing(real_from, real_to)) {
		case 0: break;
		case ENOENT: return KERNEL_ERROR_ENOENT;
		case EACCES: return KERNEL_ERROR_EACCES;
		case ENOTEMPTY: return KERNEL_ERROR_ENOTEMPTY;
		case ENOSPC: return KERNEL_ERROR_ENOSPC;
		case EROFS: return KERNEL_ERROR_EROFS;
		// Different host volumes: report it; never copy and delete.
		case EXDEV: return KERNEL_ERROR_EXDEV;
		default: return KERNEL_ERROR_EIO;
	}

	LOGF("\tKernelRename: %s -> %s\n", from, to);

	return OK;
}

int KYTY_SYSV_ABI KernelGetdirentries(int fd, char* buf, int nbytes, int64_t* basep) {
	PRINT_NAME();

	if (fd < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	auto* file = g_files->GetFile(fd);

	if (file == nullptr) {
		return KERNEL_ERROR_EBADF;
	}

	if (!file->directory || nbytes < static_cast<int>(DIR_BLOCK_SIZE)) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_IF(!file->opened);
	Common::LockGuard lock(file->mutex);
	if (file->dents_offset > file->dirents.size() || file->dents_offset % DIR_BLOCK_SIZE != 0) {
		return KERNEL_ERROR_EINVAL;
	}

	LOGF("\t dir    = %s\n"
	     "\t nbytes = %d\n"
	     "\t offset = %" PRIu64 "\n",
	     Common::PathToString(file->real_name).c_str(), nbytes, file->dents_offset);

	if (basep != nullptr) {
		*basep = static_cast<int64_t>(file->dents_offset);
	}
	return static_cast<int>(
	    ReadDirectory(file, buf, AlignDown(static_cast<uint64_t>(nbytes), DIR_BLOCK_SIZE)));
}

int KYTY_SYSV_ABI KernelGetdents(int fd, char* buf, int nbytes) {
	PRINT_NAME();

	return KernelGetdirentries(fd, buf, nbytes, nullptr);
}

int KYTY_SYSV_ABI KernelMkdir(const char* path, uint16_t mode) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	if (std::strlen(path) > 255) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	LOGF("\t path = %s\n"
	     "\t mode = %04" PRIx16 "\n",
	     path, mode);

	auto real_name = g_mount_points->ResolvePath(std::string(path));
	if (real_name.empty()) {
		return KERNEL_ERROR_ENOENT;
	}
	if (Common::IsArchivePath(real_name)) {
		return KERNEL_ERROR_EROFS;
	}

	if (Common::File::GetInfo(real_name)) {
		return KERNEL_ERROR_EEXIST;
	}
	if (const int error = CreationParentError(real_name); error != OK) {
		return error;
	}

	return Common::File::CreateDirectory(real_name) ? OK : KERNEL_ERROR_EIO;
}

int KYTY_SYSV_ABI KernelRmdir(const char* path) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(path) > 255) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	LOGF("\t path = %s\n", path);

	const auto directory = Common::FixDirectorySlash(NormalizeGuestPath(path));
	if (directory.ends_with("/./") || directory.ends_with("/../")) {
		return KERNEL_ERROR_EINVAL;
	}

	if (g_mount_points->IsMountPoint(directory)) {
		return KERNEL_ERROR_EBUSY;
	}

	const auto real_name = g_mount_points->ResolvePath(path);
	if (Common::IsArchivePath(real_name)) {
		return KERNEL_ERROR_EROFS;
	}

	const auto info = Common::File::GetInfo(real_name);
	if (!info) {
		return MissingNodeError(real_name);
	}

	if (info->is_file) {
		return KERNEL_ERROR_ENOTDIR;
	}

	std::error_code error;
	if (!std::filesystem::remove(real_name, error)) {
		return error == std::errc::directory_not_empty ? KERNEL_ERROR_ENOTEMPTY : KERNEL_ERROR_EIO;
	}

	return OK;
}

int KYTY_SYSV_ABI KernelCheckReachability(const char* path) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	if (std::strlen(path) > 255) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	LOGF("\t KernelCheckReachability: %s\n", path);

	std::string mounted_path = std::string(path);

	if (IsRandomDevice(mounted_path)) {
		return OK;
	}

	auto real_name = g_mount_points->ResolvePath(mounted_path);

	return Common::File::GetInfo(real_name) ? OK : KERNEL_ERROR_ENOENT;
}

} // namespace Libs::LibKernel::FileSystem
