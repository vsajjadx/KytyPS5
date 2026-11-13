#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/guest_gpu/pm4.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "loader/symbolDatabase.h"
#include "loader/systemContent.h"
#include "loader/elf.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace Libs {
void InitLibKernel_1(Loader::SymbolDatabase*);
}

namespace {

namespace Pm4 = Libs::Graphics::Pm4;

TEST(Trinity, ModeQueriesPacketsAndMemoryAgree) {
	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions options;
	options.printf_direction = Config::LogDirection::Silent;
	Config::Load(options);
	subsystems.Initialize<Log::Lifecycle>();
	Loader::SymbolDatabase symbols;
	Libs::InitLibKernel_1(&symbols);
	const auto* query = symbols.FindByNid("tU5e3f9gSiU", Loader::SymbolType::Func);
	const auto* operation = symbols.FindByNid("NH6xARDOVv8", Loader::SymbolType::Func);
	ASSERT_NE(query, nullptr);
	ASSERT_NE(operation, nullptr);
	const auto is_trinity = reinterpret_cast<int(KYTY_SYSV_ABI*)()>(query->vaddr);
	const auto operation_mode = reinterpret_cast<int(KYTY_SYSV_ABI*)(int*, int*)>(operation->vaddr);

	struct CommandBuffer {
		uint32_t* bottom;
		uint32_t* top;
		uint32_t* cursor_up;
		uint32_t* cursor_down;
		void* callback;
		void* user_data;
		uint32_t reserved_dw;
	};
	const auto path = std::filesystem::temp_directory_path() /
	                  ("kyty_trinity_" + std::to_string(reinterpret_cast<uintptr_t>(&symbols)) + ".json");
	struct Case { bool enabled; uint32_t attribute; bool expected; };
	for (const auto test: {Case {false, 0, false}, Case {false, 1u << 22, false}, Case {true, 0, false},
	                       Case {true, 1u << 21, false}, Case {true, 1u << 22, true}}) {
		options.pro_enabled = test.enabled;
		Config::Load(options);
		{
			std::ofstream file(path);
			file << "{\"attribute3\":" << test.attribute
			     << R"(,"kernel":{"flexibleMemorySize":3221225472,"cpuPageTableSize":134217728,"gpuPageTableSize":0},"amm":{"pagetableMemorySizeInMib":96}})";
			ASSERT_TRUE(file.good());
		}
		testing::internal::CaptureStdout();
		Loader::SystemContentLoadParamSfo(path);
		std::filesystem::remove(path);
		EXPECT_EQ(Loader::SystemContentIsTrinityMode(), test.expected);
		EXPECT_EQ(is_trinity(), test.expected ? 1 : 0);
		int mode = -1, submode = -1;
		EXPECT_EQ(operation_mode(&mode, &submode), 0);
		EXPECT_EQ(mode, test.expected ? 3 : 2);
		EXPECT_EQ(submode, 0);
		uint8_t graphics_mode = 0xff;
		Libs::Graphics::Gen5::AgcGetIsTrinityMode(&graphics_mode);
		EXPECT_EQ(graphics_mode, test.expected ? 1 : 0);
		std::array<uint32_t, 5> packet {};
		CommandBuffer buffer {packet.data(), packet.data() + packet.size(), packet.data(),
		                      packet.data() + packet.size(), nullptr, nullptr, 0};
		Libs::Graphics::Gen5::AgcDcbSetUcRegistersIndirect(
		    reinterpret_cast<Libs::Graphics::Gen5::CommandBuffer*>(&buffer), nullptr, 0);
		EXPECT_EQ(packet[0], KYTY_PM4(5, Pm4::IT_SET_UCONFIG_REG_INDIRECT,
		                              test.expected ? 1 : 0));
		const auto console = testing::internal::GetCapturedStdout();
		if (test.enabled) {
			EXPECT_EQ(console, test.expected ? "[PS5] title Trinity support=yes\n"
			                                 : "[PS5] title Trinity support=no\n");
		} else {
			EXPECT_TRUE(console.empty()) << console;
		}
	}

	using namespace Libs::LibKernel::Memory;
	subsystems.Initialize<Lifecycle>();
	constexpr uint64_t mib = 1024 * 1024;
	constexpr uint64_t backing = (12800 + 1216 - 64 - 96) * mib;
	constexpr uint64_t flexible = 3072 * mib;
	EXPECT_EQ(TestGuestBackingSize(), backing);
	EXPECT_EQ(KernelGetDirectMemorySize(), backing - flexible);
	size_t configured = 0;
	ASSERT_EQ(KernelConfiguredFlexibleMemorySize(&configured), 0);
	EXPECT_EQ(configured, flexible);
	int64_t available_start = -1;
	size_t available_size = 0;
	ASSERT_EQ(KernelAvailableDirectMemorySize(0, backing - flexible, 0x4000,
	                                         &available_start, &available_size), 0);
	EXPECT_EQ(available_start, 0);
	EXPECT_EQ(available_size, backing - flexible);
	int64_t physical = -1;
	ASSERT_EQ(KernelAllocateDirectMemory(backing - flexible - 0x4000, backing - flexible,
	                                     0x4000, 0x4000, 0, &physical), 0);
	void* mapped = nullptr;
	ASSERT_EQ(KernelMapDirectMemory(&mapped, 0x4000, 2, 0, physical, 0), 0);
	*static_cast<volatile uint64_t*>(mapped) = 0x123456789abcdef0ull;
	EXPECT_EQ(*static_cast<volatile uint64_t*>(mapped), 0x123456789abcdef0ull);
	EXPECT_EQ(KernelMunmap(reinterpret_cast<uint64_t>(mapped), 0x4000), 0);
	EXPECT_EQ(KernelCheckedReleaseDirectMemory(physical, 0x4000), 0);
}

} // namespace

namespace {

struct ElfFixtureFile {
	std::filesystem::path path = std::filesystem::temp_directory_path() /
	    ("kyty-elf-loader-" + std::to_string(
	        std::chrono::steady_clock::now().time_since_epoch().count()) + ".elf");

	~ElfFixtureFile() {
		std::error_code ignored;
		std::filesystem::remove(path, ignored);
	}
	void Write(const Loader::Elf64_Ehdr& header, const Loader::Elf64_Phdr& segment,
	           bool invalid_self_index = false) const {
		constexpr size_t PayloadOffset = 0x100;
		constexpr size_t PayloadSize = 16;
		std::vector<uint8_t> bytes(PayloadOffset + PayloadSize, 0x5a);
		size_t elf_offset = 0;
		if (invalid_self_index) {
			Loader::SelfHeader self {};
			const uint8_t ident[] = {0x4f, 0x15, 0x3d, 0x1d, 0, 1, 1, 0x12, 1, 1, 0, 0};
			std::memcpy(self.ident, ident, sizeof(ident));
			self.file_size = bytes.size();
			self.segments_num = 1;
			self.unknown = 0x22;
			const Loader::SelfSegment encoded {0x800ull | (4095ull << 20u), PayloadOffset,
			                                   PayloadSize, PayloadSize};
			std::memcpy(bytes.data(), &self, sizeof(self));
			std::memcpy(bytes.data() + sizeof(self), &encoded, sizeof(encoded));
			elf_offset = sizeof(self) + sizeof(encoded);
		}
		std::memcpy(bytes.data() + elf_offset, &header, sizeof(header));
		std::memcpy(bytes.data() + elf_offset + sizeof(header), &segment, sizeof(segment));
		std::ofstream output(path, std::ios::binary | std::ios::trunc);
		output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
		ASSERT_TRUE(output.good());
	}
};

Loader::Elf64_Ehdr TestElfHeader() {
	Loader::Elf64_Ehdr header {};
	header.e_ident[Loader::EI_MAG0] = 0x7f;
	header.e_ident[Loader::EI_MAG1] = 'E';
	header.e_ident[Loader::EI_MAG2] = 'L';
	header.e_ident[Loader::EI_MAG3] = 'F';
	header.e_ident[Loader::EI_CLASS] = Loader::ELFCLASS64;
	header.e_ident[Loader::EI_DATA] = Loader::ELFDATA2LSB;
	header.e_ident[Loader::EI_VERSION] = Loader::EV_CURRENT;
	header.e_ident[Loader::EI_OSABI] = Loader::ELFOSABI_FREEBSD;
	header.e_type = Loader::ET_DYNAMIC;
	header.e_machine = Loader::EM_X86_64;
	header.e_version = Loader::EV_CURRENT;
	header.e_ehsize = sizeof(header);
	header.e_phoff = sizeof(header);
	header.e_phentsize = sizeof(Loader::Elf64_Phdr);
	header.e_phnum = 1;
	return header;
}

Loader::Elf64_Phdr TestElfSegment() {
	return {.p_type = Loader::PT_LOAD, .p_flags = Loader::PF_R, .p_offset = 0x100,
	        .p_vaddr = 0, .p_paddr = 0, .p_filesz = 16, .p_memsz = 0x4000, .p_align = 0x4000};
}

TEST(ElfLoader, RejectsSegmentArithmeticAndCopyOverflows) {
	const auto header = TestElfHeader();
	const auto normal = TestElfSegment();
	std::vector<Loader::Elf64_Phdr> invalid;
	auto segment = normal;
	segment.p_memsz = 8;
	invalid.push_back(segment); // File bytes exceed the segment's declared memory.
	segment = normal;
	segment.p_align = 3;
	invalid.push_back(segment);
	segment = normal;
	segment.p_memsz = UINT64_MAX;
	invalid.push_back(segment); // Aligned-size rounding wraps to zero.
	segment = normal;
	segment.p_vaddr = UINT64_MAX - 0x3fff;
	segment.p_memsz = 0x8000;
	invalid.push_back(segment); // Base-size calculation wraps and loading writes before the base.
	segment = normal;
	segment.p_offset = UINT64_MAX - 7;
	invalid.push_back(segment);
	segment = normal;
	segment.p_offset = 0x108;
	invalid.push_back(segment); // The read would extend past the physical ELF file.
	for (size_t index = 0; index < invalid.size(); ++index) {
		SCOPED_TRACE(index);
		ElfFixtureFile fixture;
		fixture.Write(header, invalid[index]);
		Loader::Elf64 elf;
		elf.Open(fixture.path);
		EXPECT_FALSE(elf.IsValid());
	}
}

TEST(ElfLoader, RejectsTruncatedOrWrappedProgramHeaderTables) {
	for (const uint64_t offset: {UINT64_MAX, uint64_t {0x100}}) {
		SCOPED_TRACE(offset);
		auto header = TestElfHeader();
		header.e_phoff = offset;
		ElfFixtureFile fixture;
		fixture.Write(header, TestElfSegment());
		Loader::Elf64 elf;
		elf.Open(fixture.path);
		EXPECT_FALSE(elf.IsValid());
	}
}

TEST(ElfLoader, RejectsSelfProgramHeaderIndexBeforeSegmentLoading) {
	ElfFixtureFile fixture;
	fixture.Write(TestElfHeader(), TestElfSegment(), true);
	Loader::Elf64 elf;
	elf.Open(fixture.path);
	EXPECT_FALSE(elf.IsValid());
}

TEST(ElfLoader, PreservesValidAlignmentAndExactEndReads) {
	for (const uint64_t alignment: {uint64_t {0}, uint64_t {1}, uint64_t {0x4000}}) {
		SCOPED_TRACE(alignment);
		auto segment = TestElfSegment();
		segment.p_align = alignment;
		ElfFixtureFile fixture;
		fixture.Write(TestElfHeader(), segment);
		Loader::Elf64 elf;
		elf.Open(fixture.path);
		ASSERT_TRUE(elf.IsValid());
		std::array<uint8_t, 18> destination;
		destination.fill(0xa5);
		elf.LoadSegment(reinterpret_cast<uint64_t>(destination.data() + 1), 0x100, 16);
		EXPECT_EQ(destination.front(), 0xa5);
		EXPECT_EQ(destination.back(), 0xa5);
		for (size_t index = 1; index < 17; ++index) EXPECT_EQ(destination[index], 0x5a);
	}
	// A pure BSS segment has no file extent to read, even when its offset is past EOF.
	auto bss = TestElfSegment();
	bss.p_filesz = 0;
	bss.p_offset = UINT64_MAX;
	ElfFixtureFile fixture;
	fixture.Write(TestElfHeader(), bss);
	Loader::Elf64 elf;
	elf.Open(fixture.path);
	EXPECT_TRUE(elf.IsValid());
}

} // namespace

int RunTrinityTests() {
	int argc = 1;
	char name[] = "trinity_tests";
	char* argv[] = {name, nullptr};
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
