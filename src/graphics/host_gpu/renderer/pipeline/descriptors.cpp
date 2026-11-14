#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/readbackAttribution.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/ActivityDiagnostics.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <cinttypes>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <bit>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <set>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

// A plain sampled view may resolve to an existing color-target owner retaining CMASK.
// The REQUESTED descriptor has no metadata, so FindImage's MaterializeColorClear
// returns before consuming CMASK bytes. Certify only this finite, exact owner shape;
// normal discovery, refresh and final native rebinding still run on every dispatch.
[[nodiscard]] bool IsPlainCmaskActivityOwner(
    const ShaderRecompiler::IR::ImageResource& resource, const TextureBinding& binding,
    const Image& image) {
	const auto& requested = binding.desc.info;
	const auto& native = image.info;
	const auto& view = binding.desc.view_info;
	const auto& meta = native.metadata.range;
	return resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	       resource.read && !resource.written && !resource.atomic && !resource.heap &&
	       !resource.depth_compare && binding.desc.type == TextureCache::BindingType::Texture &&
	       !binding.desc.heap && image.registered && image.usage.render_target &&
	       !image.IsCpuDirty() && !image.IsBufferModified() &&
	       native.metadata.kind == ImageMetadataKind::Cmask &&
	       native.metadata.compression == VideoOutCompression::Uncompressed &&
	       !native.metadata.stencil_compressed && requested.metadata.kind == ImageMetadataKind::None &&
	       requested.metadata.range.Empty() &&
	       requested.metadata.compression == VideoOutCompression::Uncompressed &&
	       !requested.metadata.stencil_compressed &&
	       native.data.Valid() && requested.data == native.data && meta.Valid() &&
	       meta.address % 0x1000u == 0u && meta.size % 0x1000u == 0u &&
	       (meta.End() <= native.data.address || native.data.End() <= meta.address) &&
	       native.type == Prospero::ImageType::kColor2D && requested.type == native.type &&
	       native.extent == requested.extent && native.samples == 1u && requested.samples == 1u &&
	       image.backing.samples == 1u && native.resources.levels == 1u && native.resources.layers == 1u &&
	       requested.resources.levels == 1u && requested.resources.layers == 1u &&
	       native.pixel_format == requested.pixel_format && native.pixel_format == view.format &&
	       native.pixel_format == image.backing.format && view.type == vk::ImageViewType::e2D &&
	       view.aspect == vk::ImageAspectFlagBits::eColor && view.base_level == 0u &&
	       view.level_count == 1u && view.base_layer == 0u && view.layer_count == 1u;
}

// Scalar resource reads are folded into the pipeline specialization when it is compiled, so a
// write to the same guest range by the same dispatch could in principle leave stale constants
// behind. The ranges compared are conservative (whole descriptors, not the bytes actually
// touched), and the overlap is often harmless, e.g. a large UAV that merely spans a descriptor
// table. Aborting the whole process made such titles unplayable, so report each distinct pair
// once and keep going; a genuine stale-constant problem then shows up as a visual glitch.
void WarnScalarReadOverlap(const char* kind, uint64_t read_address, uint64_t read_size,
                           uint64_t write_address, uint64_t write_size) {
	static std::mutex                              mutex;
	static std::set<std::pair<uint64_t, uint64_t>> reported;
	std::scoped_lock                               lock {mutex};
	if (reported.size() >= 64 || !reported.emplace(read_address, write_address).second) {
		return;
	}
	std::printf("Warning: scalar resource read [0x%llx, +0x%llx) overlaps %s write "
	            "[0x%llx, +0x%llx); continuing with the specialization as compiled\n",
	            static_cast<unsigned long long>(read_address),
	            static_cast<unsigned long long>(read_size), kind,
	            static_cast<unsigned long long>(write_address),
	            static_cast<unsigned long long>(write_size));
	std::fflush(stdout);
}

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	if (!texture.image_id) {
		EXIT_IF(texture.image_view != nullptr || !texture.mip_views.empty());
		return {nullptr, nullptr, vk::ImageLayout::eGeneral};
	}
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

// 1D textures are 2D images one texel high, so they share 2D surfaces at their address.
static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1D:
		case Prospero::ImageType::kColor1DArray:
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

static bool kDisableNarrowWrites() {
	static const bool disabled = std::getenv("KYTY_NO_NARROW_WRITES") != nullptr;
	return disabled;
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset,
                    BufferCache::PreparedUpload& uploads) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address == 0 || size == 0) {
		// Null descriptors read zero and drop writes, with or without shader bounds checks.
		return {nullptr, 0, vk::WholeSize};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	// Swizzled or ADD_TID buffers move fixed-offset writes by lane; only plain ones keep them.
	const bool linear = (resource.packed_stride & ((1u << 14u) | (1u << 20u))) == 0u;
	const bool narrow = resource.written && linear && resource.fixed_write_end != 0u &&
	                    resource.fixed_write_end != UINT32_MAX &&
	                    resource.fixed_write_end < size && !kDisableNarrowWrites();
	if (resource.written && std::getenv("KYTY_NARROW_DIAG") != nullptr) {
		static uint64_t n = 0, n_narrow = 0, n_dynamic = 0, n_swz = 0, n_full = 0;
		n++;
		if (narrow) n_narrow++;
		else if (resource.fixed_write_end == UINT32_MAX) n_dynamic++;
		else if (!linear) n_swz++;
		else n_full++;
		if (n % 5000 == 0) {
			std::printf("NARROWDIAG: written bindings=%llu narrowed=%llu dynamic=%llu swizzled=%llu covers_all=%llu%c",
			            (unsigned long long)n, (unsigned long long)n_narrow, (unsigned long long)n_dynamic,
			            (unsigned long long)n_swz, (unsigned long long)n_full, 10);
			std::fflush(stdout);
		}
	}
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(
	    address, size, resource.written && !narrow,
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id,
	    !resource.written, &uploads);
	if (narrow && resource.fixed_write_end != 0u) {
		// The full range above must retain its persistent owner: a streaming upload is
		// read-only guest data and cannot receive writes that later readbacks must see.
		// Upload the full CPU-owned contents first, then mark only the written prefix.
		(void)context.GetBufferCache().ObtainBuffer(address, resource.fixed_write_end, true, false,
		                                            id);
	}
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written) {
		context.GetTextureCache().InvalidateMemoryFromGPU(
		    address, narrow ? std::max<uint64_t>(resource.fixed_write_end, 1u) : size);
	}
	return result;
}

// Reserved and unsupported T# fields must be zero; perf_mod only scales sampler state and the
// mip-stats counter only counts sampling.
static bool IsCanonicalTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 1u << 29u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (uint32_t {descriptor.MipStatsCntEn()} << 25u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if ((descriptor.fields[1] & field1_reserved_mask) != 0 ||
	    (descriptor.fields[2] & field2_reserved_mask) != 0 || descriptor.fields[3] != field3_expected) {
		return false;
	}
	if (r128) {
		return descriptor.fields[4] == 0 && descriptor.fields[5] == 0 && descriptor.fields[6] == 0 &&
		       descriptor.fields[7] == 0;
	}
	return descriptor.fields[4] == field4_expected && descriptor.fields[5] == field5_expected;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	if (!IsCanonicalTextureEncoding(descriptor, r128)) {
		return false;
	}
	if (r128) {
		return true;
	}
	if (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type())) {
		return false;
	}
	// Bits 0-7 name the mip-stats counter. A clear COMPRESSION_EN disables the metadata and its
	// settings.
	const auto         metadata_control = descriptor.fields[6] & 0x00ffff00u;
	constexpr uint32_t pipe_aligned     = 1u << 19u;
	if (!descriptor.MetaCompress()) {
		return true;
	}
	constexpr uint32_t htile_control = pipe_aligned | 1u << 21u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	// Comparison sampling changes the host format, not the guest metadata encoding.
	const bool encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                         IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	const bool is_2d_msaa =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
	    descriptor.Type() == Prospero::ImageType::kColor2DMsaa && descriptor.Depth() == 0 &&
	    descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	// A 4.8 fixed-point MIN_LOD at or below BASE_LEVEL clamps no level of the view.
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d || is_2d_msaa) &&
	       supported_tile && descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() <= descriptor.BaseLevel() * 256u && supported_swizzle &&
	       descriptor.BCSwizzle() == 0 && !descriptor.MsaaDepth();
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsCanonicalTextureEncoding(descriptor, resource.r128);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = uint_resource &&
	    ShaderRecompiler::IR::RawSignedStorageFormat(resource, format) != Prospero::BufferFormat::kInvalid;
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

// sRGB views decode before filtering; a sampler that skips degamma reads through a UNorm view.
static vk::Format SampledViewFormat(const GraphicContext& graphics, Prospero::BufferFormat format,
                                    vk::Format pixel_format, bool skip_degamma) {
	if (skip_degamma) return SrgbUnormViewFormat(pixel_format);
	switch (format) {
		case Prospero::BufferFormat::k8Srgb:
			return graphics.r8_srgb_views ? vk::Format::eR8Srgb : pixel_format;
		case Prospero::BufferFormat::k8_8Srgb:
			return graphics.r8g8_srgb_views ? vk::Format::eR8G8Srgb : pixel_format;
		default: return pixel_format;
	}
}

// Atomics and raw signed stores use unsigned integer views; sRGB images store through UNorm.
static vk::Format StorageViewFormat(const ShaderRecompiler::IR::ImageResource& resource,
                                    Prospero::BufferFormat format, vk::Format pixel_format) {
	if (resource.atomic64) return vk::Format::eR64Uint;
	if (resource.atomic) return vk::Format::eR32Uint;
	const auto raw_sint_format = ShaderRecompiler::IR::RawSignedStorageFormat(resource, format);
	if (raw_sint_format != Prospero::BufferFormat::kInvalid) return VulkanFormat(raw_sint_format);
	return SrgbUnormViewFormat(pixel_format);
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || resource.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (!info.IsTiled()) {
		return false;
	}
	ImageInfo physical = info;
	// A view entirely past MAX_MIP addresses its own chain starting at its base mip (PPSA02110
	// compresses mip 1 through a T# that declares one level).
	if (base_level >= physical_levels) {
		const auto mip  = info.MipExtent(base_level);
		physical.extent = {mip.width, mip.height, info.IsVolume() ? mip.depth : 1u};
		physical_levels = view_levels;
	}
	physical.resources.levels = physical_levels;
	physical.UpdateSize();
	info.UpdateSize();
	const auto same_mip = [&](uint32_t target, uint32_t source) {
		const auto& left  = physical.mip_layout[target];
		const auto& right = info.mip_layout[source];
		return left.offset == right.offset && left.size == right.size &&
		       left.pitch == right.pitch && left.height == right.height &&
		       left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
		       left.surface_z == right.surface_z &&
		       physical.MipExtent(target) == info.MipExtent(source);
	};
	if (physical.first_tail_level == info.first_tail_level &&
	    physical.tiled_slice_stride == info.tiled_slice_stride &&
	    physical.data.size == info.data.size) {
		bool matches = true;
		for (uint32_t level = 0; level < physical_levels; ++level) {
			matches &= same_mip(level, level);
		}
		if (matches) {
			return true;
		}
	}
	if (metadata ||
	    (info.TransferLayers() > 1 && physical.tiled_slice_stride != info.tiled_slice_stride)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= physical_levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (!same_mip(target, source) ||
			    (target >= physical.first_tail_level) != (source >= info.first_tail_level)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			info       = std::move(physical);
			base_level = base;
			return true;
		}
	}
	return false;
}

// Members of an indirect (GPU-indexed) image table share one shader slot, so the shader is
// specialised for a single numeric class / dimensionality. A candidate whose descriptor does not
// fit that slot cannot legally be bound there (e.g. sampling a UINT image through a float
// sampler is invalid Vulkan). Such candidates are replaced by a neutral null image instead of
// aborting the whole emulator.
static bool IndirectCandidateFitsSlot(const ShaderRecompiler::IR::ImageResource& resource,
                                      const ShaderTextureResource&               descriptor) {
	using Dim = ShaderRecompiler::Decoder::ImageDimension;
	if (resource.numeric_class != Prospero::SampledTextureNumericClass(descriptor.Format())) {
		return false;
	}
	const auto type    = descriptor.Type();
	const bool is_3d   = type == Prospero::ImageType::kColor3D;
	const bool is_1d   = type == Prospero::ImageType::kColor1D || type == Prospero::ImageType::kColor1DArray;
	const bool is_cube = type == Prospero::ImageType::kCube;
	const bool slot_3d = resource.dimension == Dim::Dim3D;
	const bool slot_1d = resource.dimension == Dim::Dim1D || resource.dimension == Dim::Dim1DArray;
	return is_3d == slot_3d && is_1d == slot_1d && is_cube == resource.cube;
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	// Image heap resolves do not follow the draws that sample them and storage access is not
	// sampling: neither reports samples.
	if (m_lod_stats && descriptor.MipStatsCntEn() && !resource.heap && !storage) {
		auto&         counter = m_mip_stats[descriptor.MipStatsCntId()];
		const uint8_t base    = descriptor.BaseLevel();
		counter.finest_level  = std::min(counter.finest_level, base);
		if (descriptor.MinLod() > base << 8u) {
			counter.clamped = true;
		} else {
			counter.unclamped = true;
		}
	}

	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    resource.indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage &&
	    !IndirectCandidateFitsSlot(resource, descriptor)) {
		static std::atomic<uint32_t> warned {0};
		if (warned.fetch_add(1, std::memory_order_relaxed) < 16u) {
			std::printf("Warning: indirect image table candidate does not fit its shader slot "
			            "(slot numeric=%u dimension=%u cube=%d, descriptor format=%u type=%u "
			            "addr=0x%016" PRIx64 "); binding a null image instead\n",
			            static_cast<uint32_t>(resource.numeric_class),
			            static_cast<uint32_t>(resource.dimension), resource.cube,
			            static_cast<uint32_t>(descriptor.Format()),
			            static_cast<uint32_t>(descriptor.Type()), descriptor.Base40());
			std::fflush(stdout);
		}
		// Same native null binding as an unbound descriptor above.
		return {.layout = vk::ImageLayout::eGeneral};
	}

	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	auto view_levels = multisampled || single_storage_mip
	                       ? 1u
	                       : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = ShaderRecompiler::IR::ReadsDepthBits(resource)
	                                ? Prospero::BufferFormat::k16UNorm
	                                : descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class ==
	                   Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	auto          image_layers = layered ? depth : 1u;
	// A cube holds six faces even when DEPTH, its last face slice, ends before them (Teardown's
	// sky cube has DEPTH 0).
	if (descriptor.Type() == Prospero::ImageType::kCube) {
		image_layers = std::max(image_layers, 6u);
	}
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto view_format =
	    storage ? StorageViewFormat(resource, format, pixel_format) : pixel_format;
	const auto              block_bytes = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, 0};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		desc.info.metadata.kind = ImageMetadataKind::Dcc;
	}
	if (levels > physical_levels) {
		if (!ResolveTextureMipView(desc.info, physical_levels,
		                           !resource.r128 && descriptor.MetaCompress(), view_levels,
		                           view_base)) {
			// The layout stores no mip past MAX_MIP here, so a view starting in the chain ends at
			// MAX_MIP; dynamic-mip shaders still count mips up to LAST_LEVEL.
			if (base_level >= physical_levels ||
			    resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
				EXIT("unsupported texture mip view changes physical layout: base=%u last=%u "
				     "max=%u extent=%ux%ux%u tile=%u\n",
				     base_level, last_level, max_mip, width, height, depth,
				     static_cast<uint32_t>(tile));
			}
			view_levels                = physical_levels - base_level;
			desc.info.resources.levels = physical_levels;
			desc.info.data.size        = 0;
			desc.info.UpdateSize();
		}
	} else {
		desc.info.UpdateSize();
	}
	const auto alignment = desc.info.DataAlignment();
	EXIT_NOT_IMPLEMENTED(desc.info.data.size == 0 || alignment == 0 ||
	                     (address & (static_cast<uint64_t>(alignment) - 1u)) != 0);
	// Image heaps keep the descriptors of unmapped textures, which PS5 never samples.
	if (resource.heap && !m_context.IsMapped(address, desc.info.data.size)) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	if (storage) {
		ValidateStorageTexture(resource, descriptor, desc.info.data.size);
	}
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	desc.heap = resource.heap;

	auto id = texture_cache.FindImage(desc, shader_conversion);
	if (!id) {
		// Another extent at the address holds it; the heap retries once an image is created.
		return {.desc = std::move(desc), .layout = vk::ImageLayout::eGeneral};
	}
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format,
		                            desc.info.data.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		// Set after FindImage, whose metadata clears need the attachment-capable format.
		desc.view_info.format = SampledViewFormat(m_context.GetGraphics(), format, pixel_format,
		                                          resource.skip_degamma);
		(void)SelectSampledColorView(image->info.pixel_format, desc.view_info.format,
		                             descriptor.DstSelXYZW());
	}
	return {id, nullptr, std::move(desc)};
}

static vk::Sampler NativeSampler(RenderContext&                       context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	if (sampler.gather_lod &&
	    static_cast<Prospero::SamplerMipFilter>(descriptor.MipFilter()) !=
	        Prospero::SamplerMipFilter::kNone) {
		// GatherMip selects the guest mip explicitly using a per-level image view. Vulkan
		// gather operations require point mip selection and zero LOD bias, so normalize
		// only the host sampler copy; the shader snapshot retains the guest state.
		descriptor.fields[2] &= ~((0x3u << 26u) | 0xfffffu);
		descriptor.fields[2] |=
		    static_cast<uint32_t>(Prospero::SamplerMipFilter::kPoint) << 26u;
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext& context,
                                             BufferCache::PreparedUpload& uploads,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	const auto [buffer, offset] = context.GetBufferCache().UploadPrepared(
	    uploads, data.data(), data.size_bytes(), 256);
	return {buffer->Handle(), offset, data.size_bytes()};
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	if (!id) {
		return;
	}
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		m_context.GetTextureCache().m_slot_images[id].binding = {};
	}
	m_bound_images.clear();
	// Commit retained the GPU tick. Discard old preparation references now so an
	// inactive shader stage cannot keep upload chunks pinned indefinitely.
	for (auto& stage: m_graphics_bindings.vertex) stage.uploads = {};
	m_graphics_bindings.pixel.uploads = {};
	m_compute_bindings.uploads = {};
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = runtime.module->info;
	const auto& snapshot = *runtime.resources;
	prepared.uploads = {};
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (program.info.images[i].heap) {
			prepared.images[i] = {};
			continue;
		}
		auto binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_FUNCTION();
	auto& cache = m_context.GetBufferCache();
	for (auto* stage: stages) {
		auto&       prepared = *stage;
		const auto& program  = prepared.runtime->module->info;
		const auto& snapshot = *prepared.runtime->resources;
		const auto& layout   = program.bindings;
		prepared.buffer_sources.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
		if (layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)] == 0) {
			continue;
		}
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			if (metadata.descriptor_index == UINT32_MAX) continue;
			auto& source = prepared.buffer_sources[metadata.descriptor_index];
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			const auto address = descriptor.Base48();
			auto size = descriptor.GetSize();
			if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 &&
			    !metadata.scalar) {
				// OOB_SELECT=2 only tests NumRecords != 0; a zero-stride fetch
				// still needs its full payload when NumRecords is smaller than it.
				size = std::max(size, uint64_t {metadata.max_byte_extent});
			}
			if (address == 0 || size == 0) {
				source = {};
				continue;
			}
			if (descriptor.NumRecords() == UINT32_MAX) {
				if (program.info.buffers[resource].written) {
					// Sentinel write ranges end before tables captured by any bound stage.
					for (const auto* reader: stages) {
						for (const auto [read_address, read_size]: reader->runtime->resources->specialization_reads) {
							if (read_size == 0) continue;
							if (read_address > address) {
								size = std::min(size, read_address - address);
							} else if (address - read_address < read_size) {
								EXIT("scalar resource reads overlap a shader buffer write\n");
							}
						}
					}
				}
				//  PPSA15246
				const auto& graphics = m_context.GetGraphics();
				const auto  limit =
				    uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
				    (graphics.StorageMinAlignment() - 1);
				size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
			}
			// Finite GPU descriptors can span sparse PRT holes; clamp only oversized/sentinel ranges.
			if (size >= (uint64_t {1} << 30u) || descriptor.NumRecords() == UINT32_MAX) {
				size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
			}
			source = {address, size, cache.FindBuffer(address, size)};
		}
	}
}

void RenderExecutor::NoteRecordedWriters(std::span<PreparedBindings* const> stages) {
	if (!ReadbackAttribution::Enabled()) return; // off: no collection, formatting or allocation
	auto& cache = m_context.GetBufferCache();
	// NoteRecordedWork only counts work; no submit happens between the native command and here,
	// so this is the tick of the command buffer that holds it (not a submit/complete time).
	const uint64_t tick     = m_context.GetCommandScheduler().CurrentTick();
	const uint64_t sequence = cache.NextRecordedWorkSequence();
	for (const auto* stage: stages) {
		if (stage == nullptr || stage->runtime == nullptr) continue;
		const auto& program = stage->runtime->module->info;
		for (const auto& resource: program.info.buffers) {
			const auto index = resource.descriptor_index;
			if (!resource.written || index == UINT32_MAX || index >= stage->buffer_sources.size() ||
			    index >= stage->buffers.size()) continue;
			const auto& source = stage->buffer_sources[index];
			const auto& native = stage->buffers[index];
			if (source.address == 0 || source.size == 0 || !native.buffer) continue; // null: writes dropped
			ReadbackAttribution::RecordedWriter writer;
			writer.shader     = program.shader_hash;
			writer.module     = stage->runtime->module->id; // already-final bound module identity
			writer.stage      = static_cast<uint32_t>(program.stage);
			writer.resource   = resource.source;
			writer.descriptor = index;
			writer.source_id  = source.id.index; // label only; never resolved
			writer.begin      = source.address;
			writer.size       = source.size;
			const auto handle = static_cast<VkBuffer>(native.buffer);
			static_assert(sizeof(handle) <= sizeof(writer.native_handle));
			std::memcpy(&writer.native_handle, &handle, sizeof(handle));
			writer.native_offset = native.offset;
			writer.native_range  = native.range;
			cache.NoteRecordedWriter(writer, sequence, tick);
		}
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	const auto& program  = prepared.runtime->module->info;
	ReadbackAttribution::Scope attribution_scope(program.shader_hash, ReadbackAttribution::Phase::Bindings);
	const auto& snapshot = *prepared.runtime->resources;
	const auto& layout   = program.bindings;

	prepared.buffers.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (const auto& resource: program.info.buffers) {
		ReadbackAttribution::Scope resource_scope(ReadbackAttribution::Resource(resource.source));
		const auto index = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		uint32_t buffer_offset = 0;
		prepared.buffers[index] = NativeStorageBuffer(m_context, prepared.buffer_sources[index],
		                                              resource, buffer_offset, prepared.uploads);
		pack_memory_offset(index, buffer_offset);
		// Null bindings allow 0 bytes, not the shared null buffer's 16, so no store reaches it.
		prepared.shader_data[layout.MemorySizeDword() + index] =
		    static_cast<uint32_t>(prepared.buffer_sources[index].size + buffer_offset);
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0) {
		prepared.flattened_srt = NativeUpload(m_context, prepared.uploads, snapshot.flattened_srt);
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.uploads, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	const auto& program       = prepared.runtime->module->info;
	const auto& snapshot      = *prepared.runtime->resources;
	auto&       images        = prepared.images;
	auto&       texture_cache = m_context.GetTextureCache();
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!images[i].image_id) continue;
		auto& old_image = texture_cache.m_slot_images[images[i].image_id];
		if (!old_image.registered || old_image.binding.needs_rebind) {
			old_image.binding = {};
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool uses_dma        = false;
	bool uses_image_heap = false;
	FindBuffers(stages);
	for (auto* stage: stages) {
		uses_dma |= stage->runtime->module->info.info.uses_dma;
		uses_image_heap |= stage->runtime->module->info.info.uses_image_heap;
	}
	if (uses_dma) {
		m_context.PrepareBda();
	}
	if (uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		auto& old_image = cache.m_slot_images[target.image_id];
		if (!old_image.registered || old_image.binding.needs_rebind) {
			old_image.binding = {};
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	size_t                         image_descriptor_count = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	for (const auto* prepared: prepared_bindings) {
		const auto& program = prepared->runtime->module->info;
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto count = program.bindings.descriptor_counts[index];
			if (index >= ShaderRecompiler::IR::FirstImageBinding &&
			    index <= static_cast<size_t>(BindingKind::Samplers)) {
				image_descriptor_count += count;
			}
		}
		EXIT_IF(!(pipeline.push_stages & NativeShaderStage(program.stage)));
	}
	if (pipeline.uses_image_heap) {
		m_context.GetImageHeap().Commit(buffer, pipeline_bind_point, pipeline.pipeline_layout);
	}
	// Uploading image-heap slots can wrap the stream and restart command recording.
	// Capture the native buffer only after that allocation-capable work is complete.
	const auto vk_buffer = buffer.Handle();
	m_descriptor_images.resize(image_descriptor_count);
	m_descriptor_writes.clear();
	size_t image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;

	for (auto* prepared: prepared_bindings) {
		const auto& program     = prepared->runtime->module->info;
		auto&       descriptors = *prepared;
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier       = MakeGdsDependency(descriptors.gds.buffer);
			const auto shader_stages = ShaderPipelineStages(NativeShaderStage(program.stage));
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& binding = descriptors.images[i];
			if (!binding.image_id) continue;
			auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.ReadLayout(), vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}

		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind = static_cast<BindingKind>(index);
			vk::WriteDescriptorSet write {};
			write.dstBinding = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			write.descriptorType = NativeDescriptorType(kind);
			write.descriptorCount = program.bindings.descriptor_counts[index];
			const auto image_start = image_cursor;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = image_start;
				image_cursor += write.descriptorCount;
			} else {
				switch (kind) {
					case BindingKind::Buffers:
						write.pBufferInfo = descriptors.buffers.data();
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto& cache = m_context.GetBufferCache();
						const auto* bda_buffer = kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						auto& view = address_buffers[kind == BindingKind::FaultBuffer];
						view = {bda_buffer->Handle(), 0, bda_buffer->Size()};
						write.pBufferInfo = &view;
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						write.pBufferInfo = view;
						break;
					}
					case BindingKind::Samplers:
						for (const auto sampler: descriptors.samplers) {
							m_descriptor_images[image_cursor++] = {sampler, nullptr, vk::ImageLayout::eUndefined};
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (image_cursor != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture = descriptors.images[i];
			if (resource.heap) continue;
			const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
		}

		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
		                        sizeof(push_data), push_data.dwords.data());
	}

	if (!m_descriptor_writes.empty()) {
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			m_context.GetGraphics().device.updateDescriptorSets(
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data(), 0,
			    nullptr);
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set,
			                             0, nullptr);
		}
	}
	if (ShaderRecompiler::IR::HostActivityProofEnabled()) {
		for (const auto* prepared: prepared_bindings) {
			const auto& runtime = *prepared->runtime;
			const auto& info = runtime.module->info;
			const auto& snapshot = *runtime.resources;
			using Reason = ShaderRecompiler::IR::ActivityDiagnosticReason;
			const auto note = [&](Reason reason) { ShaderRecompiler::IR::ActivityDiagnosticNote(info.shader_hash, reason); };
			if (info.stage != ShaderType::Compute || !snapshot.activity_cache ||
			    info.info.uses_dma || info.info.uses_image_heap) { note(Reason::NativeNoCache); continue; }
			std::vector<std::pair<uint64_t, uint64_t>> targets;
			bool valid = snapshot.buffers.size() == info.info.buffers.size() &&
			             snapshot.images.size() == info.info.images.size();
			Reason invalid_reason = valid ? Reason::NativeBufferRejected : Reason::NativeLayoutRejected;
			for (uint32_t index = 0; valid && index < info.info.buffers.size(); ++index) {
				const auto& resource = info.info.buffers[index];
				const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[index]);
				static_assert(static_cast<uint32_t>(BufferCache::ActivityBindingFailure::Reason::Count)==ShaderRecompiler::IR::ActivityDiagnosticBindingReasonCount);
				static_assert(static_cast<uint32_t>(BufferCache::ActivityBindingFailure::PreparedReason::Count)==ShaderRecompiler::IR::ActivityDiagnosticPreparedReasonCount);
				const bool buffer_diag = ShaderRecompiler::IR::ActivityDiagnosticsEnabled(info.shader_hash);
				const auto note_buffer = [&](uint32_t reason,const BufferCache::ActivityBindingFailure* failure=nullptr) {
					if(!buffer_diag)return;
					ShaderRecompiler::IR::ActivityDiagnosticBufferMetadata m;
					m.module=runtime.module->id;m.index=index;m.source=resource.source;m.descriptor_index=resource.descriptor_index;m.reason=reason;
					m.descriptor_address=descriptor.Base48();m.descriptor_size=descriptor.GetSize();
					m.resource_flags=(resource.written?1u:0u)|(resource.read?2u:0u)|(resource.scalar?4u:0u)|(resource.formatted?8u:0u);
					m.prepared_sources=prepared->uploads.sources.size();m.prepared_buffers=prepared->uploads.buffers.size();
					if(resource.descriptor_index<prepared->buffer_sources.size()){
						const auto& s=prepared->buffer_sources[resource.descriptor_index];m.address=s.address;m.size=s.size;m.fields|=1u;
					}
					if(resource.descriptor_index<prepared->buffers.size()){
						const auto& b=prepared->buffers[resource.descriptor_index];m.native_offset=b.offset;m.native_range=b.range;m.fields|=2u;
						const auto handle=static_cast<VkBuffer>(b.buffer);static_assert(sizeof(handle)<=sizeof(m.native_handle));
						std::memcpy(&m.native_handle,&handle,sizeof(handle));
					}
					if(failure){m.fields|=4u;m.binding_fields=failure->fields;m.prefix=failure->prefix;m.binding_reason=static_cast<uint32_t>(failure->reason);m.prepared_reason=static_cast<uint32_t>(failure->prepared_reason);
						m.alignment=failure->alignment;m.owner_begin=failure->owner_begin;m.owner_size=failure->owner_size;
						m.expected_offset=failure->expected_offset;m.expected_range=failure->expected_range;
						m.footprint_begin=failure->footprint_begin;m.footprint_size=failure->footprint_size;m.prepared_offset=failure->prepared_offset;
						const auto copy_overlap=[](ShaderRecompiler::IR::ActivityDiagnosticOverlapMetadata& o,const BufferCache::ActivityBindingFailure::ImageOverlap& x){
							o.kind=x.kind;o.image_id=x.image_id;o.serial=x.serial;o.data_begin=x.data_begin;o.data_size=x.data_size;
							o.query_begin=x.query_begin;o.query_size=x.query_size;o.relation=x.relation;o.overlap_class=x.overlap_class;
						};
						copy_overlap(m.prepared_overlap,failure->prepared_overlap);copy_overlap(m.persistent_overlap,failure->persistent_overlap);
					}
					ShaderRecompiler::IR::ActivityDiagnosticBufferFailure(info.shader_hash,m);
				};
				if (descriptor.NumRecords() == UINT32_MAX) { note_buffer(1); valid = false; break; }
				if (descriptor.Base48() == 0u || descriptor.GetSize() == 0u) continue;
				if (resource.descriptor_index == UINT32_MAX ||
				    resource.descriptor_index >= prepared->buffer_sources.size()) { note_buffer(2); valid = false; break; }
				const auto& source = prepared->buffer_sources[resource.descriptor_index];
				if (source.address != descriptor.Base48()) { note_buffer(3); valid = false; break; }
				if (source.size < descriptor.GetSize()) { note_buffer(4); valid = false; break; }
				targets.emplace_back(source.address, source.size);
				if (resource.descriptor_index >= prepared->buffers.size()) { note_buffer(5); valid = false; break; }
				uint64_t owner_begin = 0, owner_size = 0;
				BufferCache::ActivityBindingFailure binding_failure;
				if (!m_context.GetBufferCache().BindingActivityRange(
				        source.address, source.size, prepared->buffers[resource.descriptor_index],
				        resource.written, owner_begin, owner_size, &prepared->uploads,
				        buffer_diag ? &binding_failure : nullptr)) { note_buffer(6,&binding_failure); valid = false; break; }
				targets.emplace_back(owner_begin, owner_size);
			}
			for (uint32_t index = 0; valid && index < info.info.images.size(); ++index) {
				invalid_reason = Reason::NativeImageRejected;
				const auto& resource = info.info.images[index];
				const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[index]);
				using ImageReason = ShaderRecompiler::IR::ActivityDiagnosticImageReason;
				const auto note_image = [&](ImageReason reason, const Image* native = nullptr) {
					if (!ShaderRecompiler::IR::ActivityDiagnosticsEnabled(info.shader_hash)) return;
					ShaderRecompiler::IR::ActivityDiagnosticNativeImageMetadata m;
					m.reason = reason; m.module = runtime.module->id; m.index = index;
					m.source = resource.source; m.descriptor_index = resource.descriptor_index;
					m.first_use_pc = resource.first_use_pc;
					m.resource_class = static_cast<uint32_t>(resource.resource_class);
					m.resource_flags = (resource.read ? 1u : 0u) | (resource.written ? 2u : 0u) |
					    (resource.heap ? 4u : 0u) | (resource.depth_compare ? 8u : 0u) |
					    (resource.atomic ? 16u : 0u) | (resource.r128 ? 32u : 0u);
					m.descriptor_flags = (descriptor.IsNull() ? 1u : 0u) | (descriptor.MipStatsCntEn() ? 2u : 0u);
					if (index < prepared->images.size()) {
						const auto& b = prepared->images[index];
						m.native_id = b.image_id.index; m.binding_type = static_cast<uint32_t>(b.desc.type);
						const auto raw_view = static_cast<VkImageView>(b.image_view);
						static_assert(sizeof(raw_view) <= sizeof(m.view_handle));
						std::memcpy(&m.view_handle, &raw_view, sizeof(raw_view));
						m.requested_begin = b.desc.info.data.address; m.requested_size = b.desc.info.data.size;
						m.requested_meta_begin = b.desc.info.metadata.range.address;
						m.requested_meta_size = b.desc.info.metadata.range.size;
						m.requested_stencil_begin = b.desc.info.stencil.address;
						m.requested_stencil_size = b.desc.info.stencil.size;
						m.requested_format = static_cast<uint32_t>(b.desc.info.pixel_format);
						m.requested_metadata = static_cast<uint32_t>(b.desc.info.metadata.kind);
						m.view_type = static_cast<uint32_t>(b.desc.view_info.type);
						m.view_format = static_cast<uint32_t>(b.desc.view_info.format);
						m.view_aspect = static_cast<uint32_t>(b.desc.view_info.aspect);
						m.base_level = b.desc.view_info.base_level; m.level_count = b.desc.view_info.level_count;
						m.base_layer = b.desc.view_info.base_layer; m.layer_count = b.desc.view_info.layer_count;
						m.image_flags = (b.image_id ? 1u : 0u) | (b.image_view != nullptr ? 2u : 0u) |
						    (b.desc.info.IsDepth() ? 4u : 0u) | (!b.desc.info.stencil.Empty() ? 8u : 0u);
					}
					if (native != nullptr) {
						m.serial = native->serial;
						m.native_begin = native->info.data.address; m.native_size = native->info.data.size;
						m.native_meta_begin = native->info.metadata.range.address;
						m.native_meta_size = native->info.metadata.range.size;
						m.native_stencil_begin = native->info.stencil.address;
						m.native_stencil_size = native->info.stencil.size;
						m.native_format = static_cast<uint32_t>(native->info.pixel_format);
						m.native_metadata = static_cast<uint32_t>(native->info.metadata.kind);
						m.image_flags |= (native->registered ? 16u : 0u) | (native->info.IsDepth() ? 32u : 0u) |
						    (native->depth_id ? 64u : 0u) | (!native->stencil_ids.empty() ? 128u : 0u) |
						    (!native->info.stencil.Empty() ? 256u : 0u);
					}
					ShaderRecompiler::IR::ActivityDiagnosticNativeImageFailure(info.shader_hash, m);
				};
				if (resource.written || resource.heap || resource.depth_compare || descriptor.MipStatsCntEn()) {
					if (ShaderRecompiler::IR::ActivityDiagnosticsEnabled(info.shader_hash))
						note_image(resource.written ? ImageReason::Written : resource.heap ? ImageReason::Heap :
						           resource.depth_compare ? ImageReason::DepthCompare : ImageReason::MipStats);
					valid = false; break;
				}
				if (descriptor.IsNull()) continue;
				const auto& binding = prepared->images[index];
				if (!binding.image_id || binding.image_view == nullptr) {
					if (ShaderRecompiler::IR::ActivityDiagnosticsEnabled(info.shader_hash))
						note_image(!binding.image_id ? ImageReason::MissingImage : ImageReason::MissingView);
					valid = false; break;
				}
				const auto& image = m_context.GetTextureCache().m_slot_images[binding.image_id];
				const bool plain_cmask_owner = IsPlainCmaskActivityOwner(resource, binding, image);
				// Keep DCC/HTile, metadata-bearing requests, depth/stencil and heaps excluded.
				if (!image.registered || image.info.IsDepth() || binding.desc.info.IsDepth() ||
				    image.depth_id || !image.stencil_ids.empty() ||
				    (image.info.metadata.kind != ImageMetadataKind::None && !plain_cmask_owner) ||
				    binding.desc.info.metadata.kind != ImageMetadataKind::None ||
				    !image.info.stencil.Empty() || !binding.desc.info.stencil.Empty()) {
					if (ShaderRecompiler::IR::ActivityDiagnosticsEnabled(info.shader_hash))
						note_image(!image.registered ? ImageReason::Unregistered :
					           image.info.IsDepth() ? ImageReason::NativeDepth :
					           binding.desc.info.IsDepth() ? ImageReason::RequestedDepth :
					           image.depth_id ? ImageReason::DepthAssociation :
					           !image.stencil_ids.empty() ? ImageReason::StencilAssociation :
					           image.info.metadata.kind != ImageMetadataKind::None ? ImageReason::NativeMetadata :
					           binding.desc.info.metadata.kind != ImageMetadataKind::None ? ImageReason::RequestedMetadata :
					           !image.info.stencil.Empty() ? ImageReason::NativeStencilRange :
					           ImageReason::RequestedStencilRange, &image);
					valid = false; break;
				}
				targets.emplace_back(binding.desc.info.data.address, binding.desc.info.data.size);
				targets.emplace_back(image.info.data.address, image.info.data.size);
				if (plain_cmask_owner) {
					// Include the FULL owner metadata range in mapped/physical-uniqueness checks
					// now and on reuse. Its content is never cached or interpreted by this proof.
					targets.emplace_back(image.info.metadata.range.address, image.info.metadata.range.size);
				}
			}
			ShaderRecompiler::IR::ActivityEpoch epoch, after;
			if (!valid) { note(invalid_reason); continue; }
			if (!m_context.ReadActivityEpoch(epoch)) { note(Reason::NativeEpochUnavailable); continue; }
			const bool target_diag = ShaderRecompiler::IR::ActivityDiagnosticsEnabled(info.shader_hash);
			RenderContext::ActivityTargetFailure target_failure;
			if (!m_context.ValidateActivityTargets(targets, target_diag ? &target_failure : nullptr)) {
				if (target_diag) {
					using TargetReason = ShaderRecompiler::IR::ActivityDiagnosticTargetReason;
					using Category = ShaderRecompiler::IR::ActivityDiagnosticTargetCategory;
					ShaderRecompiler::IR::ActivityDiagnosticNativeTargetMetadata metadata;
					metadata.module = runtime.module->id; metadata.target_index = target_failure.index;
					metadata.address = target_failure.address; metadata.size = target_failure.size;
					switch (target_failure.reason) {
					case RenderContext::ActivityTargetFailure::Reason::WrongThread: metadata.reason = TargetReason::WrongThread; break;
					case RenderContext::ActivityTargetFailure::Reason::EmptyTargets: metadata.reason = TargetReason::EmptyTargets; break;
					case RenderContext::ActivityTargetFailure::Reason::ZeroSize: metadata.reason = TargetReason::ZeroSize; break;
					case RenderContext::ActivityTargetFailure::Reason::GpuUnmapped: metadata.reason = TargetReason::GpuUnmapped; break;
					case RenderContext::ActivityTargetFailure::Reason::PhysicalNotUnique: metadata.reason = TargetReason::PhysicalNotUnique; break;
					default: break;
					}
					if (metadata.target_index < targets.size()) {
						// Reconstruct ONLY the append order from already validated prepared metadata.
						// No mapping/uniqueness recheck, binding-owner requery, raw memory read or helper probe.
						uint64_t cursor = 0;
						for (uint32_t i = 0; i < info.info.buffers.size(); ++i) {
							const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
							if (descriptor.Base48() == 0u || descriptor.GetSize() == 0u) continue;
							if (metadata.target_index == cursor || metadata.target_index == cursor + 1u) {
								metadata.category = metadata.target_index == cursor ? Category::RequestedBuffer : Category::OwnerBuffer;
								metadata.resource_index = i; metadata.source = info.info.buffers[i].source;
								metadata.descriptor_index = info.info.buffers[i].descriptor_index;
								break;
							}
							cursor += 2u;
						}
						if (metadata.category == Category::Unknown) {
							for (uint32_t i = 0; i < info.info.images.size(); ++i) {
								const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(snapshot.images[i]);
								if (descriptor.IsNull()) continue;
								const auto& binding = prepared->images[i];
								const auto* image = m_context.GetTextureCache().m_slot_images.try_get(binding.image_id);
								if (image == nullptr) break; // Unknown category is explicit; never invent an owner.
								// Earlier valid guard admits native metadata ONLY through plain-CMASK owner.
								const bool cmask = image->info.metadata.kind == ImageMetadataKind::Cmask;
								const uint64_t added = cmask ? 3u : 2u;
								if (cursor <= metadata.target_index && metadata.target_index - cursor < added) {
									const auto offset = metadata.target_index - cursor;
									metadata.category = offset == 0u ? Category::RequestedImage :
									                    offset == 1u ? Category::NativeImage : Category::NativeCmask;
									metadata.resource_index = i; metadata.source = info.info.images[i].source;
									metadata.descriptor_index = info.info.images[i].descriptor_index;
									break;
								}
								cursor += added;
							}
						}
					}
					ShaderRecompiler::IR::ActivityDiagnosticNativeTargetFailure(info.shader_hash, metadata);
				}
				note(Reason::NativeTargetsRejected); continue;
			}
			if (!m_context.ReadActivityEpoch(after)) { note(Reason::NativeEpochUnavailable); continue; }
			if (after != epoch) { note(Reason::NativeEpochChanged); ShaderRecompiler::IR::ActivityDiagnosticEpoch(info.shader_hash, epoch, after); continue; }
			if (!ShaderRecompiler::IR::ConfirmPreparedActivity(snapshot, epoch, runtime.module->id, targets)) note(Reason::NativeConfirmRejected);
		}
	}
	// All allocation-capable discovery has finished. The following draw/dispatch uses
	// this current command tick; also keep the preparation's own references until reuse.
	for (const auto* prepared: prepared_bindings) {
		m_context.GetBufferCache().RetainPreparedUploads(prepared->uploads);
	}

}

} // namespace Libs::Graphics
