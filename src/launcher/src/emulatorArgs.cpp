#include "emulatorArgs.h"

#include "configuration.h"
#include "gameContent.h"

#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>

static void AddFlag(QStringList& args, bool enabled, const char* name) {
	if (enabled) {
		args << name;
	}
}

// UFC 5 and UFC 6 (PS5): PORTHEON's UFC 6 build runs them at 33-40 fps on an RTX 4060 with a
// 1080p display, Immediate presentation and its shader journal. Matched by title ID, or by name
// for other regions' releases.
static bool IsTunedGame(const Configuration& info) {
	static const QStringList title_ids = {QStringLiteral("PPSA23566"), QStringLiteral("PPSA03541")};
	static const QRegularExpression name(QStringLiteral("\\bUFC\\W*[56]\\b"),
	                                     QRegularExpression::CaseInsensitiveOption);
	return title_ids.contains(info.title_id.trimmed().toUpper()) || name.match(info.name).hasMatch();
}

static bool IsUfc6(const Configuration& info) {
	static const QRegularExpression name(QStringLiteral("\\bUFC\\W*6\\b"),
	                                     QRegularExpression::CaseInsensitiveOption);
	return info.title_id.trimmed().compare(QStringLiteral("PPSA23566"), Qt::CaseInsensitive) == 0 ||
	       name.match(info.name).hasMatch();
}

QStringList CreateEmulatorArgs(const Configuration& info, const QString& patch_plan) {
	QStringList args;
	const bool  tuned = info.game_tuning_enabled && IsTunedGame(info);
	// UFC 6's asynchronous launch path stalled on the splash screen in local testing, while
	// the synchronous 60 Hz baseline reached the title/menu. Keep that compatibility preset
	// separate from the selected frame-generation backend, factor and enable state.
	const bool  ufc6_compatibility = tuned && IsUfc6(info);
	auto        r = Configuration::NormalizeResolution(info.screen_resolution).split('x');

	if (r.size() != 2) {
		return {};
	}

	args << "--screen-width" << r.at(0);
	args << "--screen-height" << r.at(1);
	args << "--user-name" << info.user_name;
	args << "--user-id" << QString::number(info.user_id);
	if (!info.audio_input_device.isEmpty()) {
		args << "--mic" << info.audio_input_device;
	}
	if (!info.controller.color.isEmpty()) {
		args << "--controller-color" << info.controller.color;
	}
	args << "--controller-volume" << QString::number(info.controller.speaker_volume);
	args << "--controller-vibration" << QString::number(info.controller.vibration_intensity);
	args << "--keyboard-player" << QString::number(info.controller.keyboard_player);
	args << "--present-mode"
	     << (tuned ? EnumToText(Configuration::PresentMode::Immediate) : EnumToText(info.present_mode));
	if (info.gpu_index >= 0) {
		args << "--gpu" << QString::number(info.gpu_index);
	}
	AddFlag(args, info.fullscreen_enabled, "--fullscreen");
	AddFlag(args, info.hdr_enabled, "--hdr");
	AddFlag(args, info.output_1080p_enabled || tuned, "--output-1080p");
	AddFlag(args, info.frame_generation_enabled, "--frame-generation");
	args << "--frame-generation-backend" << info.frame_generation_backend;
	args << "--frame-generation-multiplier" << QString::number(info.frame_generation_multiplier);
	AddFlag(args, info.hide_cursor_enabled, "--hide-cursor");
	AddFlag(args, info.stretch_enabled, "--stretch");
	AddFlag(args, info.readback_linear_images, "--readback-linear-images");
	AddFlag(args, info.readback_tiled_images, "--readback-tiled-images");
	AddFlag(args, info.readback_storage_images, "--readback-storage-images");
	AddFlag(args, info.sync_raw_image_buffers, "--sync-raw-image-buffers");
	args << "--flush-draws" << QString::number(info.flush_draws);
	AddFlag(args, !info.trophy_enabled, "--no-trophy-notifications");
	args << "--trophy-duration" << QString::number(info.trophy_duration_seconds);
	AddFlag(args, info.perf_overlay_enabled, "--perf-overlay");
	AddFlag(args, info.skip_notice_screen, "--skip-notice-screen");
	AddFlag(args, info.network_connected, "--network-connected");
	AddFlag(args, info.psn_emulation_enabled, "--emulate-psn");
	AddFlag(args, info.tessellation_enabled, "--tessellation");
	AddFlag(args, info.occlusion_queries_enabled, "--occlusion-queries");
	AddFlag(args, info.relaxed_shader_float, "--relaxed-shader-float");
	AddFlag(args, info.shader_cache_enabled || tuned, "--shader-cache");
	AddFlag(args, info.async_shaders_enabled && !ufc6_compatibility, "--async-shaders");
	AddFlag(args, info.async_shaders_enabled && info.async_compute_enabled && !ufc6_compatibility,
	        "--async-compute");
	AddFlag(args, info.async_pipelines_enabled && !ufc6_compatibility, "--async-pipelines");
	if (info.gpu_timestamp_scale != 100) {
		args << "--gpu-timestamp-scale" << QString::number(info.gpu_timestamp_scale);
	}
	args << "--vram-budget-mb"
	     << (info.vram_budget_mb > 0 ? QString::number(info.vram_budget_mb) : QStringLiteral("auto"));
	if (!info.skip_shaders.isEmpty()) {
		args << "--skip-shaders" << info.skip_shaders;
	}
	args << "--vblank-frequency" << QString::number(ufc6_compatibility ? 60 : info.vblank_frequency);
	args << "--console-language" << QString::number(info.console_language);
	AddFlag(args, info.vulkan_validation_enabled, "--vulkan-validation");
	AddFlag(args, info.shader_validation_enabled, "--shader-validation");
	args << "--shader-log-direction" << EnumToText(info.shader_log_direction);
	args << "--shader-log-folder" << info.shader_log_folder;
	AddFlag(args, info.command_buffer_dump_enabled, "--command-buffer-dump");
	args << "--command-buffer-dump-folder" << info.command_buffer_dump_folder;
	args << "--printf-direction" << EnumToText(info.printf_direction);
	args << "--printf-output-file" << info.printf_output_file;
	AddFlag(args, info.profiler_enabled, "--profile");
	AddFlag(args, info.pro_enabled, "--trinity");
	if (info.pro_enabled) {
		args << "--pssr-upscaler" << EnumToText(info.pssr_upscaler);
		if (info.pssr_upscaler == Configuration::PssrUpscaler::DLSS) {
			args << "--dlss-folder" << info.dlss_folder;
		} else if (info.pssr_upscaler == Configuration::PssrUpscaler::FSR) {
			args << "--fsr-folder" << info.fsr_folder;
		}
	}
	AddFlag(args, info.amd_cpu_enabled, "--amd-cpu");
	AddFlag(args, info.guest_tsc_enabled, "--guest-tsc");
#if defined(_WIN32)
	AddFlag(args, info.red_zone_protection_enabled, "--redzone");
#endif
	for (const auto& binding: info.host_input_mapping) {
		args << "--keymap" << binding;
	}
	AddFlag(args, info.renderdoc_enabled, "--rd");

	QString game = info.basedir;
	if (!info.elf.isEmpty() && !GameContent::IsArchive(info.basedir)) {
		game = QDir(info.basedir).filePath(info.elf);
	}
	args << "--game" << game;
	if (!info.addons_dir.isEmpty()) {
		args << "--addcont-dir" << info.addons_dir;
	}
	for (const auto& entry: info.addcont) {
		args << "--addcont" << entry;
	}

	if (QFileInfo::exists(patch_plan)) {
		args << "--game-patch" << patch_plan;
	}

	return args;
}
