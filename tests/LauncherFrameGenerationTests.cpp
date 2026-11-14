#include "configuration.h"
#include "emulatorArgs.h"
#include "common/emulatorConfig.h"

#include <QCoreApplication>
#include <QJsonObject>
#include <QSettings>
#include <QTemporaryDir>

#include <cstdio>
#include <cstdlib>

namespace {
void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "LauncherFrameGenerationTests: %s\n", message);
		std::abort();
	}
}
QString Value(const QStringList& args, const QString& key) {
	const auto index = args.indexOf(key);
	Check(index >= 0 && index + 1 < args.size() && args.count(key) == 1,
	      "missing or duplicated command-line option");
	return args.at(index + 1);
}
void CheckSelection(const Configuration& cfg, const char* backend, int multiplier, bool enabled) {
	Check(cfg.frame_generation_backend == QLatin1String(backend) &&
	      cfg.frame_generation_multiplier == multiplier && cfg.frame_generation_enabled == enabled,
	      "backend/multiplier/enable selection changed");
}
} // namespace

int main(int argc, char** argv) {
	QCoreApplication application(argc, argv);
	QTemporaryDir temporary;
	Check(temporary.isValid(), "temporary settings directory");

	Configuration legacy;
	QSettings missing(temporary.filePath("legacy.ini"), QSettings::IniFormat);
	legacy.ReadSettings(&missing);
	CheckSelection(legacy, "fsr", 2, false);
	// Old enabled profiles still use their original FSR path.
	missing.setValue("frame_generation_enabled", true);
	legacy.ReadSettings(&missing);
	CheckSelection(legacy, "fsr", 2, true);

	// Malformed stored settings fail closed to defaults, retaining the enable checkbox.
	missing.setValue("frame_generation_backend", "not-a-backend");
	missing.setValue("frame_generation_multiplier", "not-an-integer");
	legacy.ReadSettings(&missing);
	CheckSelection(legacy, "fsr", 2, true);
	for (const auto value: {QStringLiteral("1"), QStringLiteral("7"), QStringLiteral("-1"),
	                        QStringLiteral("4294967295"), QStringLiteral("2.5")}) {
		missing.setValue("frame_generation_multiplier", value);
		legacy.ReadSettings(&missing);
		CheckSelection(legacy, "fsr", 2, true);
	}

	for (int multiplier = 2; multiplier <= 6; ++multiplier) {
		Configuration selected;
		selected.frame_generation_backend = QStringLiteral("multi");
		selected.frame_generation_multiplier = multiplier;
		selected.frame_generation_enabled = false;
		selected.basedir = temporary.path();
		selected.game_tuning_enabled = false;
		selected.fullscreen_enabled = true;
		selected.hdr_enabled = true;
		selected.output_1080p_enabled = true;
		selected.host_input_mapping = {QStringLiteral("Cross=J"), QStringLiteral("Circle=K")};
		// Save/load is the path used by stored game profiles and shortcut configuration.
		QSettings settings(temporary.filePath(QStringLiteral("factor%1.ini").arg(multiplier)),
		                   QSettings::IniFormat);
		selected.WriteSettings(&settings);
		settings.sync();
		Check(settings.status() == QSettings::NoError, "writing settings");
		Configuration restored;
		restored.ReadSettings(&settings);
		restored.CopyGameInfoFrom(selected);
		CheckSelection(restored, "multi", multiplier, false);
		Configuration copied;
		copied.CopyEmulatorSettingsFrom(selected);
		copied.CopyGameInfoFrom(selected);
		CheckSelection(copied, "multi", multiplier, false);
		const auto run_args = CreateEmulatorArgs(selected);
		const auto shortcut_args = CreateEmulatorArgs(restored);
		Check(run_args == shortcut_args && run_args == CreateEmulatorArgs(copied),
		      "normal-run, persisted-shortcut and copied-profile arguments differ");
		Check(Value(run_args, "--frame-generation-backend") == QStringLiteral("multi") &&
		      Value(run_args, "--frame-generation-multiplier") == QString::number(multiplier),
		      "FG-off launch lost the chosen backend/multiplier needed by later F3");
		Check(!run_args.contains("--frame-generation"), "selection enabled frame generation");
		Check(run_args.contains("--fullscreen") && run_args.contains("--hdr") &&
		      run_args.contains("--output-1080p") && run_args.count("--keymap") == 2 &&
		      Value(run_args, "--game").endsWith("eboot.bin"), "unrelated arguments changed");
		selected.frame_generation_enabled = true;
		auto enabled_args = CreateEmulatorArgs(selected);
		Check(enabled_args.count("--frame-generation") == 1, "enable checkbox not emitted once");
		enabled_args.removeOne("--frame-generation");
		Check(enabled_args == run_args, "enable checkbox changed backend or other arguments");
	}

	// The UFC 6 compatibility preset must survive unsafe saved/global settings without
	// changing the user's FG selection. Other titles and the tuning opt-out retain their values.
	for (int multiplier = 2; multiplier <= 6; ++multiplier) {
		for (const bool enabled: {false, true}) {
			Configuration ufc6;
			ufc6.title_id = QStringLiteral("PPSA23566");
			ufc6.name = QStringLiteral("EA SPORTS UFC 6");
			ufc6.game_tuning_enabled = true;
			ufc6.async_shaders_enabled = true;
			ufc6.async_compute_enabled = true;
			ufc6.async_pipelines_enabled = true;
			ufc6.vblank_frequency = 360;
			ufc6.frame_generation_backend = QStringLiteral("multi");
			ufc6.frame_generation_multiplier = multiplier;
			ufc6.frame_generation_enabled = enabled;
			const auto safe_args = CreateEmulatorArgs(ufc6);
			Check(Value(safe_args, "--vblank-frequency") == QStringLiteral("60") &&
			      !safe_args.contains("--async-shaders") &&
			      !safe_args.contains("--async-compute") &&
			      !safe_args.contains("--async-pipelines"), "UFC 6 compatibility preset missing");
			Check(Value(safe_args, "--frame-generation-backend") == QStringLiteral("multi") &&
			      Value(safe_args, "--frame-generation-multiplier") == QString::number(multiplier) &&
			      safe_args.contains("--frame-generation") == enabled,
			      "UFC 6 compatibility preset changed FG selection");
			Check(!safe_args.contains("--skip-notice-screen"), "UFC 6 preset skipped notices");
			ufc6.skip_notice_screen = true;
			Check(CreateEmulatorArgs(ufc6).contains("--skip-notice-screen"),
			      "UFC 6 preset discarded skip-notice selection");
			ufc6.skip_notice_screen = false;
			Check(ufc6.vblank_frequency == 360 && ufc6.async_shaders_enabled &&
			      ufc6.async_compute_enabled && ufc6.async_pipelines_enabled,
			      "argument generation overwrote saved settings");
			ufc6.title_id = QStringLiteral(" ppsa23566 ");
			ufc6.name = QStringLiteral("Regional title");
			Check(CreateEmulatorArgs(ufc6) == safe_args,
			      "UFC 6 title ID missed compatibility preset");
			ufc6.title_id = QStringLiteral("OTHER-REGION");
			ufc6.name = QStringLiteral("EA SPORTS UFC 6");
			Check(CreateEmulatorArgs(ufc6) == safe_args,
			      "UFC 6 name fallback missed compatibility preset");
			ufc6.game_tuning_enabled = false;
			const auto opted_out = CreateEmulatorArgs(ufc6);
			Check(Value(opted_out, "--vblank-frequency") == QStringLiteral("360") &&
			      opted_out.contains("--async-shaders") && opted_out.contains("--async-compute") &&
			      opted_out.contains("--async-pipelines"), "UFC 6 tuning opt-out ignored");
			for (const auto& title: {QStringLiteral("PPSA03541"), QStringLiteral("PPSA03671")}) {
				ufc6.game_tuning_enabled = true;
				ufc6.title_id = title;
				ufc6.name = title == QStringLiteral("PPSA03541") ? QStringLiteral("UFC 5")
				                                                   : QStringLiteral("Marvel's Wolverine");
				const auto other_args = CreateEmulatorArgs(ufc6);
				Check(Value(other_args, "--vblank-frequency") == QStringLiteral("360") &&
				      other_args.contains("--async-shaders") && other_args.contains("--async-compute") &&
				      other_args.contains("--async-pipelines"), "UFC 6 policy changed another game");
			}
		}
	}

	Configuration json;
	QString error;
	Configuration legacy_json;
	Check(legacy_json.SetGameSettings({{"frame_generation_enabled", "true"}}, error),
	      "legacy JSON profile rejected");
	CheckSelection(legacy_json, "fsr", 2, true);
	Check(json.SetGameSettings({{"frame_generation_backend", "multi"},
	                            {"frame_generation_multiplier", "5"}}, error), "JSON Multi 5 rejected");
	CheckSelection(json, "multi", 5, false);
	Check(json.SetGameSettings({{"frame_generation_backend", "fsr"}}, error), "JSON FSR rejected");
	CheckSelection(json, "fsr", 5, false); // FSR uses 2x, preserving the inactive Multi request.
	for (const auto invalid: {QStringLiteral("0"), QStringLiteral("1"), QStringLiteral("7"),
	                          QStringLiteral("-1"), QStringLiteral("5x"), QStringLiteral("2.5")}) {
		Configuration candidate;
		Check(!candidate.SetGameSettings({{"frame_generation_multiplier", invalid}}, error) &&
		      !error.isEmpty(), "invalid JSON multiplier accepted");
	}
	Configuration unknown;
	Check(!unknown.SetGameSettings({{"frame_generation_backend", "Multi"}}, error) &&
	      !error.isEmpty(), "invalid JSON backend accepted");
	Configuration numeric;
	Check(!numeric.SetGameSettings({{"frame_generation_multiplier", 5}}, error),
	      "non-string JSON setting accepted");

	Config::Initialize();
	Config::ConfigOptions options;
	Check(!options.frame_generation_enabled && options.frame_generation_backend == "fsr" &&
	      options.frame_generation_multiplier == 2, "legacy emulator defaults changed");
	options.frame_generation_backend = "multi";
	options.frame_generation_multiplier = 5;
	Config::Load(options);
	Check(!Config::FrameGenerationEnabled() && Config::FrameGenerationBackend() == "multi" &&
	      Config::FrameGenerationMultiplier() == 5, "emulator accessors lost an FG-off selection");
	Config::Shutdown();
	std::puts("Launcher frame generation settings and shared arguments PASS");
}
