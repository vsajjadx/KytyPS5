#include "configuration.h"
#include "emulatorArgs.h"
#include "ufc6GraphicsProfile.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSettings>
#include <QStandardPaths>
#include <QTemporaryDir>

#include <cstdio>
#include <cstdlib>

namespace {
void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "LauncherGraphicsProfileTests: %s\n", message);
		std::exit(1);
	}
}
QByteArray Read(const QString& path) {
	QFile file(path);
	Check(file.open(QIODevice::ReadOnly), "read fixture/plan");
	return file.readAll();
}
QString Value(const QStringList& args, const QString& key) {
	const auto index = args.indexOf(key);
	Check(index >= 0 && index + 1 < args.size() && args.count(key) == 1, "unique argument");
	return args.at(index + 1);
}
QJsonObject Mod(const QString& name, const QString& offset) {
	return {{"name", name}, {"enabled", true},
	        {"memory", QJsonArray {QJsonObject {{"offset", offset}, {"off", "AABB"}, {"on", "CCDD"}}}}};
}
}

int main(int argc, char** argv) {
	QCoreApplication app(argc, argv);
	QCoreApplication::setApplicationName(QStringLiteral("KytyGraphicsProfileTests"));
	QStandardPaths::setTestModeEnabled(true);
	QTemporaryDir directory;
	Check(directory.isValid(), "temporary directory");
	Configuration info;
	info.title_id = QStringLiteral("PPSA23566");
	info.gameVersion = QStringLiteral("01.004.000");
	info.elf = QStringLiteral("eboot.bin");
	info.basedir = directory.path();
	info.frame_generation_enabled = true;
	info.frame_generation_backend = QStringLiteral("multi");
	info.frame_generation_multiplier = 6;
	Check(!info.ufc6_540p_enabled && info.SupportsUfc6GraphicsProfile(), "default off and exact supported game");
	const auto baseline = CreateEmulatorArgs(info);
	Check(!baseline.contains("--game-patch"), "default-off injected a patch");
	info.ufc6_540p_enabled = true;
	QSettings settings(directory.filePath("profile.ini"), QSettings::IniFormat);
	info.WriteSettings(&settings);
	Configuration restored;
	restored.ReadSettings(&settings);
	restored.CopyGameInfoFrom(info);
	Configuration copied;
	copied.CopyEmulatorSettingsFrom(info);
	copied.CopyGameInfoFrom(info);
	Check(restored.ufc6_540p_enabled && copied.ufc6_540p_enabled, "profile save/load/copy");

	const auto builtin = QJsonDocument::fromJson(Read(":/profiles/ufc6-540p.json"));
	const auto profile_mods = builtin.object().value("mods").toArray();
	Check(profile_mods.size() == 3, "graphics-only profile count");
	QByteArray fingerprint;
	for (const auto& value: profile_mods) {
		const auto mod = value.toObject();
		Check(mod.value("enabled").toBool(), "all three graphics patches enabled");
		for (const auto& entry: mod.value("memory").toArray()) {
			const auto write = entry.toObject();
			fingerprint += write.value("offset").toString().toUtf8() + '|' +
			               write.value("off").toString().toUtf8() + '|' +
			               write.value("on").toString().toUtf8() + '\n';
		}
	}
	Check(QCryptographicHash::hash(fingerprint, QCryptographicHash::Sha256).toHex() ==
	          QByteArray("3b4d9b9b5b102ba8d200b6eb3c61962755084d4e4cc207054bd19f6a1fed51de"), "validated graphics patch bytes changed");

	auto disabled = profile_mods.first().toObject();
	disabled.insert("enabled", false);
	const QJsonArray user_mods {Mod("Selected user cheat", "0x1000"), disabled};
	const QJsonObject user {{"id", "PPSA23566"}, {"version", "01.004.000"}, {"process", "eboot.bin"},
	                        {"credits", QJsonArray {"User author"}}, {"mods", user_mods}};
	const auto original = QJsonDocument(user).toJson();
	const auto user_path = directory.filePath("user.json");
	{
		QFile output(user_path);
		Check(output.open(QIODevice::WriteOnly) && output.write(original) == original.size(), "write user fixture");
	}
	QString error;
	const auto args = CreateEmulatorArgs(info, user_path, &error);
	Check(!args.isEmpty() && error.isEmpty(), "profile launch plan generation");
	const auto plan_path = Value(args, "--game-patch");
	Check(plan_path != user_path && Read(user_path) == original, "user cheat file overwritten");
	Check(CreateEmulatorArgs(restored, user_path, &error) == args &&
	      CreateEmulatorArgs(copied, user_path, &error) == args,
	      "run, shortcut and restored/copied profile differ");
	const auto merged = QJsonDocument::fromJson(Read(plan_path)).object();
	const auto merged_mods = merged.value("mods").toArray();
	Check(merged_mods.size() == 5 && merged_mods.at(0) == user_mods.at(0) &&
	      merged_mods.at(1) == user_mods.at(1), "user selections were changed or dropped");
	for (int i = 0; i < 3; ++i) Check(merged_mods.at(i + 2) == profile_mods.at(i), "graphics patch lost");
	Check(merged.value("credits").toArray().contains("User author"), "user credit lost");
	Check(args.count("--frame-generation") == 1 && Value(args, "--frame-generation-backend") == "multi" &&
	      Value(args, "--frame-generation-multiplier") == "6", "graphics profile changed FG");
	Check(args.contains("--output-1080p"), "graphics profile omitted required 1080p output");
	info.game_tuning_enabled = false;
	Check(CreateEmulatorArgs(info, user_path, &error).contains("--output-1080p"),
	      "profile relies on unrelated recommended-game setting");

	QByteArray combined("untouched");
	auto conflicting = user;
	conflicting.insert("mods", QJsonArray {Mod("Overlapping user cheat", "0x1B453D7")});
	error = Ufc6GraphicsProfile::MergePlan(info, QJsonDocument(conflicting).toJson(), &combined);
	Check(error.contains("Overlapping user cheat") && error.contains("overlaps") && combined == "untouched",
	      "enabled overlapping write did not fail clearly and atomically");
	Check(!Ufc6GraphicsProfile::MergePlan(info, QByteArray("not JSON"), &combined).isEmpty(),
	      "malformed selected cheat accepted");
	auto wrong_user = user;
	wrong_user.insert("version", "01.003.000");
	Check(!Ufc6GraphicsProfile::MergePlan(info, QJsonDocument(wrong_user).toJson(), &combined).isEmpty(),
	      "wrong-version user cheat accepted");
	info.gameVersion = QStringLiteral("01.005.000");
	Check(CreateEmulatorArgs(info, {}, &error).isEmpty() && error.contains("01.004.000"), "wrong game version accepted");
	info.gameVersion = QStringLiteral("01.004.000");
	info.title_id = QStringLiteral("PPSA03541");
	Check(CreateEmulatorArgs(info, {}, &error).isEmpty(), "profile applied to another title");
	info.title_id = QStringLiteral("PPSA23566");
	info.elf = QStringLiteral("other.elf");
	Check(CreateEmulatorArgs(info, {}, &error).isEmpty(), "profile applied to another executable");
	info.ufc6_540p_enabled = false;
	Check(CreateEmulatorArgs(info, user_path, &error).contains(user_path), "profile-off changed normal cheat handling");
	std::puts("PASS: scoped graphics preset, immutable merge, selection preservation and conflict checks");
}
