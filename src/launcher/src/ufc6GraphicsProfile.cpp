#include "ufc6GraphicsProfile.h"

#include "cheatFile.h"
#include "configuration.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QStandardPaths>

#include <algorithm>
#include <limits>

namespace Ufc6GraphicsProfile {
namespace {
QString Tr(const char* text) {
	return QCoreApplication::translate("Ufc6GraphicsProfile", text);
}

struct Range {
	quint64 begin;
	quint64 end;
	QString name;
};

QString EnabledRanges(const QJsonDocument& document, QList<Range>* ranges) {
	for (const auto& value: document.object().value(QStringLiteral("mods")).toArray()) {
		const auto mod = value.toObject();
		if (!mod.value(QStringLiteral("enabled")).toBool(true)) continue;
		for (const auto& entry: mod.value(QStringLiteral("memory")).toArray()) {
			const auto write = entry.toObject();
			bool valid = false;
			const auto begin = write.value(QStringLiteral("offset")).toString().toULongLong(&valid, 16);
			const auto size = static_cast<quint64>(write.value(QStringLiteral("on")).toString().size() / 2);
			if (!valid || size == 0 || begin > std::numeric_limits<quint64>::max() - size) {
				return Tr("The enabled cheat contains an invalid address range.");
			}
			ranges->append({begin, begin + size, mod.value(QStringLiteral("name")).toString()});
		}
	}
	return {};
}
} // namespace

QString MergePlan(const Configuration& info, const QByteArray& user_data, QByteArray* merged) {
	if (!info.SupportsUfc6GraphicsProfile()) {
		return Tr("The 540p graphics profile supports only UFC 6 PPSA23566 v01.004.000 (eboot.bin). Disable the profile for this game/version.");
	}
	QFile builtin(QStringLiteral(":/profiles/ufc6-540p.json"));
	if (!builtin.open(QIODevice::ReadOnly)) return Tr("The bundled UFC 6 graphics profile is missing.");
	QJsonDocument profile;
	auto error = Cheats::ParseDocument(builtin.readAll(), &profile);
	if (error.isEmpty()) error = Cheats::ValidateImport(profile, info.title_id, info.gameVersion, info.elf);
	if (!error.isEmpty()) return Tr("Invalid bundled graphics profile: %1").arg(error);
	if (user_data.isEmpty()) {
		*merged = profile.toJson();
		return {};
	}
	QJsonDocument user;
	error = Cheats::ParseDocument(user_data, &user);
	if (error.isEmpty()) error = Cheats::ValidateImport(user, info.title_id, info.gameVersion, info.elf);
	if (!error.isEmpty()) return Tr("Cannot combine the graphics profile with your cheats: %1").arg(error);
	QList<Range> profile_ranges;
	QList<Range> user_ranges;
	error = EnabledRanges(profile, &profile_ranges);
	if (error.isEmpty()) error = EnabledRanges(user, &user_ranges);
	if (!error.isEmpty()) return error;
	for (const auto& selected: user_ranges) {
		for (const auto& graphics: profile_ranges) {
			if (selected.begin < graphics.end && graphics.begin < selected.end) {
				return Tr("Enabled cheat \"%1\" overlaps the 540p graphics profile at 0x%2. Disable that cheat or the graphics profile before launch.")
				    .arg(selected.name).arg(std::max(selected.begin, graphics.begin), 0, 16);
			}
		}
	}
	auto root = user.object();
	auto mods = root.value(QStringLiteral("mods")).toArray();
	for (const auto& mod: profile.object().value(QStringLiteral("mods")).toArray()) mods.append(mod);
	root.insert(QStringLiteral("mods"), mods);
	const auto user_credits = root.value(QStringLiteral("credits"));
	auto credits = user_credits.toArray();
	if (!user_credits.isArray() && !user_credits.isUndefined() && !user_credits.isNull()) {
		credits.append(user_credits);
	}
	for (const auto& credit: profile.object().value(QStringLiteral("credits")).toArray()) {
		if (!credits.contains(credit)) credits.append(credit);
	}
	root.insert(QStringLiteral("credits"), credits);
	root.insert(QStringLiteral("graphics_profile_source"), profile.object().value(QStringLiteral("source")));
	root.insert(QStringLiteral("graphics_profile_license"), profile.object().value(QStringLiteral("license")));
	*merged = QJsonDocument(root).toJson();
	return {};
}

QString PreparePlan(const Configuration& info, const QString& user_path, QString* plan_path) {
	QByteArray user_data;
	QFile user(user_path);
	if (!user_path.isEmpty() && user.exists()) {
		if (user.size() > 4 * 1024 * 1024 || !user.open(QIODevice::ReadOnly)) {
			return Tr("Could not read the selected cheats file, or it exceeds 4 MiB.");
		}
		user_data = user.readAll();
		if (user.error() != QFile::NoError || user_data.isEmpty()) return Tr("Could not read the selected cheats file.");
	}
	QByteArray merged;
	const auto error = MergePlan(info, user_data, &merged);
	if (!error.isEmpty()) return error;
	const auto location = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
	if (location.isEmpty()) return Tr("No writable location is available for the graphics launch plan.");
	const auto directory = QDir(location).filePath(QStringLiteral("graphics-plans"));
	const auto hash = QString::fromLatin1(QCryptographicHash::hash(merged, QCryptographicHash::Sha256).toHex());
	const auto path = QDir(directory).filePath(QStringLiteral("ufc6-540p-%1.json").arg(hash));
	QFile existing(path);
	if (existing.exists()) {
		if (!existing.open(QIODevice::ReadOnly) || existing.readAll() != merged) {
			return Tr("The generated graphics launch plan changed unexpectedly: %1").arg(path);
		}
	} else {
		if (!QDir().mkpath(directory)) return Tr("Could not create the graphics launch-plan directory.");
		QSaveFile output(path);
		if (!output.open(QIODevice::WriteOnly) || output.write(merged) != merged.size() || !output.commit()) {
			return Tr("Could not save the graphics launch plan: %1").arg(output.errorString());
		}
	}
	*plan_path = path;
	return {};
}
} // namespace Ufc6GraphicsProfile
