#include "gameShortcut.h"

#include <QBuffer>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPainter>
#include <QRegularExpression>
#include <QStandardPaths>

#ifdef _WIN32
#include <windows.h> // IWYU pragma: keep

#include <objbase.h>
#include <shlobj.h>
#include <shobjidl.h>
#endif

namespace GameShortcut {

namespace {

QString SafeFileName(const QString& text) {
	QString name = text;
	name.replace(QRegularExpression(QStringLiteral("[\\\\/:*?\"<>|\\x00-\\x1f]")), QStringLiteral("_"));
	name = name.trimmed();
	while (name.endsWith(QLatin1Char('.')) || name.endsWith(QLatin1Char(' '))) {
		name.chop(1);
	}
	return name;
}

// Quotes one token following the CommandLineToArgvW rules.
QString QuoteArg(const QString& arg) {
	if (!arg.isEmpty() && !arg.contains(QRegularExpression(QStringLiteral("[\\s\"]")))) {
		return arg;
	}
	QString out = QStringLiteral("\"");
	qsizetype backslashes = 0;
	for (const QChar c: arg) {
		if (c == QLatin1Char('\\')) {
			backslashes++;
			continue;
		}
		if (c == QLatin1Char('"')) {
			out += QString(backslashes * 2 + 1, QLatin1Char('\\'));
			out += QLatin1Char('"');
		} else {
			out += QString(backslashes, QLatin1Char('\\'));
			out += c;
		}
		backslashes = 0;
	}
	out += QString(backslashes * 2, QLatin1Char('\\'));
	out += QLatin1Char('"');
	return out;
}

// Single-image .ico (256x256, PNG compressed) built from the game's icon.
bool WriteIco(const QByteArray& png_data, const QString& ico_path) {
	QImage source;
	if (png_data.isEmpty() || !source.loadFromData(png_data) || source.isNull()) {
		return false;
	}
	constexpr int kSize = 256;
	QImage canvas(kSize, kSize, QImage::Format_ARGB32);
	canvas.fill(Qt::transparent);
	{
		const QImage scaled = source.scaled(kSize, kSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
		QPainter     painter(&canvas);
		painter.drawImage((kSize - scaled.width()) / 2, (kSize - scaled.height()) / 2, scaled);
	}
	QByteArray png;
	QBuffer    buffer(&png);
	buffer.open(QIODevice::WriteOnly);
	if (!canvas.save(&buffer, "PNG")) {
		return false;
	}

	QByteArray ico;
	const auto put16 = [&ico](quint16 v) {
		ico.append(static_cast<char>(v & 0xff));
		ico.append(static_cast<char>((v >> 8) & 0xff));
	};
	const auto put32 = [&ico](quint32 v) {
		for (int i = 0; i < 4; i++) {
			ico.append(static_cast<char>((v >> (8 * i)) & 0xff));
		}
	};
	put16(0); // reserved
	put16(1); // type: icon
	put16(1); // image count
	ico.append(char(0)); // width  (0 == 256)
	ico.append(char(0)); // height (0 == 256)
	ico.append(char(0)); // palette size
	ico.append(char(0)); // reserved
	put16(1);            // color planes
	put16(32);           // bits per pixel
	put32(static_cast<quint32>(png.size()));
	put32(22); // offset of the image data
	ico.append(png);

	QFile file(ico_path);
	if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		return false;
	}
	return file.write(ico) == ico.size();
}

#ifdef _WIN32
bool CreateLink(const QString& link_path, const QString& target, const QString& arguments,
                const QString& work_dir, const QString& icon_path, const QString& description,
                QString* error) {
	const HRESULT init   = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
	const bool    uninit = SUCCEEDED(init);

	bool          ok   = false;
	IShellLinkW*  link = nullptr;
	IPersistFile* file = nullptr;

	HRESULT hr = CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_IShellLinkW,
	                              reinterpret_cast<void**>(&link));
	if (SUCCEEDED(hr) && link != nullptr) {
		link->SetPath(QDir::toNativeSeparators(target).toStdWString().c_str());
		link->SetArguments(arguments.toStdWString().c_str());
		link->SetWorkingDirectory(QDir::toNativeSeparators(work_dir).toStdWString().c_str());
		link->SetDescription(description.toStdWString().c_str());
		link->SetIconLocation(QDir::toNativeSeparators(icon_path).toStdWString().c_str(), 0);

		hr = link->QueryInterface(IID_IPersistFile, reinterpret_cast<void**>(&file));
		if (SUCCEEDED(hr) && file != nullptr) {
			hr = file->Save(QDir::toNativeSeparators(link_path).toStdWString().c_str(), TRUE);
			ok = SUCCEEDED(hr);
		}
	}
	if (!ok && error != nullptr) {
		*error = QStringLiteral("Windows shell error 0x%1").arg(static_cast<quint32>(hr), 8, 16, QLatin1Char('0'));
	}
	if (file != nullptr) {
		file->Release();
	}
	if (link != nullptr) {
		link->Release();
	}
	if (uninit) {
		CoUninitialize();
	}
	return ok;
}
#endif

} // namespace

Result CreateOnDesktop(const QString& display_name, const QString& unique_id,
                       const QString& emulator_path, const QStringList& emulator_args,
                       const QByteArray& icon_png) {
	Result result;

	const QString desktop = QStandardPaths::writableLocation(QStandardPaths::DesktopLocation);
	if (desktop.isEmpty() || !QDir(desktop).exists()) {
		result.error = QStringLiteral("Desktop folder was not found");
		return result;
	}
	if (!QFileInfo::exists(emulator_path)) {
		result.error = QStringLiteral("Emulator was not found: %1").arg(emulator_path);
		return result;
	}

	QString base_name = SafeFileName(display_name);
	if (base_name.isEmpty()) {
		base_name = SafeFileName(unique_id);
	}
	if (base_name.isEmpty()) {
		base_name = QStringLiteral("Game");
	}
	const QString id_part = SafeFileName(unique_id);

	// Support files (argument file + icon) live in the per-user application data folder.
	QString data_root = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
	if (data_root.isEmpty()) {
		data_root = QDir::tempPath();
	}
	const QString shortcuts_dir = QDir(data_root).filePath(QStringLiteral("shortcuts"));
	if (!QDir().mkpath(shortcuts_dir)) {
		result.error = QStringLiteral("Can't create folder: %1").arg(shortcuts_dir);
		return result;
	}
	const QString support_name = id_part.isEmpty() ? base_name : (id_part + QLatin1Char('_') + base_name);

	// One argument per line (UTF-8); the emulator expands "@file" arguments.
	const QString args_path = QDir(shortcuts_dir).filePath(support_name + QStringLiteral(".args"));
	{
		QFile file(args_path);
		if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
			result.error = QStringLiteral("Can't write file: %1").arg(args_path);
			return result;
		}
		for (const auto& arg: emulator_args) {
			file.write(arg.toUtf8());
			file.write("\n");
		}
	}

#ifdef _WIN32
	QString icon_path = QDir(shortcuts_dir).filePath(support_name + QStringLiteral(".ico"));
	if (!WriteIco(icon_png, icon_path)) {
		icon_path = emulator_path; // fall back to the emulator's own icon
	}

	const QString link_path = QDir(desktop).filePath(base_name + QStringLiteral(".lnk"));
	const QString arguments = QuoteArg(QStringLiteral("@") + QDir::toNativeSeparators(args_path));
	QString       error;
	if (!CreateLink(link_path, emulator_path, arguments, QFileInfo(emulator_path).absolutePath(),
	                icon_path, display_name, &error)) {
		result.error = error;
		return result;
	}
	result.ok   = true;
	result.path = link_path;
	return result;
#elif defined(__linux__)
	// Freedesktop launcher: the icon is the game's PNG.
	QString icon_path = QDir(shortcuts_dir).filePath(support_name + QStringLiteral(".png"));
	QFile   icon_file(icon_path);
	if (icon_png.isEmpty() || !icon_file.open(QIODevice::WriteOnly | QIODevice::Truncate) ||
	    icon_file.write(icon_png) != icon_png.size()) {
		icon_path.clear();
	}
	icon_file.close();

	const auto desktop_quote = [](QString v) {
		v.replace(QLatin1Char('\\'), QStringLiteral("\\\\"));
		v.replace(QLatin1Char('"'), QStringLiteral("\\\""));
		v.replace(QLatin1Char('`'), QStringLiteral("\\`"));
		v.replace(QLatin1Char('$'), QStringLiteral("\\$"));
		return QLatin1Char('"') + v + QLatin1Char('"');
	};

	const QString link_path = QDir(desktop).filePath(base_name + QStringLiteral(".desktop"));
	QFile         link(link_path);
	if (!link.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
		result.error = QStringLiteral("Can't write file: %1").arg(link_path);
		return result;
	}
	QString content = QStringLiteral("[Desktop Entry]\nType=Application\nName=%1\nExec=%2 %3\nPath=%4\nTerminal=true\n")
	                      .arg(display_name, desktop_quote(emulator_path),
	                           desktop_quote(QStringLiteral("@") + args_path).replace(QLatin1Char('%'), QStringLiteral("%%")),
	                           QFileInfo(emulator_path).absolutePath());
	if (!icon_path.isEmpty()) {
		content += QStringLiteral("Icon=%1\n").arg(icon_path);
	}
	link.write(content.toUtf8());
	link.close();
	link.setPermissions(link.permissions() | QFile::ExeUser | QFile::ExeOwner | QFile::ExeGroup);
	result.ok   = true;
	result.path = link_path;
	return result;
#else
	Q_UNUSED(icon_png);
	result.error = QStringLiteral("Desktop shortcuts are not supported on this platform");
	return result;
#endif
}

} // namespace GameShortcut
