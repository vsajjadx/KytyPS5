#ifndef GAME_SHORTCUT_H
#define GAME_SHORTCUT_H

#include <QByteArray>
#include <QString>
#include <QStringList>

namespace GameShortcut {

struct Result {
	bool    ok = false;
	QString path;  // created shortcut (on success)
	QString error; // human readable reason (on failure)
};

// Creates a desktop shortcut that starts `emulator_path` directly with `emulator_args` for one
// game, using the game's own icon (PNG data from sce_sys/icon0.png, may be empty).
// The arguments are stored in an argument file ("@file") so the shortcut command line stays short.
[[nodiscard]] Result CreateOnDesktop(const QString& display_name, const QString& unique_id,
                                     const QString& emulator_path, const QStringList& emulator_args,
                                     const QByteArray& icon_png);

} // namespace GameShortcut

#endif // GAME_SHORTCUT_H
