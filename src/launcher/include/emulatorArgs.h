#ifndef LAUNCHER_EMULATOR_ARGS_H_
#define LAUNCHER_EMULATOR_ARGS_H_

#include <QString>
#include <QStringList>

class Configuration;

// Normal launches and desktop shortcuts must use this same argument list.
[[nodiscard]] QStringList CreateEmulatorArgs(const Configuration& info, const QString& patch_plan = {});

#endif
