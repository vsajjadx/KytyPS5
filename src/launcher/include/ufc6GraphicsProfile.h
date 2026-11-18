#ifndef LAUNCHER_UFC6_GRAPHICS_PROFILE_H_
#define LAUNCHER_UFC6_GRAPHICS_PROFILE_H_

#include <QByteArray>
#include <QString>

class Configuration;

namespace Ufc6GraphicsProfile {
// Returns an error without modifying either output or the user's source file.
[[nodiscard]] QString MergePlan(const Configuration& info, const QByteArray& user_data,
                                 QByteArray* merged);
// Writes an immutable, content-addressed launch/shortcut plan; keeps the user's file intact.
[[nodiscard]] QString PreparePlan(const Configuration& info, const QString& user_path,
                                   QString* plan_path);
}

#endif
