#include "configuration.h"
#include "configurationEditDialog.h"
#include "launcherTheme.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QTimer>

#include <cstdio>
#include <cstdlib>

namespace {
void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "LauncherSettingsDialogTests: %s\n", message);
		std::exit(1);
	}
}
template <typename T>
T* Control(QObject& dialog, const char* name) {
	auto* result = dialog.findChild<T*>(QString::fromLatin1(name));
	Check(result != nullptr, name);
	return result;
}
int VisibleCategories(QListWidget* list) {
	int count = 0;
	for (int i = 0; i < list->count(); ++i)
		count += !list->item(i)->isHidden();
	return count;
}
} // namespace

int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	std::puts("Starting settings dialog regression checks");
	QApplication app(argc, argv);
	LauncherTheme::Initialize(app);
	QTemporaryDir temporary;
	Check(temporary.isValid(), "temporary folder");
	Configuration config;
	config.frame_generation_backend       = QStringLiteral("multi");
	config.frame_generation_multiplier    = 6;
	config.frame_generation_enabled       = true;
	config.screen_resolution              = QStringLiteral("1920x1080");
	config.user_name                      = QStringLiteral("UI test");
	config.controller.keyboard_player     = 2;
	config.controller.color               = QStringLiteral("#123456");
	config.controller.vibration_intensity = 43;
	config.controller.speaker_volume      = 27;
	const auto original                   = config.GameSettings();
	{
		std::puts("Constructing global settings");
		ConfigurationEditDialog dialog(config);
		dialog.SetGlobalSettings({temporary.path()}, temporary.path());
		dialog.show();
		app.processEvents();
		auto* categories = Control<QListWidget>(dialog, "settings_categories");
		auto* search     = Control<QLineEdit>(dialog, "settings_search");
		Check(VisibleCategories(categories) == 10, "global categories");
		search->setText(QStringLiteral("frame generation"));
		const auto* current_category = categories->currentItem();
		const auto current_name = current_category ? current_category->text() : QStringLiteral("<none>");
		if (VisibleCategories(categories) != 1 ||
		    current_name != QStringLiteral("Enhancements")) {
			std::fprintf(stderr, "FG search visible=%d current=%s\n", VisibleCategories(categories),
			             qPrintable(current_name));
			for (int i = 0; i < categories->count(); ++i) {
				if (!categories->item(i)->isHidden())
					std::fprintf(stderr, "FG search category: %s\n", qPrintable(categories->item(i)->text()));
			}
		}
		Check(VisibleCategories(categories) == 1 &&
		          categories->currentItem()->text() == "Enhancements",
		      "search finds frame generation in its category");
		auto* multiplier = Control<QComboBox>(dialog, "comboBox_frame_generation_multiplier");
		Check(multiplier->currentData().toInt() == 6 && multiplier->isEnabled(),
		      "Multi selection preserved");
		search->setText(QStringLiteral("no-matching-option-123"));
		Check(VisibleCategories(categories) == 0, "empty search result");
		search->clear();
		Check(VisibleCategories(categories) == 10, "search clear restores categories");
		for (int i = 0; i < categories->count(); ++i) {
			categories->setCurrentRow(i);
			Check(Control<QTabWidget>(dialog, "settings_tabs")->currentIndex() == i,
			      "navigation selects matching content");
			const auto preview_dir = qEnvironmentVariable("KYTY_UI_PREVIEW_DIR");
			if (!preview_dir.isEmpty()) {
				app.processEvents();
				Check(dialog.grab().save(
				          QDir(preview_dir).filePath(categories->item(i)->text() + ".png")),
				      "save settings preview");
			}
		}
		// Saving while a filter hides the other categories must keep every setting.
		search->setText(QStringLiteral("microphone"));
		std::puts("Saving global settings after search/navigation");
		Control<QPushButton>(dialog, "ok_button")->click();
		Check(dialog.result() == QDialog::Accepted, "global dialog saves");
		Check(config.GameSettings() == original,
		      "all game settings survive category navigation and save");
		Check(config.controller.keyboard_player == 2 && config.controller.color == "#123456" &&
		          config.controller.vibration_intensity == 43 &&
		          config.controller.speaker_volume == 27,
		      "global controller settings survive save");
		Check(dialog.GetGameDirectories() == QStringList {temporary.path()} &&
		          QDir::fromNativeSeparators(dialog.GetAddonsDirectory()) == temporary.path(),
		      "library folders survive save");
	}
	{
		std::puts("Checking per-game validation and cancel");
		ConfigurationEditDialog dialog(config);
		dialog.show();
		app.processEvents();
		auto* categories = Control<QListWidget>(dialog, "settings_categories");
		Check(VisibleCategories(categories) == 8, "per-game settings omit global-only categories");
		// Invalid input must reveal the owning category even when search has hidden it.
		auto* search     = Control<QLineEdit>(dialog, "settings_search");
		auto* resolution = Control<QComboBox>(dialog, "comboBox_screen_resolution");
		resolution->setCurrentText(QStringLiteral("invalid"));
		search->setText(QStringLiteral("microphone"));
		bool validation_seen = false;
		QTimer::singleShot(0, [&] {
			for (auto* window: app.topLevelWidgets()) {
				if (auto* message = qobject_cast<QMessageBox*>(window)) {
					validation_seen = true;
					message->accept();
				}
			}
		});
		Control<QPushButton>(dialog, "ok_button")->click();
		Check(validation_seen && dialog.result() != QDialog::Accepted,
		      "invalid resolution blocks save");
		Check(search->text().isEmpty() && categories->currentItem()->text() == "Graphics",
		      "validation reveals the correct category");
		Check(config.GameSettings() == original, "invalid save leaves configuration untouched");
		resolution->setCurrentText(QStringLiteral("2560x1440"));
		Control<QCheckBox>(dialog, "checkBox_frame_generation")->setChecked(false);
		Control<QPushButton>(dialog, "cancel_button")->click();
		Check(config.GameSettings() == original, "cancel discards changes across categories");
	}
	std::puts(
	    "PASS: settings navigation, search, global/per-game scope, save, validation and cancel");
	return 0;
}
