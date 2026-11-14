#ifndef GAME_VIEWS_H
#define GAME_VIEWS_H

#include "gameContent.h"

#include <QGuiApplication>
#include <QIcon>
#include <QListView>
#include <QPaintEvent>
#include <QPainter>
#include <QPalette>
#include <QPixmap>
#include <QResizeEvent>
#include <QStackedWidget>
#include <QString>
#include <QStyledItemDelegate>
#include <QTreeWidget>

inline constexpr int GAME_NAME_COLUMN             = 0;
inline constexpr int GAME_SERIAL_COLUMN           = 1;
inline constexpr int GAME_VERSION_COLUMN          = 2;
inline constexpr int GAME_FIRMWARE_VERSION_COLUMN = 3;
inline constexpr int GAME_SIZE_COLUMN             = 4;
inline constexpr int GAME_PATH_COLUMN             = 5;
inline constexpr int GAME_STATUS_COLUMN           = 6;

// Holds the transparent list and grid views and paints the selected game's pic0 behind both.
class GameViews: public QStackedWidget {
public:
	using QStackedWidget::QStackedWidget;

	void SetBackgroundImage(const QString& key) {
		if (m_path == key) {
			return;
		}
		m_path   = key;
		m_source = QPixmap();
		if (!key.isEmpty()) {
			m_source.loadFromData(GameContent::ReadFile(key, QStringLiteral("sce_sys/pic0.png"),
			                                            GameContent::MaxImageSize));
		}
		UpdateScaledBackground();
		update();
	}

protected:
	void resizeEvent(QResizeEvent* event) override {
		QStackedWidget::resizeEvent(event);
		UpdateScaledBackground();
	}

	void paintEvent(QPaintEvent* event) override {
		QPainter painter(this);
		auto     overlay = palette().color(QPalette::Base);
		if (!m_scaled.isNull()) {
			painter.drawPixmap((width() - m_scaled.width()) / 2, (height() - m_scaled.height()) / 2,
			                   m_scaled);
			overlay.setAlpha(160);
		}
		painter.fillRect(event->rect(), overlay);
	}

private:
	void UpdateScaledBackground() {
		m_scaled = m_source.isNull() ? QPixmap()
		                             : m_source.scaled(size(), Qt::KeepAspectRatioByExpanding,
		                                               Qt::FastTransformation);
	}

	QString m_path;
	QPixmap m_source;
	QPixmap m_scaled;
};

// Marks a title that supports PS5 Pro mode with a pill in the top-right corner of its cover.
inline void PaintProBadge(QPainter* painter, const QRect& cover, QFont font, const QString& text) {
	font.setBold(true);
	const QFontMetrics metrics(font);
	const int          inset = metrics.height() / 3;
	QRect              badge(0, 0, metrics.horizontalAdvance(text) + 2 * inset, metrics.height());
	badge.moveTopRight(cover.topRight() + QPoint(-inset, inset));
	painter->save();
	painter->setRenderHint(QPainter::Antialiasing);
	painter->setPen(Qt::NoPen);
	painter->setBrush(QColor(0x534ab7));
	painter->drawRoundedRect(badge, 4, 4);
	painter->setFont(font);
	painter->setPen(QColor(0xeeedfe));
	painter->drawText(badge, Qt::AlignCenter, text);
	painter->restore();
}

// A card on a translucent panel, so the text keeps its contrast over the game background.
class GameGridDelegate: public QStyledItemDelegate {
public:
	using QStyledItemDelegate::QStyledItemDelegate;

	QSize sizeHint(const QStyleOptionViewItem& option,
	               const QModelIndex& /*index*/) const override {
		return option.decorationSize +
		       QSize(2 * CARD_PADDING, 3 * CARD_PADDING + 2 * option.fontMetrics.lineSpacing());
	}

	void paint(QPainter* painter, const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override {
		// The stylesheet made the view's own palette transparent, so take the panel from the app's.
		auto panel      = QGuiApplication::palette().color(QPalette::Window);
		auto text_color = option.palette.color(QPalette::Text);
		panel.setAlpha(210);
		painter->save();
		painter->setRenderHint(QPainter::Antialiasing);
		painter->setPen(option.state.testFlag(QStyle::State_Selected) ? QPen(text_color)
		                                                              : Qt::NoPen);
		painter->setBrush(panel);
		painter->drawRoundedRect(QRectF(option.rect).adjusted(0.5, 0.5, -0.5, -0.5), 4, 4);

		const QRect cover(option.rect.topLeft() + QPoint(CARD_PADDING, CARD_PADDING),
		                  option.decorationSize);
		index.data(Qt::DecorationRole).value<QIcon>().paint(painter, cover);
		if (index.data(Qt::UserRole).toBool()) {
			PaintProBadge(painter, cover, option.font, QStringLiteral("PS5 PRO"));
		}

		QRect line(cover.left(), cover.bottom() + 1 + CARD_PADDING, cover.width(),
		           option.fontMetrics.lineSpacing());
		auto  title_font = option.font;
		title_font.setBold(true);
		painter->setFont(title_font);
		painter->setPen(text_color);
		painter->drawText(line, Qt::AlignLeft | Qt::AlignVCenter,
		                  painter->fontMetrics().elidedText(index.data().toString(), Qt::ElideRight,
		                                                    line.width()));

		line.translate(0, line.height());
		const auto size    = index.siblingAtColumn(GAME_SIZE_COLUMN).data().toString();
		const auto details = QStringLiteral("%1  %2").arg(
		    index.siblingAtColumn(GAME_SERIAL_COLUMN).data().toString(),
		    index.siblingAtColumn(GAME_VERSION_COLUMN).data().toString());
		text_color.setAlpha(200);
		painter->setFont(option.font);
		painter->setPen(text_color);
		painter->drawText(line, Qt::AlignRight | Qt::AlignVCenter, size);
		line.setRight(line.right() - option.fontMetrics.horizontalAdvance(size) - CARD_PADDING);
		painter->drawText(line, Qt::AlignLeft | Qt::AlignVCenter,
		                  option.fontMetrics.elidedText(details, Qt::ElideRight, line.width()));
		painter->restore();
	}

private:
	static constexpr int CARD_PADDING = 8;
};

// Keep the title and its identifying metadata together in one compact cell.
class GameListNameDelegate: public QStyledItemDelegate {
public:
	using QStyledItemDelegate::QStyledItemDelegate;

	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override {
		auto size = QStyledItemDelegate::sizeHint(option, index);
		size.setHeight(qMax(size.height(), 2 * option.fontMetrics.lineSpacing() + 20));
		return size;
	}

	void paint(QPainter* painter, const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override {
		QStyleOptionViewItem opt(option);
		initStyleOption(&opt, index);
		const auto* style = opt.widget->style();
		const auto icon = style->subElementRect(QStyle::SE_ItemViewItemDecoration, &opt, opt.widget);
		auto text_rect = style->subElementRect(QStyle::SE_ItemViewItemText, &opt, opt.widget);
		const auto title = opt.text;
		opt.text.clear();
		style->drawControl(QStyle::CE_ItemViewItem, &opt, painter, opt.widget);
		painter->save();
		painter->setClipRect(option.rect);
		const int line_height = opt.fontMetrics.lineSpacing();
		text_rect.setTop(option.rect.center().y() - line_height);
		text_rect.setHeight(line_height);
		auto title_font = opt.font;
		title_font.setBold(true);
		painter->setFont(title_font);
		const auto color = opt.palette.color(opt.state.testFlag(QStyle::State_Selected)
		                                        ? QPalette::HighlightedText : QPalette::Text);
		painter->setPen(color);
		painter->drawText(text_rect, Qt::AlignLeft | Qt::AlignVCenter,
		                  painter->fontMetrics().elidedText(title, Qt::ElideRight, text_rect.width()));
		text_rect.translate(0, line_height + 2);
		auto detail_font = opt.font;
		detail_font.setBold(false);
		painter->setFont(detail_font);
		auto detail_color = color;
		detail_color.setAlpha(190);
		painter->setPen(detail_color);
		const auto details = QStringLiteral("%1  \u00b7  v%2").arg(
		    index.siblingAtColumn(GAME_SERIAL_COLUMN).data().toString(),
		    index.siblingAtColumn(GAME_VERSION_COLUMN).data().toString());
		painter->drawText(text_rect, Qt::AlignLeft | Qt::AlignVCenter,
		                  painter->fontMetrics().elidedText(details, Qt::ElideRight, text_rect.width()));
		painter->restore();
		if (!index.data(Qt::UserRole).toBool()) return;
		// The list's small icon takes a smaller badge than the grid's covers.
		auto font = opt.font;
		font.setPixelSize(icon.height() / 5);
		PaintProBadge(painter, icon, font, QStringLiteral("PRO"));
	}
};

// Shows the list's games as cards in the grid: both views share the items and the current game.
inline void SetupGameGrid(QTreeWidget* list, QListView* grid) {
	grid->setModel(list->model());
	grid->setSelectionModel(list->selectionModel());
	grid->setViewMode(QListView::IconMode);
	grid->setMovement(QListView::Static);
	grid->setResizeMode(QListView::Adjust);
	grid->setSpacing(8);
	grid->setUniformItemSizes(true);
	grid->setItemDelegate(new GameGridDelegate(grid));
	// Both the name cell and cards show metadata stored in other (possibly hidden) columns.
	QObject::connect(list->model(), &QAbstractItemModel::dataChanged, list->viewport(),
	                 qOverload<>(&QWidget::update));
	QObject::connect(list->model(), &QAbstractItemModel::dataChanged, grid->viewport(),
	                 qOverload<>(&QWidget::update));
}

#endif // GAME_VIEWS_H
