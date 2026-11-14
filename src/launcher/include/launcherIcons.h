#pragma once

#include <QApplication>
#include <QIcon>
#include <QIconEngine>
#include <QPainter>
#include <QPainterPath>

// Code-drawn outlines work without Qt's optional SVG image plugin.
namespace LauncherIcons {
enum class Symbol { Refresh, Settings, Controller, Edit, Trash, Trophy, Grid, Folder, Play };

class OutlineEngine final: public QIconEngine {
public:
	explicit OutlineEngine(Symbol symbol): m_symbol(symbol) {}
	QIconEngine* clone() const override { return new OutlineEngine(m_symbol); }
	void         paint(QPainter* p, const QRect& rect, QIcon::Mode mode, QIcon::State) override {
		const auto palette = QApplication::palette();
		const auto group   = mode == QIcon::Disabled ? QPalette::Disabled : QPalette::Active;
		const auto role =
		    mode == QIcon::Selected ? QPalette::HighlightedText : QPalette::ButtonText;
		p->save();
		p->setRenderHint(QPainter::Antialiasing);
		const auto side = qMin(rect.width(), rect.height());
		p->translate(rect.x() + (rect.width() - side) / 2.0,
		             rect.y() + (rect.height() - side) / 2.0);
		p->scale(side / 24.0, side / 24.0);
		p->setPen(
		    QPen(palette.color(group, role), 1.7, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
		p->setBrush(Qt::NoBrush);
		switch (m_symbol) {
			case Symbol::Refresh:
				p->drawArc(QRectF(4, 4, 16, 16), 35 * 16, 285 * 16);
				p->drawLine(20, 3, 20, 9);
				p->drawLine(14, 9, 20, 9);
				break;
			case Symbol::Settings:
				for (const auto knob: {QPoint(8, 6), QPoint(16, 12), QPoint(10, 18)}) {
					p->drawLine(3, knob.y(), knob.x() - 2, knob.y());
					p->drawLine(knob.x() + 2, knob.y(), 21, knob.y());
					p->drawEllipse(knob, 2, 2);
				}
				break;
			case Symbol::Controller: {
				QPainterPath path;
				path.moveTo(7, 7);
				path.lineTo(17, 7);
				path.cubicTo(21, 7, 21, 12, 22, 17);
				path.cubicTo(22, 21, 18, 20, 16, 16);
				path.lineTo(8, 16);
				path.cubicTo(6, 20, 2, 21, 2, 17);
				path.cubicTo(3, 12, 3, 7, 7, 7);
				p->drawPath(path);
				p->drawLine(6, 11, 10, 11);
				p->drawLine(8, 9, 8, 13);
				p->drawEllipse(QPointF(16, 10), .7, .7);
				p->drawEllipse(QPointF(19, 13), .7, .7);
				break;
			}
			case Symbol::Edit: {
				QPainterPath path;
				path.moveTo(5, 15);
				path.lineTo(16, 4);
				path.lineTo(20, 8);
				path.lineTo(9, 19);
				path.lineTo(4, 20);
				path.closeSubpath();
				p->drawPath(path);
				p->drawLine(14, 6, 18, 10);
				break;
			}
			case Symbol::Trash:
				p->drawLine(4, 6, 20, 6);
				p->drawLine(9, 3, 15, 3);
				p->drawRoundedRect(QRectF(6, 6, 12, 15), 1, 1);
				p->drawLine(10, 10, 10, 17);
				p->drawLine(14, 10, 14, 17);
				break;
			case Symbol::Trophy: {
				QPainterPath cup;
				cup.moveTo(7, 3);
				cup.lineTo(17, 3);
				cup.lineTo(16, 12);
				cup.quadTo(12, 18, 8, 12);
				cup.closeSubpath();
				p->drawPath(cup);
				p->drawArc(QRectF(2, 5, 7, 8), 90 * 16, 180 * 16);
				p->drawArc(QRectF(15, 5, 7, 8), -90 * 16, 180 * 16);
				p->drawLine(12, 15, 12, 21);
				p->drawLine(8, 21, 16, 21);
				break;
			}
			case Symbol::Grid:
				for (int y = 4; y <= 14; y += 10) {
					for (int x = 4; x <= 14; x += 10)
						p->drawRoundedRect(QRectF(x, y, 6, 6), 1, 1);
				}
				break;
			case Symbol::Folder: {
				QPainterPath path;
				path.moveTo(3, 19);
				path.lineTo(3, 5);
				path.lineTo(9, 5);
				path.lineTo(11, 8);
				path.lineTo(21, 8);
				path.lineTo(21, 19);
				path.closeSubpath();
				p->drawPath(path);
				break;
			}
			case Symbol::Play: {
				QPainterPath path;
				path.moveTo(7, 4);
				path.lineTo(20, 12);
				path.lineTo(7, 20);
				path.closeSubpath();
				p->drawPath(path);
				break;
			}
		}
		p->restore();
	}
	QPixmap pixmap(const QSize& size, QIcon::Mode mode, QIcon::State state) override {
		QPixmap result(size);
		result.fill(Qt::transparent);
		QPainter painter(&result);
		paint(&painter, QRect(QPoint(), size), mode, state);
		return result;
	}

private:
	Symbol m_symbol;
};

inline QIcon Make(Symbol symbol) {
	return QIcon(new OutlineEngine(symbol));
}
} // namespace LauncherIcons
