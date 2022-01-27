import argparse
import logging
import sys
import pathlib
from PySide2 import QtWidgets, QtCore, QtGui
from typing import Callable

import threading
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


def count(rg, r0):
    return ((r0 / rg) ** np.sign(r0 - rg)) - 1


def find_x_percent(init_data: pd.Series, air_time: float, gas_time: float, percent: float):
    """Возвращает значения точек, в которых значения сопротивлений равны 90%
    от конечных значений."""
    air = init_data.at[air_time]
    gas = init_data.at[gas_time]
    delta = air - gas
    air_data = init_data.loc[:air_time]
    gas_data = init_data.loc[air_time:]
    # air_data - air + percent * (air - gas) = air_data - percent * gas - air * ( 1 - percent )
    percent_air_time = air_data.index[np.abs(air_data - air + (1 - percent) * delta).argmin()]
    percent_gas_time = gas_data.index[np.abs(gas_data - gas - (1 - percent) * delta).argmin()]
    return percent_gas_time, percent_air_time

def all_parametrs(data, t0, tg, percent):
    # t0 - это время первой точки
    # tg - это время второй точки
    # t_gas - time of percent of first point
    rg = data.at[tg]
    r0 = data.at[t0]
    S = count(rg, r0)
    t_gas, t_air = find_x_percent(data, t0, tg, percent)
    logger.debug(f"{tg}, {t_gas}, {t0}, {t_air}")
    return S, t_gas-t0, t_air - data.index[0], rg, r0, tg, t0


class WidgetsDict():
    def __init__(self, fields):
        self._intra_dict = {label: QtWidgets.QLineEdit() for label in fields}

    def __getitem__(self, key):
        return self._intra_dict[key].text()

    def __setitem__(self, key, value):
        self._intra_dict[key].setText(value)

    def items(self):
        return self._intra_dict.items()


class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simple Laba Cut")

        self.fig = Figure(figsize=(4, 3), dpi=72)

        fields = ("produv", "gas1", "gas2", "delta", "percent", "file")
        self.widgets = WidgetsDict(fields)

        self._init_variables()
        self._init_ui()

    def _init_variables(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("Time, s")
        self.ax.set_ylabel("R, Ohm")
        self.ax.set_yscale("log")
        self.data = None
        self.gas1_lines = []
        self.gas2_lines = []
        self.highlighted_region = None
        self.gas_line = None
        self.air_line = None
        self.produvka_line = self.ax.axvline(0)

    def _init_ui(self):
        layout = QtWidgets.QHBoxLayout()
        self.table = QtWidgets.QTableWidget(self)
        self.plot_widget = FigureCanvasQTAgg(figure=self.fig)
        self.navigation_toolbar = NavigationToolbar2QT(self.plot_widget, self)
        self.open_file_button = QtWidgets.QPushButton("Open file", self)
        self.open_file_button.clicked.connect(self.open_file)
        self.cut_button = QtWidgets.QPushButton("Cut", self)
        self.cut_button.clicked.connect(self.cut)
        self.save_button = QtWidgets.QPushButton("Save", self)
        self.save_button.clicked.connect(self.save)
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.plot_widget)
        right_layout.addWidget(self.navigation_toolbar)
        form_layout = QtWidgets.QFormLayout()
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.open_file_button)
        buttons_layout.addWidget(self.cut_button)
        buttons_layout.addWidget(self.save_button)
        left_layout.addLayout(form_layout)
        left_layout.addLayout(buttons_layout)
        left_layout.addWidget(self.table)
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        for label, widget in self.widgets.items():
            form_layout.addRow(label, widget)
        self.setLayout(layout)

        self.mark_region_cid = self.plot_widget.mpl_connect(
            "button_press_event", self.mark_region)

    def open_file(self):
        filename, filters = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", (pathlib.Path.home(
        )/"projects/repos/simple_laba_cut/tests/ignore").as_posix(), "DAT file (*.dat)")
        if filename:
            self.widgets["file"] = filename
            try:
                thread = threading.Thread(target=self.read_file)
                thread.start()
            except TypeError:
                pass

    def read_file(self):
        self._init_variables()
        try:
            self.data = pd.read_csv(
                self.widgets["file"], decimal=',', skiprows=1, sep='\t', index_col="Time")
        except:
            raise TypeError
        else:
            self.draw_line()

    def draw_line(self):
        self.ax.plot(self.data.index, self.data.R1)
        self.plot_widget.draw()

    def get_minimum_in_delta(self, x):
        try:
            delta = float(self.widgets["delta"])
        except:
            delta = 0
        small_set = self.data.loc[x-delta:x+delta, "R1"]
        try:
            return small_set.idxmin()
        except:
            return x

    def cut(self):
        try:
            produvka_seconds = float(self.widgets["produv"])
            gas1_seconds = float(self.widgets["gas1"])
            gas2_seconds = float(self.widgets["gas2"])
        except ValueError:
            return
        else:
            self.produvka_line.set_xdata((produvka_seconds, produvka_seconds))
            onecyc = gas1_seconds + gas2_seconds
            max_time = max(self.data.index) - onecyc
            logger.debug("Max time")
            logger.debug((max_time - produvka_seconds)/onecyc)
            self.table.setRowCount(round((max_time - produvka_seconds) / onecyc) + 1)
            self.table.setColumnCount(7*4)
            if self.gas1_lines or self.gas2_lines:
                self.gas1_lines.remove()
                self.gas2_lines.remove()
            self.gas1_lines = self.ax.vlines([x for x in np.arange(
                produvka_seconds, max_time, onecyc) + gas1_seconds], *self.ax.get_ylim(), color="blue")
            self.gas2_lines = self.ax.vlines([self.get_minimum_in_delta(x) for x in np.arange(
                produvka_seconds, max_time, onecyc) + onecyc], *self.ax.get_ylim(), color="red")
            self.plot_widget.draw()
            self.cut_full()

    def cut_full(self):
        columns = ["R1", "R2", "R3", "R4"]
        try:
            percent = float(self.widgets["percent"])
        except ValueError:
            return
        else:
            for idx, (gas1_segment, gas2_segment) in enumerate(zip(self.gas1_lines.get_segments(), self.gas2_lines.get_segments())):
                t0 = self.data.index[self.data.index.get_loc(
                    gas1_segment[0, 0], method="nearest")]
                tg = self.data.index[self.data.index.get_loc(
                    gas2_segment[0, 0], method="nearest")]
                _, left_x, right_x, intra_x = self.find_gas2_segment_borders(tg)
                logger.debug(f"In cut full: idx:{idx} t0:{t0}, tg:{tg}, left_x:{left_x}, right_x:{right_x}, intra_x:{intra_x}")

                for idx2, column in enumerate(columns):
                    for idx3, field in enumerate(all_parametrs(self.data.loc[left_x:right_x, column], t0,
                                                               tg, percent)):
                        logger.debug(f"field: {field}, idx: {idx}")
                        item = self.table.item(idx, idx2 * 7 + idx3)
                        if not item:
                            item = QtWidgets.QTableWidgetItem()
                            self.table.setItem(idx, idx2 * 7 + idx3, item)
                        item.setText("{:.3f}".format(field))

    def save(self):
        # Error occurs NoneType have no Attribute, when accessing self.table.item(row, col).text())
        save_array = np.zeros(
            (self.table.rowCount(), self.table.columnCount()))
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                try:
                    save_array[row, col] = float(self.table.item(row, col).text())
                except AttributeError:
                    logger.debug(f"{row}, {col}")
                    save_array[row, col] = 0.0

        file = pathlib.Path(self.widgets["file"]).with_suffix(".tsv")
        pd.DataFrame(save_array).to_csv(
            file, sep='\t', index=False, header=False)

    def mark_region(self, event):
        x, y = event.xdata, event.ydata
        idx, left_x, right_x, intra_x = self.find_gas2_segment_borders(x)
        try:
            tg = self.data.index[self.data.index.get_loc(
                        right_x, method="nearest")]
            t0 = self.data.index[self.data.index.get_loc(
                        intra_x, method="nearest")]
        except KeyError:
            return
        else:
            if idx is None:
                return
            if self.highlighted_region:
                self.ax.patches.remove(self.highlighted_region)
            self.highlighted_region = self.ax.axvspan(
                left_x, right_x, color="#FF000044")

            try:
                percent = float(self.widgets["percent"])
            except ValueError:
                pass
            else:
                t_gas, t_air = find_x_percent(self.data.loc[left_x:right_x, "R1"], t0, tg, percent)
                logger.debug(f"In mark: {t0} {t_air} {tg} {t_gas} {left_x} {right_x}")
                if self.gas_line:
                    self.ax.lines.remove(self.gas_line)
                self.gas_line = self.ax.axvline(t_gas, color="red", ls='--')
                if self.air_line:
                    self.ax.lines.remove(self.air_line)
                self.air_line = self.ax.axvline(t_air, color="blue", ls='--')
            finally:
                logger.debug("Region is drawn")
                self.plot_widget.draw()

    def find_gas2_segment_borders(self, x):
        if not self.gas2_lines:
            return None, None, None, None

        prev_idx, prev_x = None, None
        iterator = iter(
                enumerate(zip(map(lambda x: x[0, 0], self.gas1_lines.get_segments()), map(lambda x: x[0, 0], self.gas2_lines.get_segments() ) )))
        while True:
            try:
                idx, (segment_x_gas, segment_x) = next(iterator)
            except StopIteration:
                return None, None, None, None
            if x <= segment_x:
                break
            else:
                prev_idx, prev_x = idx, segment_x
        if prev_idx is None:
            return -1, self.produvka_line.get_xdata()[0], segment_x, segment_x_gas
        else:
            return prev_idx, prev_x, segment_x, segment_x_gas


class NoParsingFilter(logging.Filter):
    def filter(self, record):
        if "matplotlib" in record.module or "matplotlib" in record.name:
            return False
        else:
            return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--filter_mpl", action="store_true")
    args = parser.parse_args()
    if args.filter_mpl:
        logger.addFilter(NoParsingFilter())
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    app = QtWidgets.QApplication()
    widget = MainWidget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
