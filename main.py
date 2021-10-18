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

def find_90_percent(data: np.ndarray, reper_air: float, reper_gas: float):
    """Возвращает значения точек, в которых значения сопротивлений равны 90%
    от конечных значений."""
    # Сомнительная идея, сделать так, чтобы мы могли учитывать оба варианта
    # расположения газов
    # Пока делать не будем
    # if reper_gas > reper_air:
    # reper_gas, reper_air = reper_air, reper_gas

    delta = reper_air - reper_gas
    procent = 0.9 * delta
    data_recovery = np.abs(data - (reper_gas + procent))
    data_sens = np.abs(data - (reper_air - procent))
    result1 = np.min(np.argsort(data_sens)[:20])
    result2 = np.max(np.argsort(data_recovery)[:4])
    return result1, result2

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
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel("Time, s")
        self.ax.set_ylabel("R, Ohm")
        self.ax.set_yscale("log")


        fields = ("produv", "gas1", "gas2", "file")
        self.widgets = WidgetsDict(fields)


        self.data = None
        self.gas1_lines = []
        self.gas2_lines = []
        self.produvka_line = self.ax.axvline(0)

        self._init_ui()

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
        try:
            self.data = pd.read_csv(self.widgets["file"], decimal=',', skiprows=1, sep='\t', index_col="Time")
        except:
            raise TypeError
        else:
            self.draw_line()

    def draw_line(self):
        self.ax.plot(self.data.index, self.data.R1)
        self.plot_widget.draw()

    def cut(self):
        try:
            produvka_seconds = int(self.widgets["produv"])
            gas1_seconds = int(self.widgets["gas1"])
            gas2_seconds = int(self.widgets["gas2"])
        except ValueError:
            return 
        else:
            self.produvka_line.set_xdata((produvka_seconds, produvka_seconds))
            onecyc = gas1_seconds + gas2_seconds
            max_time = max(self.data.index) - onecyc
            self.table.setRowCount(int((max_time - produvka_seconds)/onecyc))
            self.table.setColumnCount(4)
            if self.gas1_lines:
                segments = []
                for x, line in zip(np.arange(produvka_seconds, max_time, onecyc) + gas1_seconds, self.gas1_lines.get_segments()):
                    (x0, y0), (x1, y1) = line
                    segments.append(((x, y0), (x, y1)))
                self.gas1_lines.set_segments(segments)
            else:
                self.gas1_lines = self.ax.vlines([x for x in np.arange(produvka_seconds, max_time, onecyc) + gas1_seconds], *self.ax.get_ylim(), color="blue")

            if self.gas2_lines:
                segments = []
                for x, line in zip(np.arange(produvka_seconds, max_time, onecyc) + onecyc, self.gas2_lines.get_segments()):
                    (x0, y0), (x1, y1) = line
                    segments.append(((x, y0), (x, y1)))
                self.gas2_lines.set_segments(segments)
            else:
                self.gas2_lines = self.ax.vlines([x for x in np.arange(produvka_seconds, max_time, onecyc) + onecyc], *self.ax.get_ylim(), color="red")
            self.plot_widget.draw()
            self.cut_full()




    def cut_full(self):
        for idx, (gas1_segment, gas2_segment) in enumerate(zip(self.gas1_lines.get_segments(), self.gas2_lines.get_segments())):
            r0 = self.data.iloc[self.data.index.get_loc(gas1_segment[0, 0], method="nearest")].loc[["R1", "R2", "R3", "R4"]].to_numpy()
            rg = self.data.iloc[self.data.index.get_loc(gas2_segment[0, 0], method="nearest")].loc[["R1", "R2", "R3", "R4"]].to_numpy()
            for idx2, S in enumerate(count(rg, r0)):
                logger.debug(S)
                item = self.table.item(idx, idx2)
                if not item:
                    item = QtWidgets.QTableWidgetItem()
                    self.table.setItem(idx, idx2, item)
                item.setText("{:.3f}".format(S))

    def save(self):
        save_array = np.zeros((self.table.rowCount(), self.table.columnCount()))
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                save_array[row, col] = float(self.table.item(row, col).text())

        file = pathlib.Path(self.widgets["file"]).with_suffix(".tsv")
        pd.DataFrame(save_array, columns=["S{}".format(x) for x in range(1, 5)]).to_csv(file, sep='\t', index=False)


    def cut_old(self):

        try:
            _ = [value.get() for value in self.variables.values()]
            self.message("Все ок")
        except ValueError as e:
            self.message("Не все поля заполнены, милорд")
            print(e)
        else:
            data = pd.read_csv(
                self.variables["file"].get(), decimal=',', skiprows=1, sep='\t')
            freq = int(self.variables["freq"].get())
            onecyc = int(self.variables["onecyc"].get()) * freq
            produv = int(self.variables["prod"].get()) * freq
            logger.debug(f"{freq}, {produv}, {onecyc}")

            if self.produvka_index is None:
                self.get_indexes(data.loc[:produv + onecyc + 200, "R1"], 1)
                return None

            data = data.iloc[self.produvka_index:]
            onecyc = self.gas_index - self.produvka_index
            logger.debug("Onecyc value: {}".format(onecyc))
            data.index = np.arange(data.shape[0])
            counter = 0
            fd = self.output_filename.open("w")
            for sensor_column_name in ("R{}".format(i) for i in range(1, 13)):
                try:
                    data[sensor_column_name]
                except KeyError:
                    break
                else:
                    if sensor_column_name == "R1":
                        fd.write("Rg,R0,S,T_res,T_rec")
                    else:
                        fd.write(",Rg,R0,S,T_res,T_rec")
            while True:
                temp_data = data.iloc[counter * onecyc: (counter + 1) * onecyc]
                if temp_data.shape[0] != 0:
                    print(temp_data.shape, end=" ")
                    for sensor_column_name in ("R{}".format(i) for i in range(1, 13)):
                        try:
                            temp_series = temp_data[sensor_column_name]
                        except KeyError:
                            continue
                        else:
                            if (self.idx1 is None) or (self.idx2 is None):
                                self.get_indexes(temp_series, 2)
                                fd.close()
                                return
                            try:
                                rg = temp_series.iloc[self.idx1]
                                r0 = temp_series.iloc[self.idx2]
                            except IndexError:
                                continue
                            else:
                                s = count(rg, r0)
                                time_idx_1, time_idx_2 = self.find_90_percent(
                                    temp_series.values, r0, rg)
                                t_sens = time_idx_1 - 0
                                t_recovery = time_idx_2 - self.idx1
                                format_line = "{:3.4f},{:3.4f},{:3.4f},{:3.4f},{:3.4f}"
                                if sensor_column_name == "R1":
                                    fd.write(
                                        ("\n" + format_line).format(rg, r0, s, t_sens, t_recovery))
                                else:
                                    fd.write(
                                        ("," + format_line).format(rg, r0, s, t_sens, t_recovery))
                    counter += 1
                else:
                    fd.write("\n")
                    fd.close()
                    break




class MatplotlibCutWidget():
    def __init__(self, data: pd.Series, func: Callable, message_func: Callable):
        super().__init__()
        self.func = func
        self.message_func = message_func
        self.f = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.f.add_subplot(111)
        self.ax.plot(data.index, data.values)
        self.ax.set_yscale("log")

        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.X)
        self.canvas.mpl_connect("button_press_event", self.onclick)

        proceed_button = tk.Button(self, text="Резать резать резать дальше")
        proceed_button["command"] = self.proceed
        proceed_button.pack(fill=tk.X)

        self.idx1 = None
        self.idx2 = None
        self.red_line = None
        self.green_line = None
        self.red_line_text = None
        self.green_line_text = None

    def onclick(self, event):
        if event.button == MouseButton.LEFT:
            self.idx1 = int(event.xdata)
            if self.red_line is not None:
                self.red_line.set_xdata(event.xdata)
                self.red_line_text.set_text("{}".format(event.xdata))
                self.red_line_text.set_x(event.xdata)
            else:
                self.red_line = self.ax.axvline(event.xdata, color='r', lw=0.5)
                self.red_line_text = self.ax.text(event.xdata,
                                                  self.ax.get_ylim()[0], "{}".format(event.xdata))
        elif event.button == MouseButton.RIGHT:
            self.idx2 = int(event.xdata)
            if self.green_line is not None:
                self.green_line.set_xdata(event.xdata)
                self.green_line_text.set_text("{}".format(event.xdata))
                self.green_line_text.set_x(event.xdata)
            else:
                self.green_line = self.ax.axvline(
                    event.xdata, color='g', lw=0.5)
                self.green_line_text = self.ax.text(event.xdata,
                                                    self.ax.get_ylim()[0], "{}".format(event.xdata))
        else:
            pass
        self.canvas.draw()

    def get_indexes(self):
        return map(int, (self.idx1, self.idx2))

    def proceed(self):
        idx1, idx2 = self.get_indexes()
        logger.debug(f"{idx1} {idx2}")
        self.message_func("Indexes = {:d} {:d}".format(idx1, idx2))
        self.func(indexes=(idx1, idx2))
        self.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    app = QtWidgets.QApplication()
    widget = MainWidget()
    widget.show()
    sys.exit(app.exec_())
