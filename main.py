import argparse
import logging
import sys
import pathlib
from PySide2 import QtWidgets

import threading
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def count(rg, r0):
    return ((r0 / rg)**np.sign(r0 - rg)) - 1


def define_type(path: pathlib.Path):
    if path.suffix == ".dat":
        return 1
    elif path.suffix == ".csv":
        return 2
    else:
        return 0


def columns_definer(
        data: pd.DataFrame):  # This function defined for .csv files
    if data.columns.size == 11:
        data.columns = [
            'Time', 'Temp', 'T1', 'T2', 'T3', 'T4', 'R1', 'R2', 'R3', 'R4',
            'Rrep'
        ]
    elif data.columns.size == 12:
        data.columns = [
            'Time', 'Temp', 'step', 'T1', 'T2', 'T3', 'T4', 'R1', 'R2', 'R3',
            'R4', 'Rrep'
        ]
    else:
        data.columns = [
            'Time', 'Temp', 'etap', 'step', 'T1', 'T2', 'T3', 'T4', 'R1', 'R2',
            'R3', 'R4', 'Rrep'
        ]


def load_file(path):
    type_ = define_type(path)
    if type_ == 1:  # DAT type
        data = pd.read_csv(path,
                           decimal=',',
                           skiprows=1,
                           sep='\t',
                           index_col="Time")
        if "R12" in data.columns:
            sensors_number = 12
        else:
            sensors_number = 4
        return data, sensors_number
    elif type_ == 2:  # CSV type
        data = pd.read_csv(path, header=None)
        columns_definer(data)
        data.set_index("Time", inplace=True)
        data = data.iloc[:-1]
        logger.debug(f"Read csv file {type(data)} {data}")
        return data, 4
    else:
        logger.write("Can't load file, cause file type is wrong")
        raise Exception("Can't load file, wrong type")


def find_x_percent(init_data: pd.Series, air_time: float, gas_time: float,
                   percent: float):
    """Возвращает значения точек, в которых значения сопротивлений равны 90%
    от конечных значений."""
    air = init_data.at[air_time]
    gas = init_data.at[gas_time]
    delta = air - gas
    air_data = init_data.loc[:air_time]
    gas_data = init_data.loc[air_time:]
    # air_data - air + percent * (air - gas) = air_data - percent * gas - air * ( 1 - percent )
    percent_air_time = air_data.index[np.abs(air_data - air +
                                             (1 - percent) * delta).argmin()]
    percent_gas_time = gas_data.index[np.abs(gas_data - gas -
                                             (1 - percent) * delta).argmin()]
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
    return S, t_gas - t0, t_air - data.index[0], rg, r0, tg, t0


class WidgetsDict():

    def __init__(self, fields):
        self._intra_dict = {
            label: QtWidgets.QLineEdit(text=str(fields[label][1]))
            for label in fields.keys()
        }
        self._types_dict = fields
        for lineedit, field_values in zip(self._intra_dict.values(),
                                          fields.values()):
            lineedit.setToolTip(field_values[2])

    def __getitem__(self, key):
        text = self._intra_dict[key].text()
        type_, default_value, _ = self._types_dict[key]
        try:
            return type_(text)
        except ValueError:
            return default_value

    def __setitem__(self, key, value):
        self._intra_dict[key].setText(str(value))

    def items(self):
        return self._intra_dict.items()

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __setattr(self, name, value):
        self.__setitem__(name, value)


class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simple Laba Cut")

        self.fig = Figure(figsize=(4, 3), dpi=72)

        fields = {
            "produv": [float, 0, "Время продувки, с"],
            "gas1": [float, 0, "Время выдержки первого газа, с"],
            "gas2": [float, 0, "Время выдержки второго газа, с"],
            "delta": [
                int, 5,
                "Количество точек, определяющих, в каком радиусе искать минимум/максимум для интервалов выдержки"
            ],
            "percent": [
                float, 0.9,
                "Доля, для которой рассчитать времена релаксации/восстановления"
            ],
            "file": [str, "", "Путь до файла"],
            "sensor":
            [str, "R1", "Сигнал сенсора, который отрисовывать на графике"]
        }
        self.widgets = WidgetsDict(fields)

        self._init_variables()
        self._init_ui()

    def _init_variables(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.callbacks.connect("ylim_changed", self.axis_limit_changed)
        self.ax.set_xlabel("Time, s")
        self.ax.set_ylabel("R, Ohm")
        self.ax.set_yscale("log")
        self.data = None
        self.main_line = None
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
        self.cut_button = QtWidgets.QPushButton("Precut", self)
        self.cut_button.setToolTip(
            "Отметить позиции, на которых будут взяты значения для расчета")
        self.cut_button.clicked.connect(self.cut)
        self.cut_full_button = QtWidgets.QPushButton("Cut to values", self)
        self.cut_full_button.clicked.connect(self.cut_full)
        self.cut_full_button.setToolTip("Рассчитать значения")
        self.save_button = QtWidgets.QPushButton("Save", self)
        self.save_button.clicked.connect(self.save)
        self.save_button.setToolTip("Сохранить полученные данные")
        self.redraw_button = QtWidgets.QPushButton("Redraw", self)
        self.redraw_button.clicked.connect(self.draw_line)
        self.redraw_button.setToolTip(
            "Перерисовать данные, которые были загружены, если изменили сенсор, по которому рисовать"
        )
        self.max_min_inverse_checkbox = QtWidgets.QCheckBox(text="Inverse")
        self.max_min_inverse_checkbox.setToolTip(
            "Если отмечено, то для первого газа идет поиск максимума, иначе минимума"
        )

        self.progress_bar = QtWidgets.QProgressBar()

        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.plot_widget)
        right_layout.addWidget(self.navigation_toolbar)

        form_layout = QtWidgets.QFormLayout()
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.open_file_button)
        buttons_layout.addWidget(self.cut_button)
        buttons_layout.addWidget(self.cut_full_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.redraw_button)
        buttons_layout.addWidget(self.max_min_inverse_checkbox)
        left_layout.addLayout(form_layout)
        left_layout.addLayout(buttons_layout)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.table)
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        for label, widget in self.widgets.items():
            form_layout.addRow(label, widget)
        self.setLayout(layout)

        self.mark_region_cid = self.plot_widget.mpl_connect(
            "button_press_event", self.mark_region)

    def disable_important_buttons(self):
        self.open_file_button.setDisabled(True)
        self.cut_button.setDisabled(True)
        self.cut_full_button.setDisabled(True)
        self.save_button.setDisabled(True)
        self.redraw_button.setDisabled(True)
        self.max_min_inverse_checkbox.setDisabled(True)

    def enable_important_buttons(self):
        self.open_file_button.setDisabled(False)
        self.cut_button.setDisabled(False)
        self.cut_full_button.setDisabled(False)
        self.save_button.setDisabled(False)
        self.redraw_button.setDisabled(False)
        self.max_min_inverse_checkbox.setDisabled(False)

    def open_file(self):
        filename, filters = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open file",
            (pathlib.Path.home() /
             "projects/repos/simple_laba_cut/tests/ignore").as_posix(),
            "DAT file (*.dat);;CSV files (*.csv)")
        if filename:
            self.widgets["file"] = filename
            try:
                thread = threading.Thread(target=self.read_file)
                thread.start()
            except TypeError:
                pass

    def read_file(self):
        self._init_variables()
        path = pathlib.Path(self.widgets["file"])
        try:
            self.data, self.sensors_number = load_file(path)
            logger.debug(self.data.head(10))
        except:
            raise
        else:
            self.draw_line()

    def draw_line(self):
        if self.main_line is None:
            self.main_line, = self.ax.plot(self.data.index,
                                           self.data[self.widgets.sensor])
        else:
            self.main_line.set_data(self.data.index,
                                    self.data[self.widgets.sensor])
        self.plot_widget.draw()

    def axis_limit_changed(self, ax):
        min_, max_ = self.ax.get_ylim()
        try:
            self._set_segments_of_line(self.gas1_lines, min_, max_)
            self._set_segments_of_line(self.gas2_lines, min_, max_)
        except AttributeError:
            pass
        try:
            self.plot_widget.draw()
        except AttributeError:
            pass

    @staticmethod
    def _set_segments_of_line(gas_lines, min_, max_):
        if gas_lines:
            segments = gas_lines.get_segments()
            gas_lines.set_segments([[[segment[0][0], min_],
                                     [segment[1][0], max_]]
                                    for segment in segments])

    def get_minimum_in_delta(self, x):
        small_set = self.data.loc[x - self.widgets.delta:x +
                                  self.widgets.delta, self.widgets.sensor]
        try:
            return small_set.idxmin()
        except:
            return x

    def get_maximum_in_delta(self, x):
        small_set = self.data.loc[x - self.widgets.delta:x +
                                  self.widgets.delta, self.widgets.sensor]
        try:
            return small_set.idxmax()
        except:
            return x

    def get_func_in(self, x, what_line):
        inverse_is_checked: bool = self.max_min_inverse_checkbox.isChecked()

        if what_line == 0:
            return self.get_maximum_in_delta(
                x) if inverse_is_checked else self.get_minimum_in_delta(x)
        else:
            return self.get_minimum_in_delta(
                x) if inverse_is_checked else self.get_maximum_in_delta(x)

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
            logger.debug((max_time - produvka_seconds) / onecyc)
            self.table.setRowCount(
                round((max_time - produvka_seconds) / onecyc) + 1)
            self.table.setColumnCount(7 * self.sensors_number)
            if self.gas1_lines or self.gas2_lines:
                self.gas1_lines.remove()
                self.gas2_lines.remove()
            self.gas1_lines = self.ax.vlines([
                self.get_func_in(x, 0)
                for x in np.arange(produvka_seconds, max_time, onecyc) +
                gas1_seconds
            ],
                                             *self.ax.get_ylim(),
                                             color="blue")
            self.gas2_lines = self.ax.vlines([
                self.get_func_in(x, 1)
                for x in np.arange(produvka_seconds, max_time, onecyc) + onecyc
            ],
                                             *self.ax.get_ylim(),
                                             color="red")
            self.plot_widget.draw()

    def cut_full(self):
        self.disable_important_buttons()
        if self.sensors_number == 4:
            columns = ["R1", "R2", "R3", "R4"]
        else:
            columns = ["R{}".format(i) for i in range(1, 13)]
        try:
            percent = float(self.widgets["percent"])
        except ValueError:
            return
        else:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(self.data.index[-1])
            for idx, (gas1_segment, gas2_segment) in enumerate(
                    zip(self.gas1_lines.get_segments(),
                        self.gas2_lines.get_segments())):
                t0 = self.data.index[self.data.index.get_loc(gas1_segment[0,
                                                                          0],
                                                             method="nearest")]
                tg = self.data.index[self.data.index.get_loc(gas2_segment[0,
                                                                          0],
                                                             method="nearest")]
                _, left_x, right_x, intra_x = self.find_gas2_segment_borders(
                    tg)
                logger.debug(
                    f"In cut full: idx:{idx} t0:{t0}, tg:{tg}, left_x:{left_x}, right_x:{right_x}, intra_x:{intra_x}"
                )
                self.progress_bar.setValue(intra_x)

                for idx2, column in enumerate(columns):
                    for idx3, field in enumerate(
                            all_parametrs(
                                self.data.loc[left_x:right_x, column], t0, tg,
                                percent)):
                        logger.debug(f"field: {field}, idx: {idx}")
                        item = self.table.item(idx, idx2 * 7 + idx3)
                        if not item:
                            item = QtWidgets.QTableWidgetItem()
                            self.table.setItem(idx, idx2 * 7 + idx3, item)
                        item.setText("{:.3f}".format(field))
            self.progress_bar.setValue(self.data.index[-1])
        finally:
            self.enable_important_buttons()

    def save(self):
        # Error occurs NoneType have no Attribute, when accessing self.table.item(row, col).text())
        save_array = np.zeros(
            (self.table.rowCount(), self.table.columnCount()))
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                try:
                    save_array[row,
                               col] = float(self.table.item(row, col).text())
                except AttributeError:
                    logger.debug(f"{row}, {col}")
                    save_array[row, col] = 0.0

        file = pathlib.Path(self.widgets["file"]).with_suffix(".tsv")
        pd.DataFrame(save_array).to_csv(file,
                                        sep='\t',
                                        index=False,
                                        header=False)

    def mark_region(self, event):
        if not self.navigation_toolbar.mode:
            x, _ = event.xdata, event.ydata
            idx, left_x, right_x, intra_x = self.find_gas2_segment_borders(x)
            try:
                tg = self.data.index[self.data.index.get_loc(right_x,
                                                             method="nearest")]
                t0 = self.data.index[self.data.index.get_loc(intra_x,
                                                             method="nearest")]
            except KeyError:
                return
            else:
                if idx is None:
                    return
                if self.highlighted_region:
                    self.ax.patches.remove(self.highlighted_region)
                self.highlighted_region = self.ax.axvspan(left_x,
                                                          right_x,
                                                          color="#FF000044")

                try:
                    percent = float(self.widgets["percent"])
                except ValueError:
                    pass
                else:
                    try:
                        t_gas, t_air = find_x_percent(
                            self.data.loc[left_x:right_x, self.widgets.sensor],
                            t0, tg, percent)
                    except KeyError:
                        msgBox = QtWidgets.QMessageBox()
                        msgBox.setText(
                            "Значение вышло за границы сегмента.\nПопробуйте убрать/поставить галочку в Inverse.\nЕсли не помогло, уменьшите значение delta."
                        )
                        msgBox.exec()
                    else:
                        logger.debug(
                            f"In mark: {t0} {t_air} {tg} {t_gas} {left_x} {right_x}"
                        )
                        if self.gas_line:
                            self.ax.lines.remove(self.gas_line)
                        self.gas_line = self.ax.axvline(t_gas,
                                                        color="red",
                                                        ls='--')
                        if self.air_line:
                            self.ax.lines.remove(self.air_line)
                        self.air_line = self.ax.axvline(t_air,
                                                        color="blue",
                                                        ls='--')
                finally:
                    logger.debug("Region is drawn")
                    self.plot_widget.draw()

    def find_gas2_segment_borders(self, x):
        if not self.gas2_lines:
            return None, None, None, None

        prev_idx, prev_x = None, None
        iterator = iter(
            enumerate(
                zip(map(lambda x: x[0, 0], self.gas1_lines.get_segments()),
                    map(lambda x: x[0, 0], self.gas2_lines.get_segments()))))
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
            return -1, self.produvka_line.get_xdata(
            )[0], segment_x, segment_x_gas
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
