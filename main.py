import pathlib
import tkinter as tk
from tkinter.filedialog import askopenfilename
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

import logging

logger = logging.getLogger(__name__)

import argparse 


class Frame(tk.Frame):
    def __init__(self, master):
        super(Frame, self).__init__(master)
        names = {"prod": "Продувка, с",
                 "onecyc": "Время двух этапов",
                 "freq": "Частота, Гц",
                 "file": "Файл"}

        self.variables = {name: tk.StringVar(self) for name in names.keys()}
        self.labels = {name: tk.Label(self, text=name_text) for name, name_text in names.items()}
        self.entrys = {name: tk.Entry(self, textvariable=self.variables[name]) for name in names.keys()}
        self.output_filename = None

        for idx, label in enumerate(self.labels.values()):
            label.grid(row=idx, column=0)
        for idx, entry in enumerate(self.entrys.values()):
            entry.grid(row=idx, column=1, columnspan=2 if idx != (len(self.entrys) - 1) else 1, sticky=tk.W + tk.E)

        idx = len(self.labels)
        open_button = tk.Button(self, text="Открыть...")
        open_button.grid(row=idx - 1, column=2)
        open_button["command"] = self.set_file

        cut_button = tk.Button(self, text="Резать, резать, резать")
        cut_button.grid(row=idx, column=0, columnspan=3, sticky=tk.E + tk.W)
        cut_button["command"] = self.cut_clear

        self.status_variable = tk.StringVar()
        status_bar = tk.Label(self, textvariable=self.status_variable, relief=tk.SUNKEN)
        status_bar.grid(row=idx + 1, column=0, columnspan=3, sticky=tk.W + tk.E)

        self.idx1 = None
        self.idx2 = None
        self.produvka_index = None
        self.gas_index = None

    def set_file(self):
        filename = askopenfilename(initialdir=pathlib.Path.home()/"projects/repos/simple_laba_cut/tests/ignore")
        try:
            self.output_filename = pathlib.Path(filename).with_suffix(".csv")
        except TypeError:
            self.variables["file"].set("")
        else:
            self.variables["file"].set(filename)

    def set_first_indexes(self, indexes):
        self.produvka_index, self.gas_index = indexes
        self.cut()

    def set_second_indexes(self, indexes):
        self.idx1, self.idx2 = indexes
        self.cut()

    def cut_clear(self):
        self.idx1, self.idx2, self.produvka_index, self.gas_index = None, None, None, None
        self.cut()

    def cut(self):
        def count(rg, r0):
            return ((r0 / rg) ** np.sign(r0 - rg)) - 1

        try:
            _ = [value.get() for value in self.variables.values()]
            self.message("Все ок")
        except ValueError as e:
            self.message("Не все поля заполнены, милорд")
            print(e)
        else:
            data = pd.read_csv(self.variables["file"].get(), decimal=',', skiprows=1, sep='\t')
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
                                time_idx_1, time_idx_2 = self.find_90_percent(temp_series.values, r0, rg)
                                t_sens = time_idx_1 - 0
                                t_recovery = time_idx_2 - self.idx1
                                format_line = "{:3.4f},{:3.4f},{:3.4f},{:3.4f},{:3.4f}"
                                if sensor_column_name == "R1":
                                    fd.write(("\n" + format_line).format(rg, r0, s, t_sens, t_recovery))
                                else:
                                    fd.write(("," + format_line).format(rg, r0, s, t_sens, t_recovery))
                    counter += 1
                else:
                    fd.write("\n")
                    fd.close()
                    break

    def message(self, text: str):
        self.status_variable.set(text)

    def get_indexes(self, data, type_):
        if type_ == 1:
            MatplotlibCutWidget(data, self.set_first_indexes, self.message)
        else:
            MatplotlibCutWidget(data, self.set_second_indexes, self.message)

    @staticmethod
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


class MatplotlibCutWidget(tk.Toplevel):
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
                self.green_line = self.ax.axvline(event.xdata, color='g', lw=0.5)
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
    logging.basicConfig(level = logging.DEBUG if args.debug else logging.INFO)
    root = tk.Tk()
    frame = Frame(root)
    frame.pack()
    root.mainloop()
