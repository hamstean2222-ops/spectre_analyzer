# --- Получение спектрограмм с анализатора спектра R&S FSVA ---
import pyvisa
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch, torch.nn as nn
import pandas as pd
import pathlib
from scipy.ndimage import median_filter
from typing import Type, List, Union


def perform_measurement(
    rs_spectrum_analyzer,
    sweep_time: float,
    span: float,
    center_freq: float,
    result_dir: str,
    total_time: float,
    trace_name: str,
    span_points: int,
):
    """Получить спектрограмму и сохранить её в CSV."""
    try:
        # граничные частоты
        start_freq = center_freq - span / 2
        stop_freq = center_freq + span / 2

        # установка параметров прибора
        rs_spectrum_analyzer.write(f"FREQ:CENT {center_freq}")
        rs_spectrum_analyzer.write(f"FREQ:SPAN {span}")
        rs_spectrum_analyzer.write(f"SWE:TIME {sweep_time}")

        # получение актуальных параметров
        span_points_act = int(rs_spectrum_analyzer.query("SWE:POIN?"))
        start_freq_act = float(rs_spectrum_analyzer.query("FREQ:STAR?"))
        stop_freq_act = float(rs_spectrum_analyzer.query("FREQ:STOP?"))

        freqs = np.linspace(start_freq_act, stop_freq_act, span_points_act)
        timestamps: List[float] = []
        spectrum_data: List[List[float]] = []

        # цикл сбора
        start_time = time.time()
        end_time = start_time + total_time

        print(
            f"\nИзмерение: center={center_freq/1e6:.0f}MHz, span={span/1e6:.0f}MHz, sweep={sweep_time*1000:.0f}ms"
        )
        print(f"Начало сбора... Требуемое время: {total_time} сек")

        while time.time() < end_time:
            rs_spectrum_analyzer.write("INIT:IMM; *WAI")
            data = rs_spectrum_analyzer.query_ascii_values(f"TRACE? {trace_name}")

            current_time = time.time() - start_time
            timestamps.append(current_time)
            spectrum_data.append(data)

            print(
                f"Прогресс: {current_time:.1f}/{total_time} сек | Спектров: {len(spectrum_data)}",
                end="\r",
            )

        CHUNK = 1001  # сколько бинов в части
        STEP = CHUNK - 1  # сдвиг, чтобы был 1-бин-оверлап (=1000)

        if CHUNK > span_points:
            # ───── внутри perform_measurement после съёма spectra ─────
            base_name = (
                f"center_{center_freq/1e6:.0f}MHz_"
                f"span_{span/1e6:.0f}MHz_"
                f"sweep_{sweep_time*1e3:.0f}ms"
            )

            for part_idx, start in enumerate(range(0, len(freqs) - CHUNK + 1, STEP)):
                end = start + CHUNK  # ровно 1001 бин
                f_slice = slice(start, end)
                freq_part = freqs[f_slice]

                csv_filename = os.path.join(
                    result_dir, f"{base_name}_part{part_idx}.csv"
                )

                with open(csv_filename, "w", newline="") as f:
                    writer = csv.writer(f)

                    # ── заголовок в прежнем формате ──
                    writer.writerow(
                        ["Timestamp"] + [f"{f/1e6:.3f}" for f in freq_part]
                    )

                    # ── строки данных ──
                    for t, data in zip(timestamps, spectrum_data):
                        writer.writerow(
                            [f"{t:.3f}"] + [f"{v:.3f}" for v in data[f_slice]]
                        )

                print(f"saved {csv_filename}  ({len(freq_part)} bins)")
        else:
            # Формирование имени файла
            base_name = (
                f"center_{center_freq/1e6:.0f}MHz_span_{span/1e6:.0f}MHz_"
                f"sweep_{sweep_time*1000:.0f}ms"
            )

            # Сохранение в CSV
            csv_filename = os.path.join(result_dir, f"{base_name}.csv")
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp"] + [f"{f/1e6:.3f}" for f in freqs])
                for t, data in zip(timestamps, spectrum_data):
                    writer.writerow([f"{t:.3f}"] + [f"{v:.3f}" for v in data])

        print(f"\nДАнные сохранены: {csv_filename}")
        print(
            f"Собрано {len(spectrum_data)} спектров за {time.time() - start_time:.2f} секунд"
        )
        return True

    except Exception as e:  # pragma: no cover - оборудования нет в тестах
        print(f"Ошибка: {str(e)}")
        return False


def run_measurement_series(
    res_address: str = "TCPIP::192.168.1.2::INSTR",
    trace_name: str = "TRACE1",
    sweep_time: float = 0.1,
    span: float = 100e6,
    start_range: float = 100e6,
    end_range: float = 6100e6,
    total_time: float = 4,
    trace_type: str = "WRIT",
    rbw: float = 3e6,
    span_points: int = 1001,
    result_dir: str | None = None,
) -> str:
    """Провести серию измерений и вернуть путь к созданной папке."""

    result_dir = result_dir or f"measurments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(result_dir, exist_ok=True)

    rm = pyvisa.ResourceManager()
    rs_spectrum_analyzer = rm.open_resource(res_address)
    rs_spectrum_analyzer.timeout = 60000

    try:
        # предварительная установка параметров
        rs_spectrum_analyzer.write("*RST")
        rs_spectrum_analyzer.write("SYST:DISP:UPD OFF")
        rs_spectrum_analyzer.write(f"DISP:TRAC1:MODE {trace_type}")
        rs_spectrum_analyzer.write("DET RMS")
        rs_spectrum_analyzer.write("SWE:TIME:AUTO OFF")
        rs_spectrum_analyzer.write("DISP:TRAC:Y:UNIT DBM")
        rs_spectrum_analyzer.write(":DISP:TRAC:Y:RLEV -28")
        rs_spectrum_analyzer.write("INIT:CONT OFF")
        rs_spectrum_analyzer.write(f"BAND {rbw}")
        rs_spectrum_analyzer.write("BAND:AUTO OFF")
        rs_spectrum_analyzer.write("POW:ATT:AUTO OFF")
        rs_spectrum_analyzer.write("POW:ATT 0")
        rs_spectrum_analyzer.write(f"SWE:POIN {span_points}")

        # рассчет центральных частот
        step = span
        center_freqs = np.arange(start_range, end_range, step)

        print("\n" + "=" * 80)
        print(
            f"Начало серии: sweep={sweep_time*1000:.0f}ms, span={span/1e6:.0f}MHz"
        )
        print(f"Количество частот: {len(center_freqs)}")
        print(
            f"Центральные частоты: {', '.join([f'{f/1e6:.1f}MHz' for f in center_freqs])}"
        )
        print("=" * 80)

        # цикл прохода по частотной оси
        for center_freq in center_freqs:
            success = perform_measurement(
                rs_spectrum_analyzer,
                sweep_time,
                span,
                center_freq,
                result_dir,
                total_time,
                trace_name,
                span_points,
            )
            if not success:
                print("Пропуск текущей конфигурации из-за ошибки")

    except Exception as e:  # pragma: no cover - оборудования нет в тестах
        print(f"Критическая ошибка: {str(e)}")
    finally:
        rs_spectrum_analyzer.close()
        print("\nИзмерения завершены. Все данные сохранены в папке:", result_dir)

    return result_dir


if __name__ == "__main__":
    run_measurement_series()

