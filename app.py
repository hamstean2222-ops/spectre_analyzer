import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import re
from pathlib import Path

from proto import collect_dataset, init_model
from sbor import run_measurement_series


# Initialise model once to avoid repeated loading
init_model()


def load_annotations(root: Path) -> pd.DataFrame:
    annot_path = root / "annotations.json"
    if annot_path.exists():
        with annot_path.open() as f:
            data = json.load(f)
        rows = []
        for fname, ranges in data.items():
            if not ranges:
                rows.append({"file": fname})
            else:
                for r in ranges:
                    rows.append({"file": fname, **r})
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["file"])


def sort_key(name: str) -> int:
    return int(re.search(r"center_(\d+)MHz", name).group(1))


def show_visualisations(root: Path, file: str, annots: pd.DataFrame, show_rects: bool):
    csv_path = root / file
    df = pd.read_csv(csv_path)
    freqs = df.columns[1:].astype(float).to_numpy()
    times = df.iloc[:, 0].astype(float).to_numpy()
    amps = df.iloc[:, 1:].astype(float).to_numpy()

    # Ensure the amplitude matrix has time along the first axis
    if amps.shape[0] != len(times):
        amps = amps.T

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(
        amps,
        origin="lower",
        aspect="auto",
        extent=[freqs.min(), freqs.max(), times.min(), times.max()],
    )
    ax.set_xlabel("Freq, MHz")
    ax.set_ylabel("Time, s")

    if show_rects:
        file_annots = annots[annots["file"] == file]
        for _, row in file_annots.dropna(subset=["time_start", "time_end", "freq_start", "freq_end"]).iterrows():
            rect = Rectangle(
                (row["freq_start"], row["time_start"]),
                row["freq_end"] - row["freq_start"],
                row["time_end"] - row["time_start"],
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

    st.pyplot(fig, use_container_width=True)

    prob_path = root / "probs" / f"{Path(file).stem}_probs.csv"
    if prob_path.exists():
        prob_df = pd.read_csv(prob_path)
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(prob_df["freq_MHz"], prob_df["p_avg"])
        ax2.set_xlabel("Freq, MHz")
        ax2.set_ylabel("Probability")
        ax2.set_xlim(freqs.min(), freqs.max())
        ax2.margins(x=0)
        st.pyplot(fig2, use_container_width=True)


def main():
    st.title("Сбор и анализ спектрограмм")
    mode = st.radio(
        "Режим",
        ["Запустить сбор", "Обработать папку", "Только вывод"],
    )

    if mode == "Запустить сбор":
        result_dir = st.text_input("Папка для сохранения", "measurments")
    else:
        result_dir = st.text_input("Папка с данными", str(Path("dataset_root")))

    root = Path(result_dir)

    if mode == "Только вывод":
        if st.button("Показать"):
            st.session_state["root"] = str(root)
            st.session_state.pop("annots", None)
            st.session_state.pop("idx", None)
    elif st.button("Обработать"):
        if mode == "Запустить сбор":
            try:
                run_measurement_series(result_dir=result_dir)
            except Exception as e:  # pragma: no cover - depends on hardware
                st.error(str(e))
        collect_dataset(root)
        st.session_state["root"] = str(root)
        st.session_state.pop("annots", None)
        st.session_state.pop("idx", None)

    if "root" in st.session_state:
        root = Path(st.session_state["root"])
        annots = load_annotations(root)
        all_files = [str(p.name) for p in root.glob("*.csv")]
        files = sorted(all_files, key=sort_key)

        table_df = pd.DataFrame({"file": files})
        if not annots.empty:
            table_df = table_df.merge(annots, on="file", how="left")

        display_df = table_df.drop(columns=["time_start", "time_end"], errors="ignore")

        st.subheader("Аннотации")
        table_col, _ = st.columns([4, 1])
        df = display_df
        table_col.data_editor(df, key="annots", disabled=True, height=400)

        if "idx" not in st.session_state:
            st.session_state.idx = 0

        sel = st.session_state.annots.get("selected_rows", [])
        if sel:
            st.session_state.idx = files.index(df.iloc[sel[0]]["file"])

        idx = st.session_state.idx
        prev_col, select_col, next_col = st.columns([1, 4, 1])
        with prev_col:
            if st.button("◀") and idx > 0:
                st.session_state.idx -= 1
        with next_col:
            if st.button("▶") and idx < len(files) - 1:
                st.session_state.idx += 1

        idx = st.session_state.idx
        with select_col:
            selected_file = st.selectbox("Выберите файл", files, index=idx)
            st.session_state.idx = files.index(selected_file)

        selected_file = files[st.session_state.idx]
        show_rects = st.checkbox("Показать сигналы", value=False)
        if selected_file:
            show_visualisations(root, selected_file, annots, show_rects)


if __name__ == "__main__":
    main()
