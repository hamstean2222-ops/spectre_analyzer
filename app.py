import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from proto import collect_dataset, init_model
from sbor import run_measurement_series


# Initialise model once to avoid repeated loading
init_model()


def load_annotations(root: Path) -> pd.DataFrame:
    annot_path = root / "annotations.json"
    if annot_path.exists():
        raw = pd.read_json(annot_path)
        rows = []
        for fname, ranges in raw.items():
            for r in ranges:
                rows.append({"file": fname, **r})
        return pd.DataFrame(rows)
    return pd.DataFrame()


def show_visualisations(root: Path, file: str):
    csv_path = root / file
    df = pd.read_csv(csv_path)
    freqs = df.columns[1:].astype(float).to_numpy()
    times = df.iloc[:, 0].astype(float).to_numpy()
    amps = df.iloc[:, 1:].astype(float).to_numpy()

    fig, ax = plt.subplots()
    ax.imshow(amps.T, aspect="auto", origin="lower",
              extent=[times[0], times[-1], freqs[0], freqs[-1]])
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Freq, MHz")
    st.pyplot(fig)

    prob_path = root / "probs" / f"{Path(file).stem}_probs.csv"
    if prob_path.exists():
        prob_df = pd.read_csv(prob_path)
        fig2, ax2 = plt.subplots()
        ax2.plot(prob_df["freq_MHz"], prob_df["p_avg"])
        ax2.set_xlabel("Freq, MHz")
        ax2.set_ylabel("Probability")
        st.pyplot(fig2)


def main():
    st.title("Сбор и анализ спектрограмм")
    mode = st.radio("Режим", ["Запустить сбор", "Выбрать папку"])

    if mode == "Запустить сбор":
        result_dir = st.text_input("Папка для сохранения", "measurments")
    else:
        result_dir = st.text_input("Папка с данными", str(Path("dataset_root")))

    if st.button("Обработать"):
        root = Path(result_dir)
        if mode == "Запустить сбор":
            try:
                run_measurement_series(result_dir=result_dir)
            except Exception as e:  # pragma: no cover - depends on hardware
                st.error(str(e))
        collect_dataset(root)
        st.session_state["root"] = str(root)

    if "root" in st.session_state:
        root = Path(st.session_state["root"])
        annots = load_annotations(root)
        if not annots.empty:
            st.subheader("Аннотации")
            st.dataframe(annots)
            files = sorted(annots["file"].unique())
            file = st.selectbox("Выберите файл", files)
            if file:
                show_visualisations(root, file)


if __name__ == "__main__":
    main()
