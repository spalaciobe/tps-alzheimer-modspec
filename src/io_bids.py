"""Carga del dataset OpenNeuro ds004504 en formato BIDS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd


@dataclass
class SubjectRecord:
    subject_id: str          # "sub-001"
    group: str               # "A" (AD), "F" (FTD), "C" (Control)
    age: float | None
    mmse: float | None
    set_path: Path

    @property
    def label_binary(self) -> int:
        """1 = AD, 0 = HC. Asume que F (FTD) ya fue filtrado."""
        if self.group == "A":
            return 1
        if self.group == "C":
            return 0
        raise ValueError(f"Group {self.group} no soportado en binario AD vs HC")


def load_participants(bids_root: str | Path) -> pd.DataFrame:
    """Lee participants.tsv y devuelve un DataFrame con columnas estandarizadas."""
    root = Path(bids_root)
    df = pd.read_csv(root / "participants.tsv", sep="\t")
    rename = {
        "participant_id": "subject_id",
        "Group": "group",
        "Age": "age",
        "MMSE": "mmse",
        "Gender": "gender",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def list_subjects(
    bids_root: str | Path,
    groups_keep: tuple[str, ...] = ("A", "C"),
) -> list[SubjectRecord]:
    """Lista los sujetos del dataset filtrando por grupo."""
    root = Path(bids_root)
    df = load_participants(root)
    df = df[df["group"].isin(groups_keep)]

    records: list[SubjectRecord] = []
    for _, row in df.iterrows():
        sid = row["subject_id"]
        set_path = root / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
        if not set_path.exists():
            # Algunos datasets BIDS usan task-resteyesclosed o variantes
            candidates = list((root / sid / "eeg").glob("*_eeg.set"))
            if not candidates:
                continue
            set_path = candidates[0]
        records.append(
            SubjectRecord(
                subject_id=sid,
                group=row["group"],
                age=row.get("age"),
                mmse=row.get("mmse"),
                set_path=set_path,
            )
        )
    return records


def load_raw(record: SubjectRecord, preload: bool = True) -> mne.io.BaseRaw:
    """Carga el .set como mne.Raw."""
    return mne.io.read_raw_eeglab(record.set_path, preload=preload, verbose="ERROR")
