"""LongBench loaders for local workload generation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from zipfile import ZipFile

from sembench.schema import SourceRecord
from sembench.transforms import fixture_records, stable_id

DEFAULT_LONGBENCH_V1_DATASETS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)


def load_source_records(
    profile: str,
    datasets: list[str] | None = None,
    max_items_per_dataset: int | None = None,
) -> list[SourceRecord]:
    """Load normalized source records for a profile."""
    if profile == "fixture":
        return fixture_records()
    if profile == "longbench-v1":
        return load_longbench_v1(
            datasets=datasets or list(DEFAULT_LONGBENCH_V1_DATASETS),
            max_items_per_dataset=max_items_per_dataset,
        )
    if profile == "longbench-v2":
        return load_longbench_v2(max_items=max_items_per_dataset)
    raise ValueError(f"unknown profile: {profile}")


def load_longbench_v1(
    datasets: list[str],
    max_items_per_dataset: int | None = None,
) -> list[SourceRecord]:
    """Load LongBench v1 tasks from Hugging Face datasets."""
    load_dataset = _datasets_loader()
    records: list[SourceRecord] = []
    for dataset_name in datasets:
        ds = _load_longbench_v1_rows(
            load_dataset,
            dataset_name=dataset_name,
        )
        for idx, row in enumerate(_take(ds, max_items_per_dataset)):
            context = str(row.get("context", ""))
            input_text = str(row.get("input", ""))
            answers = [str(a) for a in row.get("answers", [])]
            source_id = str(row.get("_id") or stable_id(dataset_name, str(idx), context[:256]))
            records.append(
                SourceRecord(
                    source_id=source_id,
                    dataset=dataset_name,
                    context=context,
                    input=input_text,
                    answers=answers,
                    metadata={
                        "profile": "longbench-v1",
                        "length": row.get("length"),
                        "source_index": idx,
                    },
                )
            )
    return records


def load_longbench_v2(max_items: int | None = None) -> list[SourceRecord]:
    """Load LongBench v2 rows from Hugging Face datasets."""
    load_dataset = _datasets_loader()
    ds = _load_longbench_v2_rows(load_dataset)
    records: list[SourceRecord] = []
    for idx, row in enumerate(_take(ds, max_items)):
        choices = [
            f"A. {row.get('choice_A', '')}",
            f"B. {row.get('choice_B', '')}",
            f"C. {row.get('choice_C', '')}",
            f"D. {row.get('choice_D', '')}",
        ]
        question = str(row.get("question", ""))
        input_text = question + "\n" + "\n".join(choices)
        source_id = str(row.get("_id") or stable_id("longbench-v2", str(idx), question))
        records.append(
            SourceRecord(
                source_id=source_id,
                dataset="longbench-v2",
                context=str(row.get("context", "")),
                input=input_text,
                answers=[str(row.get("answer", ""))],
                metadata={
                    "profile": "longbench-v2",
                    "domain": row.get("domain"),
                    "sub_domain": row.get("sub_domain"),
                    "difficulty": row.get("difficulty"),
                    "length": row.get("length"),
                    "source_index": idx,
                },
            )
        )
    return records


def _datasets_loader():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "LongBench loading requires the datasets package. "
            "Install with: python -m pip install -e '.[longbench]'"
        ) from e
    return load_dataset


def _load_with_fallbacks(
    load_dataset,
    *,
    candidates: tuple[str, ...],
    name: str | None,
    split: str,
):
    errors: list[str] = []
    for dataset_path in candidates:
        try:
            if name is None:
                return load_dataset(dataset_path, split=split)
            return load_dataset(dataset_path, name, split=split)
        except Exception as e:
            errors.append(f"{dataset_path}: {e}")
    joined = "\n".join(errors)
    raise RuntimeError(f"failed to load dataset candidates:\n{joined}")


def _load_longbench_v1_rows(load_dataset, *, dataset_name: str):
    try:
        return _load_v1_zip_rows(dataset_name)
    except Exception as direct_error:
        try:
            return _load_with_fallbacks(
                load_dataset,
                candidates=("THUDM/LongBench", "zai-org/LongBench"),
                name=dataset_name,
                split="test",
            )
        except Exception as dataset_error:
            raise RuntimeError(
                "failed to load LongBench v1 via Hub data.zip and datasets:\n"
                f"data.zip: {direct_error}\n"
                f"datasets: {dataset_error}"
            ) from dataset_error


def _load_longbench_v2_rows(load_dataset):
    try:
        return _load_v2_json_rows()
    except Exception as direct_error:
        try:
            return _load_with_fallbacks(
                load_dataset,
                candidates=("THUDM/LongBench-v2", "zai-org/LongBench-v2"),
                name=None,
                split="train",
            )
        except Exception as dataset_error:
            raise RuntimeError(
                "failed to load LongBench v2 via Hub data.json and datasets:\n"
                f"data.json: {direct_error}\n"
                f"datasets: {dataset_error}"
            ) from dataset_error


def _load_v1_zip_rows(dataset_name: str) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    errors: list[str] = []
    member = f"data/{dataset_name}.jsonl"
    for repo in ("THUDM/LongBench", "zai-org/LongBench"):
        try:
            path = hf_hub_download(repo, "data.zip", repo_type="dataset")
            with ZipFile(path) as zf:
                with zf.open(member) as f:
                    return [json.loads(line.decode("utf-8")) for line in f if line.strip()]
        except Exception as exc:
            errors.append(f"{repo}/{member}: {exc}")
    raise RuntimeError("; ".join(errors))


def _load_v2_json_rows() -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    errors: list[str] = []
    for repo in ("THUDM/LongBench-v2", "zai-org/LongBench-v2"):
        try:
            path = hf_hub_download(repo, "data.json", repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                return [dict(row) for row in raw]
            if isinstance(raw, dict):
                data = raw.get("data") or raw.get("train") or raw.get("rows")
                if isinstance(data, list):
                    return [dict(row) for row in data]
            raise RuntimeError(f"unsupported data.json shape: {type(raw).__name__}")
        except Exception as exc:
            errors.append(f"{repo}/data.json: {exc}")
    raise RuntimeError("; ".join(errors))


def _take(rows: Iterable[dict[str, Any]], limit: int | None):
    for idx, row in enumerate(rows):
        if limit is not None and idx >= limit:
            break
        yield row
