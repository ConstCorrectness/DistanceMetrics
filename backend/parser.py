import io
import json
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


def parse_file(filename: str, content: bytes) -> list[dict]:
    ext = _ext(filename)
    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(content))
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(io.BytesIO(content))
    elif ext == ".json":
        data = json.loads(content)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df = df.dropna(how="all").fillna("")
    return df.to_dict(orient="records")


def row_to_text(row: dict) -> str:
    return " | ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())


def _ext(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower()
