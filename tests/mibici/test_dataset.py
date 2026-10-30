import hashlib
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from mibici import dataset
from mibici.config import DATA_PIPELINE_VERSION

DATA_METADATA = {
    "monthly": ["viajes_2026_07.csv", "viajes_2026_08.csv"],
    "stations": "estaciones.csv",
    "data_range": {"from": "2026-07", "to": "2026-08"},
}

USER_COLUMNS = ["Usuario_Id", "Genero", "Año_de_nacimiento"]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def make_raw_trips(trip_ids, origins, destinations, month="07") -> pd.DataFrame:
    n = len(trip_ids)
    start = pd.Series(pd.date_range(f"2026-{month}-01 08:00", periods=n, freq="30min"))
    return pd.DataFrame(
        {
            "Viaje_Id": trip_ids,
            "Usuario_Id": range(100, 100 + n),
            "Genero": ["F"] * n,
            "Año_de_nacimiento": [1990] * n,
            "Inicio_del_viaje": start.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Fin_del_viaje": (start + pd.Timedelta("15min")).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Origen_Id": origins,
            "Destino_Id": destinations,
        }
    )

@pytest.fixture(autouse=True)
def data_dirs(tmp_path, monkeypatch, raw_stations_df):
    raw_dir = tmp_path / "data" / "raw"
    interim_dir = tmp_path / "data" / "interim"
    raw_dir.mkdir(parents=True)

    monkeypatch.setattr(dataset, "PROJ_ROOT", tmp_path)
    monkeypatch.setattr(dataset, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(dataset, "INTERIM_DATA_DIR", interim_dir)
    manifest_path = tmp_path / "data" / "manifest.json"
    monkeypatch.setattr(dataset, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(dataset.write_manifest, "__defaults__", (manifest_path,))

    # El viaje 3 aparece en ambos meses: debe quedar una sola vez tras la ingesta.
    july = make_raw_trips([1, 2, 3], origins=[1, 2, 116], destinations=[2, 1, 1])
    august = make_raw_trips([4, 5], origins=[2, 1], destinations=[138, 2], month="08")
    august = pd.concat([august, july.iloc[[2]]], ignore_index=True)

    july.to_csv(raw_dir / DATA_METADATA["monthly"][0], index=False)
    august.to_csv(raw_dir / DATA_METADATA["monthly"][1], index=False)
    raw_stations_df.to_csv(raw_dir / DATA_METADATA["stations"], index=False)

    return {"raw": raw_dir, "interim": interim_dir, "root": tmp_path}

@pytest.fixture(autouse=True)
def mock_features(monkeypatch):
    FEATURE_FUNCTIONS = [
    "create_minutes_column",
    "delete_duplicates_and_nulls",
    "delete_duplicates_and_reset_idx",
    "delete_user_data_columns",
    "transform_datetime_columns",
]
    mocks = {}
    for name in FEATURE_FUNCTIONS:
        mock = MagicMock(name=name, side_effect=lambda frame, *args, **kwargs: frame)
        monkeypatch.setattr(dataset, name, mock)
        mocks[name] = mock
    yield mocks

# ---------------------------------------------------------------------------
# interim_dataset_path
# ---------------------------------------------------------------------------
def test_interim_dataset_path(data_dirs):
    path = dataset.interim_dataset_path(DATA_METADATA)

    assert path.parent == data_dirs["interim"]
    assert path.name == "joined_trips_stations-2026-07-2026-08.parquet"


def test_interim_dataset_path_requires_data_range():
    with pytest.raises(KeyError):
        dataset.interim_dataset_path({"monthly": [], "stations": "x.csv"})


# ---------------------------------------------------------------------------
# build_initial_dataset
# ---------------------------------------------------------------------------
def test_build_initial_dataset_without_saving():
    dataset.build_initial_dataset(DATA_METADATA)

    assert not dataset.interim_dataset_path(DATA_METADATA).exists()


def test_build_initial_dataset_saves_parquet():
    result = dataset.build_initial_dataset(DATA_METADATA, save_interim=True)

    saved = pd.read_parquet(dataset.interim_dataset_path(DATA_METADATA))
    assert len(saved) == len(result)
    assert list(saved.columns) == list(result.columns)


# ---------------------------------------------------------------------------
# load_monthly_trips
# ---------------------------------------------------------------------------
def test_load_monthly_trips_concatenates_and_deduplicates(mock_features):
    dedup = mock_features["delete_duplicates_and_reset_idx"]
    deduplicated = pd.DataFrame({"Viaje_Id": [1, 2, 3, 4, 5]})
    dedup.side_effect = None
    dedup.return_value = deduplicated

    result = dataset.load_monthly_trips(DATA_METADATA["monthly"])

    dedup.assert_called_once()
    combined = dedup.call_args.args[0]  # lo que load_monthly_trips le pasó
    assert list(combined["Viaje_Id"]) == [1, 2, 3, 4, 5, 3]
    assert list(combined.index) == list(range(6))
    assert result is deduplicated


def test_load_monthly_trips_missing_file_raises(mock_features):
    with pytest.raises(FileNotFoundError):
        dataset.load_monthly_trips(["no_existe.csv"])

    mock_features["delete_duplicates_and_reset_idx"].assert_not_called()


# ---------------------------------------------------------------------------
# clean_trips
# ---------------------------------------------------------------------------
def test_clean_trips_chains_feature_functions(mock_features):
    REQUIRED_TRIP_COLUMNS = ["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
    DATETIME_COLUMNS = ["Inicio_del_viaje", "Fin_del_viaje"]
    raw = make_raw_trips([1], origins=[1], destinations=[2])
    without_users, deduplicated, with_dates, with_minutes = (
        pd.DataFrame({"paso": [i]}) for i in range(4)
    )
    mock_features["delete_user_data_columns"].side_effect = None
    mock_features["delete_user_data_columns"].return_value = without_users
    mock_features["delete_duplicates_and_nulls"].side_effect = None
    mock_features["delete_duplicates_and_nulls"].return_value = deduplicated
    mock_features["transform_datetime_columns"].side_effect = None
    mock_features["transform_datetime_columns"].return_value = with_dates
    mock_features["create_minutes_column"].side_effect = None
    mock_features["create_minutes_column"].return_value = with_minutes

    result = dataset.clean_trips(raw)

    mock_features["delete_user_data_columns"].assert_called_once_with(raw, USER_COLUMNS)
    mock_features["delete_duplicates_and_nulls"].assert_called_once_with(
        without_users, REQUIRED_TRIP_COLUMNS
    )
    mock_features["transform_datetime_columns"].assert_called_once_with(
        deduplicated, DATETIME_COLUMNS
    )
    mock_features["create_minutes_column"].assert_called_once_with(with_dates)
    assert result is with_minutes

# ---------------------------------------------------------------------------
# get_clean_stations_df
# ---------------------------------------------------------------------------
def test_get_clean_stations_df_renames_and_excludes_replaced_ids():
    result = dataset.get_clean_stations_df(DATA_METADATA["stations"])

    assert {"Estacion_Id", "Nombre", "Latitud", "Longitud", "No_bicis"} <= set(result.columns)
    assert not result["Estacion_Id"].isin([116, 138]).any()
    assert sorted(result["Estacion_Id"]) == [1, 2, 248, 249]

# ---------------------------------------------------------------------------
# join_station_metadata
# ---------------------------------------------------------------------------
@pytest.fixture
def stations_df() -> pd.DataFrame:
    return dataset.get_clean_stations_df(DATA_METADATA["stations"])


def test_join_station_metadata_adds_origin_and_destination_columns(stations_df):
    trips = pd.DataFrame({"Viaje_Id": [1, 2], "Origen_Id": [1, 2], "Destino_Id": [2, 1]})

    result = dataset.join_station_metadata(trips, stations_df)

    assert len(result) == len(trips)
    assert list(result["Origen_Nombre"]) == ["A", "B"]
    assert list(result["Destino_Nombre"]) == ["B", "A"]
    assert list(result["Origen_No_bicis"]) == [10, 12]
    assert "Origen_Nombre" not in trips.columns


def test_join_station_metadata_replaces_deprecated_station_ids(stations_df):
    trips = pd.DataFrame({"Viaje_Id": [1], "Origen_Id": [116], "Destino_Id": [138]})

    result = dataset.join_station_metadata(trips, stations_df)

    assert result.loc[0, "Origen_Id"] == 248
    assert result.loc[0, "Destino_Id"] == 249
    assert result.loc[0, "Origen_Nombre"] == "Nueva 248"


def test_join_station_metadata_unknown_station_gives_nan(stations_df):
    trips = pd.DataFrame({"Viaje_Id": [1], "Origen_Id": [999], "Destino_Id": [1]})

    result = dataset.join_station_metadata(trips, stations_df)

    assert len(result) == 1
    assert pd.isna(result.loc[0, "Origen_Nombre"])


def test_join_station_metadata_rejects_duplicated_stations(stations_df):
    duplicated = pd.concat([stations_df, stations_df.iloc[[0]]], ignore_index=True)
    trips = pd.DataFrame({"Viaje_Id": [1], "Origen_Id": [1], "Destino_Id": [2]})

    with pytest.raises(pd.errors.MergeError):
        dataset.join_station_metadata(trips, duplicated)

# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------
def test_sha256_file(tmp_path):
    content = b"Viaje_Id,Origen_Id\n1,2\n"
    file_path = tmp_path / "f.csv"
    file_path.write_bytes(content)

    assert dataset.sha256_file(file_path) == hashlib.sha256(content).hexdigest()


def test_sha256_file_same_result_with_small_chunks(tmp_path):
    file_path = tmp_path / "f.bin"
    file_path.write_bytes(bytes(range(256)) * 50)

    assert dataset.sha256_file(file_path, chunk_size=7) == dataset.sha256_file(file_path)


def test_sha256_file_empty_file(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_bytes(b"")

    assert dataset.sha256_file(file_path) == hashlib.sha256(b"").hexdigest()


def test_sha256_file_changes_when_content_changes(tmp_path):
    file_path = tmp_path / "f.csv"
    file_path.write_text("a,b\n1,2\n")
    before = dataset.sha256_file(file_path)
    file_path.write_text("a,b\n1,3\n")

    assert dataset.sha256_file(file_path) != before


# ---------------------------------------------------------------------------
# describe_raw_file
# ---------------------------------------------------------------------------
def test_describe_raw_file(data_dirs):
    path = data_dirs["raw"] / DATA_METADATA["monthly"][0]

    entry = dataset.describe_raw_file(path)

    assert entry["file"] == path.name
    assert entry["bytes"] == path.stat().st_size
    assert entry["encoding"] == "utf-8"
    assert entry["rows"] == 3
    assert entry["sha256"] == dataset.sha256_file(path)
    assert entry["date_min"] == "2026-07-01 08:00:00"
    assert entry["date_max"] == "2026-07-01 09:00:00"


def test_describe_raw_file_without_date_column(data_dirs):
    entry = dataset.describe_raw_file(data_dirs["raw"] / "estaciones.csv", date_column=None)

    assert entry["rows"] == 6
    assert "date_min" not in entry
    assert "date_max" not in entry


def test_describe_raw_file_falls_back_to_cp1252(tmp_path):
    path = tmp_path / "cp1252.csv"
    make_raw_trips([1], origins=[1], destinations=[2]).to_csv(path, index=False, encoding="cp1252")

    entry = dataset.describe_raw_file(path)

    assert entry["encoding"] == "cp1252"
    assert entry["rows"] == 1


# ---------------------------------------------------------------------------
# compute_data_version
# ---------------------------------------------------------------------------
def test_compute_data_version_is_deterministic_and_order_independent():
    files = [{"file": "a.csv", "sha256": "111"}, {"file": "b.csv", "sha256": "222"}]

    version = dataset.compute_data_version(files, DATA_METADATA)

    assert version == dataset.compute_data_version(list(reversed(files)), DATA_METADATA)
    assert version.startswith(f"v{DATA_PIPELINE_VERSION}-")
    assert len(version.split("-")[-1]) == 12


def test_compute_data_version_changes_with_content_or_range():
    files = [{"file": "a.csv", "sha256": "111"}]
    base = dataset.compute_data_version(files, DATA_METADATA)

    other_hash = dataset.compute_data_version([{"file": "a.csv", "sha256": "999"}], DATA_METADATA)
    other_range = dataset.compute_data_version(
        files, {**DATA_METADATA, "data_range": {"from": "2026-01", "to": "2026-08"}}
    )

    assert base != other_hash
    assert base != other_range



# ---------------------------------------------------------------------------
# build_manifest / write_manifest
# ---------------------------------------------------------------------------
def test_build_manifest_without_interim_file(data_dirs):
    manifest = dataset.build_manifest(DATA_METADATA)

    assert manifest["pipeline_version"] == DATA_PIPELINE_VERSION
    assert manifest["data_range"] == DATA_METADATA["data_range"]
    assert manifest["summary"]["months"] == 2
    assert manifest["summary"]["trips_raw"] == 6  # 3 + 3, antes de quitar el duplicado
    assert manifest["summary"]["stations_catalog"] == 6
    assert [f["file"] for f in manifest["raw_files"]] == sorted(DATA_METADATA["monthly"])
    assert "interim_file" not in manifest
    assert manifest["data_version"] == dataset.compute_data_version(
        manifest["raw_files"], DATA_METADATA
    )


def test_build_manifest_with_interim_file(data_dirs):
    result = dataset.build_initial_dataset(DATA_METADATA, save_interim=True)

    manifest = dataset.build_manifest(DATA_METADATA, result)

    interim = dataset.interim_dataset_path(DATA_METADATA)
    assert manifest["interim_file"]["rows"] == len(result)
    assert manifest["interim_file"]["sha256"] == dataset.sha256_file(interim)
    assert manifest["interim_file"]["path"] == interim.relative_to(data_dirs["root"]).as_posix()
    assert manifest["summary"]["trips_clean"] == len(result)


def test_build_manifest_data_version_is_reproducible(data_dirs):
    first = dataset.build_manifest(DATA_METADATA)
    second = dataset.build_manifest(DATA_METADATA)

    assert first["data_version"] == second["data_version"]


def test_write_manifest_roundtrip(tmp_path):
    manifest = {"data_version": "v1.0-abc", "nota": "Año con ñ"}
    path = tmp_path / "nested" / "manifest.json"

    written = dataset.write_manifest(manifest, path)

    assert written == path
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    assert "Año con ñ" in path.read_text(encoding="utf-8")  # ensure_ascii=False


# ---------------------------------------------------------------------------
# run_ingest / load_base_dataset
# ---------------------------------------------------------------------------
def test_run_ingest_writes_parquet_and_manifest(data_dirs):
    result, manifest = dataset.run_ingest(DATA_METADATA)

    assert dataset.interim_dataset_path(DATA_METADATA).exists()
    saved = json.loads(dataset.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert saved["data_version"] == manifest["data_version"]
    assert saved["summary"]["trips_clean"] == len(result)


def test_load_base_dataset_uses_cache_when_hash_matches(data_dirs, monkeypatch):
    expected, _ = dataset.run_ingest(DATA_METADATA)

    def fail(*args, **kwargs):
        raise AssertionError("No debería reconstruir el dataset")

    monkeypatch.setattr(dataset, "run_ingest", fail)
    result, manifest = dataset.load_base_dataset(DATA_METADATA)

    pd.testing.assert_frame_equal(result, expected)
    assert "interim_file" in manifest


def test_load_base_dataset_rebuilds_when_parquet_changed(data_dirs, monkeypatch):
    dataset.run_ingest(DATA_METADATA)
    interim = dataset.interim_dataset_path(DATA_METADATA)
    pd.DataFrame({"x": [1]}).to_parquet(interim)  # hash ya no coincide con el manifiesto

    calls = []
    monkeypatch.setattr(dataset, "run_ingest", lambda meta: calls.append(meta) or ("df", {}))

    assert dataset.load_base_dataset(DATA_METADATA) == ("df", {})
    assert calls == [DATA_METADATA]


def test_load_base_dataset_rebuilds_without_manifest(data_dirs, monkeypatch):
    calls = []
    monkeypatch.setattr(dataset, "run_ingest", lambda meta: calls.append(meta) or ("df", {}))

    dataset.load_base_dataset(DATA_METADATA)

    assert calls == [DATA_METADATA]
