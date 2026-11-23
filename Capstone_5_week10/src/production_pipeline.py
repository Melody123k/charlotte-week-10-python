
"""Week 10 Class 5 capstone: production-data quality pipeline."""

from pathlib import Path
import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    "well_id",
    "timestamp",
    "oil_bbl",
    "water_bbl",
    "gas_mscf",
    "pressure_psi",
    "status",
    "ingested_at",
]

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"

MEASUREMENT_COLUMNS = [
    "oil_bbl",
    "water_bbl",
    "gas_mscf",
    "pressure_psi",
]


def ingest_and_validate(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read CSV, validate schema and separate invalid timestamps."""

    df = pd.read_csv(path)

    # Check that all required columns exist.
    missing_columns = [
        col for col in EXPECTED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only the required source columns.
    df = df[EXPECTED_COLUMNS].copy()

    # Preserve the original timestamp for auditing.
    original_timestamp = df["timestamp"].copy()

    # Parse timestamps using the required format.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format=TIMESTAMP_FORMAT,
        errors="coerce",
    )

    # Parse ingestion timestamps.
    df["ingested_at"] = pd.to_datetime(
        df["ingested_at"],
        format=TIMESTAMP_FORMAT,
        errors="coerce",
    )

    # Separate invalid timestamps.
    rejected = df[df["timestamp"].isna()].copy()

    rejected["rejection_reason"] = (
        "Invalid or missing timestamp"
    )

    rejected["original_timestamp"] = original_timestamp.loc[
        rejected.index
    ]

    # Keep only valid timestamp records.
    valid = df[df["timestamp"].notna()].copy()

    return valid, rejected


def repair_and_align(
    valid_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate, clean measurements and align all wells to six-hour intervals."""

    df = valid_rows.copy()

    # Convert well IDs to strings for consistent grouping.
    df["well_id"] = df["well_id"].astype(str)

    # Count source rows before deduplication.
    source_counts = df.groupby("well_id").size()

    # Deduplicate using the latest valid ingestion timestamp.
    df = df.sort_values(
        "ingested_at",
        na_position="first",
        kind="stable",
    )

    df = df.drop_duplicates(
        subset=["well_id", "timestamp"],
        keep="last",
    )

    deduplicated_counts = df.groupby("well_id").size()

    # Convert measurements to numeric values.
    for col in MEASUREMENT_COLUMNS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        # Negative measurements become missing.
        df.loc[df[col] < 0, col] = np.nan

    # Keep one aligned dataset for all wells.
    aligned_wells = []
    quality_records = []

    for well_id, well_df in df.groupby("well_id"):

        well_df = well_df.sort_values("timestamp").copy()

        well_df = well_df.set_index("timestamp")

        # Create six-hourly timestamps from earliest to latest.
        grid = pd.date_range(
            start=well_df.index.min(),
            end=well_df.index.max(),
            freq="6h",
        )

        # Preserve any original observations that are off the grid.
        full_index = grid.union(well_df.index).sort_values()

        # Reindex to include missing timestamps.
        aligned = well_df.reindex(full_index)

        aligned.index.name = "timestamp"

        # Identify inserted timestamps.
        aligned["is_inserted_slot"] = ~aligned.index.isin(
            well_df.index
        )

        inserted_count = int(
            aligned["is_inserted_slot"].sum()
        )

        # Remember missing measurements before interpolation.
        missing_before = {
            col: int(aligned[col].isna().sum())
            for col in MEASUREMENT_COLUMNS
        }

        # Interpolate pressure only for exactly one internal gap.
        pressure = aligned["pressure_psi"].copy()

        missing = pressure.isna().to_numpy()

        i = 0

        while i < len(pressure):

            if missing[i]:
                start = i

                while i < len(pressure) and missing[i]:
                    i += 1

                gap_length = i - start

                # Only fill one missing value between valid values.
                if (
                    gap_length == 1
                    and start > 0
                    and i < len(pressure)
                    and pd.notna(pressure.iloc[start - 1])
                    and pd.notna(pressure.iloc[i])
                ):
                    pressure.iloc[start] = np.mean([
                        pressure.iloc[start - 1],
                        pressure.iloc[i],
                    ])

            else:
                i += 1

        aligned["pressure_psi"] = pressure

        # Calculate liquid only when both inputs are available.
        aligned["liquid_bbl"] = (
            aligned["oil_bbl"] + aligned["water_bbl"]
        )

        # Calculate water cut only when liquid is greater than zero.
        aligned["water_cut_pct"] = np.where(
            aligned["liquid_bbl"] > 0,
            (
                aligned["water_bbl"]
                / aligned["liquid_bbl"]
            ) * 100,
            np.nan,
        )

        aligned = aligned.reset_index()

        aligned["well_id"] = well_id

        aligned_wells.append(aligned)

        # Create an auditable quality report.
        quality_records.append({
            "well_id": well_id,
            "source_rows": int(source_counts.get(well_id, 0)),
            "deduplicated_rows_removed": int(
                source_counts.get(well_id, 0)
                - deduplicated_counts.get(well_id, 0)
            ),
            "inserted_slots": inserted_count,
            "missing_oil_bbl": missing_before["oil_bbl"],
            "missing_water_bbl": missing_before["water_bbl"],
            "missing_gas_mscf": missing_before["gas_mscf"],
            "missing_pressure_psi": missing_before["pressure_psi"],
            "remaining_missing_pressure": int(
                aligned["pressure_psi"].isna().sum()
            ),
        })

    cleaned = pd.concat(
        aligned_wells,
        ignore_index=True,
    )

    quality_report = pd.DataFrame(quality_records)

    return cleaned, quality_report


def summarize_daily(cleaned_rows: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily production totals and completeness."""

    df = cleaned_rows.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract the production date.
    df["date"] = df["timestamp"].dt.floor("D")

    daily_records = []

    for (well_id, date), group in df.groupby(
        ["well_id", "date"]
    ):

        expected_slots = 4

        record = {
            "well_id": well_id,
            "date": date,
            "expected_slots": expected_slots,
        }

        # Calculate totals without turning all-missing data into zero.
        for col in ["oil_bbl", "water_bbl", "gas_mscf"]:

            record[f"daily_{col}"] = group[col].sum(
                min_count=1
            )

            record[f"{col}_valid_readings"] = int(
                group[col].notna().sum()
            )

            record[f"{col}_complete"] = (
                record[f"{col}_valid_readings"]
                == expected_slots
            )

        # Overall completeness requires all production readings.
        record["is_complete"] = all(
            record[f"{col}_complete"]
            for col in ["oil_bbl", "water_bbl", "gas_mscf"]
        )

        daily_records.append(record)

    daily_summary = pd.DataFrame(daily_records)

    return daily_summary.sort_values(
        ["well_id", "date"]
    ).reset_index(drop=True)


def run_pipeline(
    input_csv: str | Path,
    output_dir: str | Path,
) -> None:
    """Run the complete pipeline and export four CSV reports."""

    output_path = Path(output_dir)

    # Create output folder if it does not exist.
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 1: Ingest and validate.
    valid_rows, rejected_rows = ingest_and_validate(
        input_csv
    )

    # Step 2: Clean and align.
    cleaned_rows, quality_report = repair_and_align(
        valid_rows
    )

    # Step 3: Generate daily summaries.
    daily_summary = summarize_daily(
        cleaned_rows
    )

    # Add invalid timestamp counts to the quality report.
    if not rejected_rows.empty:
        rejected_counts = (
            rejected_rows.assign(
                well_id=rejected_rows["well_id"].fillna("UNKNOWN")
            )
            .groupby("well_id")
            .size()
        )
    else:
        rejected_counts = pd.Series(dtype="int64")

    quality_report["invalid_timestamp_rows"] = (
        quality_report["well_id"]
        .map(rejected_counts)
        .fillna(0)
        .astype(int)
    )

    # Export all four required CSV files.
    cleaned_rows.to_csv(
        output_path / "cleaned_production.csv",
        index=False,
    )

    quality_report.to_csv(
        output_path / "data_quality_report.csv",
        index=False,
    )

    daily_summary.to_csv(
        output_path / "daily_production_summary.csv",
        index=False,
    )

    rejected_rows.to_csv(
        output_path / "rejected_source_records.csv",
        index=False,
    )

    print("Production pipeline completed successfully!")

    print(f"Cleaned records: {len(cleaned_rows)}")
    print(f"Rejected records: {len(rejected_rows)}")
    print(f"Daily summaries: {len(daily_summary)}")

    print(f"Reports saved in: {output_path}")
