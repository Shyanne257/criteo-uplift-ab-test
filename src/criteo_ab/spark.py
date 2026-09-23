"""Shared SparkSession factory and dataset schema."""
from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql import types as T

FEATURES = [f"f{i}" for i in range(12)]
BINARY_COLS = ["treatment", "conversion", "visit", "exposure"]

# Column order of criteo-research-uplift-v2.1.csv.gz.
# An explicit schema avoids a full extra pass over 14M rows for inferSchema.
SCHEMA = T.StructType(
    [T.StructField(f, T.DoubleType(), True) for f in FEATURES]
    + [T.StructField(c, T.IntegerType(), True) for c in BINARY_COLS]
)


def get_spark(app_name: str = "criteo-uplift-ab") -> SparkSession:
    """Local SparkSession sized for a laptop.

    Driver memory can be overridden with SPARK_DRIVER_MEMORY (e.g. "6g").
    """
    return (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
