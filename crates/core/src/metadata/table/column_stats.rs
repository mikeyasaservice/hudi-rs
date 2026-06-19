/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
//! Decoding of the metadata table `column_stats` partition.
//!
//! The `column_stats` partition stores per-column min/max statistics for each data
//! file as Avro-serialized `HoodieMetadataRecord`s. Each record carries a
//! `columnStatsMetadata` payload (a `HoodieMetadataColumnStats`) with:
//! - `fileName` — the data file the stats describe
//! - `columnName` — the column the stats describe
//! - `minValue` / `maxValue` — typed "wrapper" unions (e.g. `IntWrapper { value }`)
//! - `nullCount` / `valueCount` and an `isDeleted` tombstone flag
//!
//! Decoded stats are converted into [`StatisticsContainer`]s keyed by file name so the
//! existing [`crate::table::file_pruner::FilePruner`] can skip files without reading
//! every Parquet footer.

use std::collections::HashMap;
use std::sync::Arc;

use apache_avro::Schema as AvroSchema;
use apache_avro::types::Value as AvroValue;
use arrow_array::{ArrayRef, BooleanArray, Float32Array, Float64Array};
use arrow_schema::{DataType, Schema};

use crate::Result;
use crate::error::CoreError;
use crate::file_group::base_file::BaseFile;
use crate::hfile::{HFileReader, HFileRecord};
use crate::metadata::table::records::decode_avro_value;
use crate::statistics::{
    ColumnStatistics, StatisticsContainer, StatsGranularity, bytes_to_array, int32_to_array,
    int64_to_array,
};
use crate::table::Table;
use std::str::FromStr;

/// Decoded statistics for one column of one data file from the `column_stats` partition.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStatsRecord {
    /// Data file the statistics describe (base file name).
    pub file_name: String,
    /// Column the statistics describe.
    pub column_name: String,
    /// Minimum value as the decoded Avro scalar (unwrapped from its type wrapper).
    pub min_value: Option<AvroValue>,
    /// Maximum value as the decoded Avro scalar (unwrapped from its type wrapper).
    pub max_value: Option<AvroValue>,
    /// Number of nulls in the column for this file, when recorded.
    pub null_count: Option<i64>,
    /// Number of values (including nulls) in the column for this file, when recorded.
    pub value_count: Option<i64>,
    /// Tombstone flag — when set, the stats for this (file, column) have been deleted.
    pub is_deleted: bool,
}

/// The metadata table partition name that stores per-file column statistics.
pub const COLUMN_STATS_PARTITION_NAME: &str = "column_stats";

/// The metadata table partition name that stores per-partition column statistics.
pub const PARTITION_STATS_PARTITION_NAME: &str = "partition_stats";

/// Recursively unwrap Avro unions to reach the underlying value.
fn unwrap_union(value: &AvroValue) -> &AvroValue {
    match value {
        AvroValue::Union(_, inner) => unwrap_union(inner),
        other => other,
    }
}

fn find_field<'a>(fields: &'a [(String, AvroValue)], name: &str) -> Option<&'a AvroValue> {
    fields
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| unwrap_union(v))
}

fn extract_string(value: &AvroValue) -> Option<String> {
    match unwrap_union(value) {
        AvroValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn extract_long(value: &AvroValue) -> Option<i64> {
    match unwrap_union(value) {
        AvroValue::Long(n) => Some(*n),
        AvroValue::Int(n) => Some(*n as i64),
        _ => None,
    }
}

fn extract_bool(value: &AvroValue) -> Option<bool> {
    match unwrap_union(value) {
        AvroValue::Boolean(b) => Some(*b),
        _ => None,
    }
}

/// Extract the scalar held by a min/max field.
///
/// Min/max are stored as a union of type wrappers (e.g. `LongWrapper { value: long }`).
/// Returns the unwrapped scalar, or `None` for a null/absent value.
fn extract_wrapped_scalar(value: &AvroValue) -> Option<AvroValue> {
    match unwrap_union(value) {
        AvroValue::Null => None,
        // Type wrapper record: a single `value` field carrying the typed scalar.
        AvroValue::Record(fields) => find_field(fields, "value").cloned(),
        // Already a bare scalar (defensive — not the usual wire shape).
        other => Some(other.clone()),
    }
}

/// Decode a `column_stats` HFile record value using the provided Avro schema.
///
/// Returns `Ok(None)` for tombstone records (empty value) or records that do not
/// carry a `columnStatsMetadata` payload.
pub fn decode_column_stats_record_with_schema(
    record: &HFileRecord,
    schema: &AvroSchema,
) -> Result<Option<ColumnStatsRecord>> {
    let value = record.value();
    if value.is_empty() {
        return Ok(None);
    }

    let avro_value = decode_avro_value(value, schema)?;
    let AvroValue::Record(top_fields) = &avro_value else {
        return Ok(None);
    };

    // The metadata record schema names this field in PascalCase (`ColumnStatsMetadata`),
    // unlike the camelCase `filesystemMetadata` used by the files partition.
    let Some(AvroValue::Record(cs_fields)) =
        find_field(top_fields, "ColumnStatsMetadata").map(unwrap_union)
    else {
        return Ok(None);
    };

    let file_name = find_field(cs_fields, "fileName")
        .and_then(extract_string)
        .unwrap_or_default();
    let column_name = find_field(cs_fields, "columnName")
        .and_then(extract_string)
        .unwrap_or_default();
    let min_value = find_field(cs_fields, "minValue").and_then(extract_wrapped_scalar);
    let max_value = find_field(cs_fields, "maxValue").and_then(extract_wrapped_scalar);
    let null_count = find_field(cs_fields, "nullCount").and_then(extract_long);
    let value_count = find_field(cs_fields, "valueCount").and_then(extract_long);
    let is_deleted = find_field(cs_fields, "isDeleted")
        .and_then(extract_bool)
        .unwrap_or(false);

    Ok(Some(ColumnStatsRecord {
        file_name,
        column_name,
        min_value,
        max_value,
        null_count,
        value_count,
        is_deleted,
    }))
}

/// Convert a decoded Avro min/max scalar into a single-element Arrow array typed as
/// the table-schema column `data_type`.
///
/// Conversion is driven by `data_type` (not the Avro branch) so the resulting array
/// matches the type that filter values are cast to, which `arrow_ord` comparisons
/// require. Returns `None` for types not yet supported (e.g. decimals); the pruner
/// treats missing stats as "include the file", so this is always safe.
fn avro_scalar_to_array(scalar: &AvroValue, data_type: &DataType) -> Option<ArrayRef> {
    match scalar {
        AvroValue::Boolean(b) => match data_type {
            DataType::Boolean => Some(Arc::new(BooleanArray::from(vec![*b])) as ArrayRef),
            _ => None,
        },
        AvroValue::Int(n) | AvroValue::Date(n) | AvroValue::TimeMillis(n) => {
            Some(int32_to_array(*n, data_type))
        }
        AvroValue::Long(n)
        | AvroValue::TimeMicros(n)
        | AvroValue::TimestampMillis(n)
        | AvroValue::TimestampMicros(n) => Some(int64_to_array(*n, data_type)),
        AvroValue::Float(f) => match data_type {
            DataType::Float32 => Some(Arc::new(Float32Array::from(vec![*f])) as ArrayRef),
            DataType::Float64 => Some(Arc::new(Float64Array::from(vec![*f as f64])) as ArrayRef),
            _ => None,
        },
        AvroValue::Double(d) => match data_type {
            DataType::Float64 => Some(Arc::new(Float64Array::from(vec![*d])) as ArrayRef),
            _ => None,
        },
        AvroValue::String(s) => Some(bytes_to_array(s.as_bytes(), data_type)),
        AvroValue::Bytes(b) | AvroValue::Fixed(_, b) => Some(bytes_to_array(b, data_type)),
        _ => None,
    }
}

/// Group decoded column-stats records into [`StatisticsContainer`]s keyed by each
/// record's subject — the data file name for `column_stats`, or the partition path for
/// `partition_stats` (both live in the record's `file_name` field).
///
/// Records whose column is absent from `schema`, or that are tombstones, are skipped.
/// The returned map is directly consumable by
/// [`crate::table::file_pruner::FilePruner::should_include`].
pub fn column_stats_to_containers(
    records: &[ColumnStatsRecord],
    schema: &Schema,
) -> HashMap<String, StatisticsContainer> {
    let type_by_col: HashMap<&str, &DataType> = schema
        .fields()
        .iter()
        .map(|f| (f.name().as_str(), f.data_type()))
        .collect();

    let mut out: HashMap<String, StatisticsContainer> = HashMap::new();
    for rec in records {
        if rec.is_deleted {
            continue;
        }
        let Some(&data_type) = type_by_col.get(rec.column_name.as_str()) else {
            continue;
        };

        let min_value = rec
            .min_value
            .as_ref()
            .and_then(|v| avro_scalar_to_array(v, data_type));
        let max_value = rec
            .max_value
            .as_ref()
            .and_then(|v| avro_scalar_to_array(v, data_type));

        let container = out
            .entry(rec.file_name.clone())
            .or_insert_with(|| StatisticsContainer::new(StatsGranularity::File));
        container.columns.insert(
            rec.column_name.clone(),
            ColumnStatistics {
                column_name: rec.column_name.clone(),
                data_type: data_type.clone(),
                min_value,
                max_value,
            },
        );
    }
    out
}

impl Table {
    /// Read the metadata table `column_stats` partition into per-data-file statistics,
    /// typed against `schema` (the data table schema).
    ///
    /// The returned map is keyed by base data file name and is directly consumable by
    /// [`crate::table::file_pruner::FilePruner::should_include`]. Returns an empty map
    /// when the table has no completed commits or the metadata table is not enabled.
    ///
    /// Must be called on a DATA table, not a METADATA table.
    pub async fn read_metadata_table_column_stats(
        &self,
        schema: &Schema,
    ) -> Result<HashMap<String, StatisticsContainer>> {
        let metadata_table = self.get_or_init_metadata_table().await?;
        metadata_table.fetch_column_stats_containers(schema).await
    }

    /// Read the metadata table `partition_stats` partition into per-partition statistics,
    /// typed against `schema` (the data table schema).
    ///
    /// The `partition_stats` partition reuses the column-stats payload but aggregates each
    /// column's min/max per data table partition; the returned map is therefore keyed by
    /// partition path and is consumable by
    /// [`crate::table::file_pruner::FilePruner::should_include`] to skip whole partitions.
    ///
    /// Must be called on a DATA table, not a METADATA table.
    pub async fn read_metadata_table_partition_stats(
        &self,
        schema: &Schema,
    ) -> Result<HashMap<String, StatisticsContainer>> {
        let metadata_table = self.get_or_init_metadata_table().await?;
        metadata_table
            .fetch_partition_stats_containers(schema)
            .await
    }

    /// Fetch and group the `column_stats` records into per-data-file containers.
    pub(crate) async fn fetch_column_stats_containers(
        &self,
        schema: &Schema,
    ) -> Result<HashMap<String, StatisticsContainer>> {
        self.fetch_index_stats_containers(COLUMN_STATS_PARTITION_NAME, schema)
            .await
    }

    /// Fetch and group the `partition_stats` records into per-partition containers.
    pub(crate) async fn fetch_partition_stats_containers(
        &self,
        schema: &Schema,
    ) -> Result<HashMap<String, StatisticsContainer>> {
        self.fetch_index_stats_containers(PARTITION_STATS_PARTITION_NAME, schema)
            .await
    }

    /// Fetch and decode the column-stats records from the latest committed base HFile of
    /// each file group in the given index partition (`column_stats` or `partition_stats`),
    /// grouping them into statistics containers keyed by the record's subject (data file
    /// name for `column_stats`, partition path for `partition_stats`).
    ///
    /// Each index partition is bucketed into multiple file groups, each a base HFile
    /// (produced by metadata-table compaction) plus delta log files. Only the base HFiles
    /// are read here; stats carried solely by post-compaction delta logs are not yet
    /// sourced, but the subjects they would cover fall back to footer-based pruning (for
    /// `column_stats`) or to being kept (for `partition_stats`), which is safe. Reading the
    /// base HFiles directly also sidesteps log-only file slices, which the file-group
    /// builder does not yet support. Must be called on a METADATA table.
    async fn fetch_index_stats_containers(
        &self,
        partition_name: &str,
        schema: &Schema,
    ) -> Result<HashMap<String, StatisticsContainer>> {
        let Some(latest_ts) = self.timeline.get_latest_commit_timestamp_as_option() else {
            return Ok(HashMap::new());
        };

        let storage = self.file_system_view.storage.as_ref();
        let files = storage.list_files(Some(partition_name)).await?;

        // Keep the latest committed base HFile per file group (bucket).
        let mut latest_base_by_file_id: HashMap<String, BaseFile> = HashMap::new();
        for file in &files {
            if !file.name.ends_with(".hfile") {
                continue;
            }
            let Ok(base_file) = BaseFile::from_str(&file.name) else {
                continue;
            };
            if base_file.commit_timestamp.as_str() > latest_ts {
                continue;
            }
            latest_base_by_file_id
                .entry(base_file.file_id.clone())
                .and_modify(|existing| {
                    if base_file.commit_timestamp > existing.commit_timestamp {
                        *existing = base_file.clone();
                    }
                })
                .or_insert(base_file);
        }

        let mut records: Vec<ColumnStatsRecord> = Vec::new();
        for base_file in latest_base_by_file_id.values() {
            let relative_path = format!("{partition_name}/{}", base_file.file_name());
            let mut reader = HFileReader::open(storage, &relative_path)
                .await
                .map_err(|e| {
                    CoreError::MetadataTable(format!(
                        "Failed to open {partition_name} base file {relative_path}: {e:?}"
                    ))
                })?;
            let avro_schema = reader
                .get_avro_schema()
                .map_err(|e| {
                    CoreError::MetadataTable(format!(
                        "Failed to get {partition_name} schema: {e:?}"
                    ))
                })?
                .cloned();
            let Some(avro_schema) = avro_schema else {
                continue;
            };
            let hfile_records = reader.collect_records().map_err(|e| {
                CoreError::MetadataTable(format!("Failed to read {partition_name} records: {e:?}"))
            })?;
            for record in hfile_records {
                if let Some(decoded) =
                    decode_column_stats_record_with_schema(&record, &avro_schema)?
                {
                    records.push(decoded);
                }
            }
        }

        Ok(column_stats_to_containers(&records, schema))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::filter::Filter;
    use crate::hfile::{HFileReader, HFileRecord};
    use crate::metadata::table::records::parse_avro_schema;
    use crate::table::file_pruner::FilePruner;
    use arrow_array::Int64Array;
    use arrow_schema::Field;
    use hudi_test::QuickstartTripsTable;
    use std::path::PathBuf;

    fn column_stats_dir() -> PathBuf {
        let table_path = QuickstartTripsTable::V8Trips8I3U1D.path_to_mor_avro();
        PathBuf::from(table_path)
            .join(".hoodie")
            .join("metadata")
            .join("column_stats")
    }

    /// Decode every record from the base HFiles of the `column_stats` partition.
    fn decode_base_records() -> Vec<ColumnStatsRecord> {
        let dir = column_stats_dir();
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.extension().map(|e| e == "hfile").unwrap_or(false) {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let mut reader = HFileReader::new(bytes).unwrap();
            let schema = reader.get_avro_schema().unwrap().unwrap().clone();
            for record in reader.collect_records().unwrap() {
                if let Some(decoded) =
                    decode_column_stats_record_with_schema(&record, &schema).unwrap()
                {
                    out.push(decoded);
                }
            }
        }
        out
    }

    #[test]
    fn test_decode_column_stats_from_fixture() {
        let records = decode_base_records();
        assert!(!records.is_empty(), "expected decoded column-stats records");
        // Every decoded record names a file and a column.
        assert!(
            records
                .iter()
                .all(|r| !r.file_name.is_empty() && !r.column_name.is_empty())
        );

        // The `driver` column is a string; min/max decode to String scalars.
        let driver = records
            .iter()
            .find(|r| {
                r.column_name == "driver"
                    && r.file_name.ends_with(".parquet")
                    && r.min_value.is_some()
            })
            .expect("expected a driver column-stats record for a parquet file");
        assert!(!driver.is_deleted);
        assert!(matches!(driver.min_value, Some(AvroValue::String(_))));
        assert!(matches!(driver.max_value, Some(AvroValue::String(_))));
        assert!(driver.value_count.unwrap_or(0) > 0);
    }

    #[test]
    fn test_column_stats_to_containers_enables_pruning() {
        let records = decode_base_records();
        let schema = Schema::new(vec![Field::new("driver", DataType::Utf8, true)]);
        let containers = column_stats_to_containers(&records, &schema);
        assert!(!containers.is_empty());

        // Pick a concrete record so we know the real min value for that file.
        let driver = records
            .iter()
            .find(|r| {
                r.column_name == "driver"
                    && r.file_name.ends_with(".parquet")
                    && r.min_value.is_some()
            })
            .unwrap();
        let AvroValue::String(min_str) = driver.min_value.as_ref().unwrap() else {
            panic!("driver min should be a string");
        };
        let container = containers.get(&driver.file_name).unwrap();
        let partition_schema = Schema::empty();

        // A value above any plausible max prunes the file.
        let prune_filter = vec![Filter::try_from(("driver", "=", "zzzzzzzz")).unwrap()];
        let pruner = FilePruner::new(&prune_filter, &schema, &partition_schema).unwrap();
        assert!(
            !pruner.should_include(container),
            "file {} should be pruned for driver='zzzzzzzz'",
            driver.file_name
        );

        // The recorded min value is in range, so the file is retained.
        let keep_filter = vec![Filter::try_from(("driver", "=", min_str.as_str())).unwrap()];
        let keeper = FilePruner::new(&keep_filter, &schema, &partition_schema).unwrap();
        assert!(keeper.should_include(container));
    }

    #[test]
    fn test_avro_scalar_to_array_supported_and_unsupported() {
        // Long -> Int64 array.
        let arr = avro_scalar_to_array(&AvroValue::Long(42), &DataType::Int64).unwrap();
        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.value(0), 42);

        // Timestamp-logical scalar maps onto a timestamp column via the i64 path.
        assert!(
            avro_scalar_to_array(
                &AvroValue::TimestampMicros(1_700_000_000_000_000),
                &DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            )
            .is_some()
        );

        // Unsupported scalar variants yield None (safe: the file is included, not pruned).
        assert!(avro_scalar_to_array(&AvroValue::Null, &DataType::Int64).is_none());
        assert!(
            avro_scalar_to_array(&AvroValue::Boolean(true), &DataType::Int64).is_none(),
            "type-mismatched scalar should not fabricate an array"
        );
    }

    #[test]
    fn test_decode_empty_value_is_tombstone_none() {
        let record = HFileRecord::new(b"some-key".to_vec(), vec![]);
        let schema = parse_avro_schema(
            r#"{"type":"record","name":"T","fields":[{"name":"type","type":"int"}]}"#,
        )
        .unwrap();
        assert!(
            decode_column_stats_record_with_schema(&record, &schema)
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_read_metadata_table_column_stats_via_table_api() {
        // Exercises the full pipeline: bucketed multi-slice read of the column_stats
        // partition (base HFiles + log files), key-dedup merge, decode, and grouping.
        let table_path = QuickstartTripsTable::V8Trips8I3U1D.path_to_mor_avro();
        let table = Table::new(&table_path).await.unwrap();
        let schema = table.get_schema().await.unwrap();

        let containers = table
            .read_metadata_table_column_stats(&schema)
            .await
            .unwrap();

        assert!(!containers.is_empty(), "expected per-file column stats");
        // Stats are keyed by base data file name.
        assert!(
            containers.keys().any(|k| k.ends_with(".parquet")),
            "expected stats keyed by parquet file names, got: {:?}",
            containers.keys().collect::<Vec<_>>()
        );
        // At least one file has a column with usable min/max.
        assert!(
            containers
                .values()
                .any(|c| c.columns.values().any(|s| s.min_value.is_some())),
            "expected at least one column with a min value"
        );
        // Every recorded column belongs to the data schema.
        for container in containers.values() {
            for col in container.columns.keys() {
                assert!(
                    schema.field_with_name(col).is_ok(),
                    "stats column {col} should exist in the table schema"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_read_metadata_table_partition_stats_via_table_api() {
        let table_path = QuickstartTripsTable::V8Trips8I3U1D.path_to_mor_avro();
        let table = Table::new(&table_path).await.unwrap();
        let schema = table.get_schema().await.unwrap();

        let containers = table
            .read_metadata_table_partition_stats(&schema)
            .await
            .unwrap();

        // Keyed by partition path (not data file name).
        assert!(containers.contains_key("city=chennai"));
        assert!(containers.contains_key("city=san_francisco"));
        assert!(containers.contains_key("city=sao_paulo"));
        // Per-partition aggregates exist for data columns.
        let sf = containers.get("city=san_francisco").unwrap();
        assert!(
            sf.columns
                .get("fare")
                .map(|s| s.min_value.is_some() && s.max_value.is_some())
                .unwrap_or(false),
            "expected fare min/max for city=san_francisco"
        );
    }

    #[test]
    fn test_extract_wrapped_scalar() {
        // Wrapper record { value: <scalar> } unwraps to the scalar.
        let wrapped = AvroValue::Union(
            7,
            Box::new(AvroValue::Record(vec![(
                "value".to_string(),
                AvroValue::String("driver-A".to_string()),
            )])),
        );
        assert_eq!(
            extract_wrapped_scalar(&wrapped),
            Some(AvroValue::String("driver-A".to_string()))
        );

        // Null branch -> None.
        let null_branch = AvroValue::Union(0, Box::new(AvroValue::Null));
        assert_eq!(extract_wrapped_scalar(&null_branch), None);
    }
}
