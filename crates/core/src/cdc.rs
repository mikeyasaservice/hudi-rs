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

//! Change Data Capture (CDC) read support.
//!
//! When `hoodie.table.cdc.enabled` is set, Hudi persists per-commit change data so a CDC query
//! can return the inserts, updates, and deletes that occurred in a commit range, rather than the
//! merged snapshot. The amount of change data persisted is controlled by
//! [`crate::config::table::CdcSupplementalLoggingModeValue`].
//!
//! This module holds the storage-agnostic pieces of the CDC read path: the change-operation
//! codes and the extraction of CDC file references from commit metadata. The orchestration that
//! reads those files for a commit range lives on [`crate::table::Table::read_cdc`].

use crate::Result;
use crate::config::HudiConfigs;
use crate::config::table::CdcSupplementalLoggingModeValue;
use crate::config::table::HudiTableConfig::{CdcEnabled, CdcSupplementalLoggingMode};
use crate::error::CoreError;
use crate::util::arrow::{adapt_batch_to_schema, reconcile_schemas};
use arrow_array::{
    Array, ArrayRef, RecordBatch, StringArray, StructArray, UInt32Array, new_null_array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// CDC change-operation codes as written by Hudi into the `op` column of a change record.
pub mod op {
    /// Insert.
    pub const INSERT: &str = "i";
    /// Update.
    pub const UPDATE: &str = "u";
    /// Delete.
    pub const DELETE: &str = "d";
}

/// Returns whether CDC is enabled for the table (`hoodie.table.cdc.enabled`).
pub fn is_cdc_enabled(hudi_configs: &HudiConfigs) -> bool {
    hudi_configs.get_or_default(CdcEnabled).into()
}

/// Resolve the configured CDC supplemental logging mode, defaulting to
/// [`CdcSupplementalLoggingModeValue::DataBeforeAfter`].
pub fn supplemental_logging_mode(
    hudi_configs: &HudiConfigs,
) -> Result<CdcSupplementalLoggingModeValue> {
    let value: String = hudi_configs
        .get_or_default(CdcSupplementalLoggingMode)
        .into();
    use std::str::FromStr;
    CdcSupplementalLoggingModeValue::from_str(&value).map_err(CoreError::Config)
}

/// Extract the CDC file paths recorded in a commit's write stats.
///
/// Hudi records, per file written in a commit, a `cdcStats` map of
/// `<cdc file path> -> record count` (and some writers a single `cdcPath` string). The paths are
/// relative to the table base path. Returns them in metadata order, de-duplicated.
pub fn cdc_file_paths_from_commit_metadata(commit_metadata: &Map<String, Value>) -> Vec<String> {
    let mut paths = Vec::new();
    let Some(partition_to_write_stats) = commit_metadata
        .get("partitionToWriteStats")
        .and_then(|v| v.as_object())
    else {
        return paths;
    };

    for stats in partition_to_write_stats.values() {
        let Some(stats) = stats.as_array() else {
            continue;
        };
        for stat in stats {
            // `cdcStats` is a map of cdc file path -> record count.
            if let Some(cdc_stats) = stat.get("cdcStats").and_then(|v| v.as_object()) {
                for path in cdc_stats.keys() {
                    if !path.is_empty() && !paths.contains(path) {
                        paths.push(path.clone());
                    }
                }
            }
            // Some writers record a single `cdcPath` string instead.
            if let Some(cdc_path) = stat.get("cdcPath").and_then(|v| v.as_str()) {
                let cdc_path = cdc_path.to_string();
                if !cdc_path.is_empty() && !paths.contains(&cdc_path) {
                    paths.push(cdc_path);
                }
            }
        }
    }
    paths
}

/// The Arrow field name a CDC log/file uses for the change operation code.
pub const CDC_OP_FIELD: &str = "op";
/// The Arrow field name a CDC log/file uses for the changed record's key.
pub const CDC_RECORD_KEY_FIELD: &str = "record_key";

/// The record (struct) schema reconciled across the available before/after snapshots.
fn record_schema(after: Option<&RecordBatch>, before: Option<&RecordBatch>) -> SchemaRef {
    let schemas: Vec<SchemaRef> = [after, before]
        .into_iter()
        .flatten()
        .map(|b| b.schema())
        .collect();
    reconcile_schemas(&schemas)
}

/// Gather one struct value per requested key from `snapshot` (adapted to `record_schema`),
/// emitting a null struct where the key is `None` or not found. Used to build the before/after
/// image columns of the change records.
fn gather_records_by_key(
    snapshot: Option<&RecordBatch>,
    record_schema: &SchemaRef,
    keys: &[Option<&str>],
    record_key_field: &str,
) -> Result<ArrayRef> {
    let struct_type = DataType::Struct(record_schema.fields().clone());
    let Some(snapshot) = snapshot else {
        return Ok(new_null_array(&struct_type, keys.len()));
    };
    let adapted = adapt_batch_to_schema(snapshot, record_schema)?;
    let key_array = adapted
        .column_by_name(record_key_field)
        .ok_or_else(|| {
            CoreError::Schema(format!(
                "CDC snapshot is missing record key field '{record_key_field}'"
            ))
        })?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            CoreError::Schema(format!(
                "CDC record key field '{record_key_field}' is not a string"
            ))
        })?;

    let mut index: HashMap<&str, u32> = HashMap::with_capacity(adapted.num_rows());
    for i in 0..adapted.num_rows() {
        if key_array.is_valid(i) {
            // Later rows win, matching last-write semantics of the merged snapshot.
            index.insert(key_array.value(i), i as u32);
        }
    }
    let indices: UInt32Array = keys
        .iter()
        .map(|k| k.and_then(|k| index.get(k).copied()))
        .collect();
    let struct_full = StructArray::from(adapted);
    arrow::compute::take(&struct_full, &indices, None).map_err(CoreError::ArrowError)
}

/// Build CDC change records from per-commit `(op, record_key)` pairs and the file-group snapshots.
///
/// `after_snapshot` is the table state at the change commit and `before_snapshot` the state at the
/// prior commit, each indexed by `record_key_field`. For each pair: an insert ([`op::INSERT`])
/// takes its after-image from `after_snapshot` and has no before; an update ([`op::UPDATE`]) takes
/// after from `after_snapshot` and before from `before_snapshot`; a delete ([`op::DELETE`]) takes
/// only the before-image. The output columns are `op`, `ts_ms` (the commit time), and
/// `before`/`after` structs of the reconciled record schema. This reconstruction backs the
/// `cdc_op_key` and `cdc_data_before` supplemental logging modes.
pub fn build_change_records(
    ops: &[String],
    keys: &[String],
    after_snapshot: Option<&RecordBatch>,
    before_snapshot: Option<&RecordBatch>,
    record_key_field: &str,
    commit_time: &str,
) -> Result<RecordBatch> {
    if ops.len() != keys.len() {
        return Err(CoreError::Schema(format!(
            "CDC op count ({}) does not match record key count ({})",
            ops.len(),
            keys.len()
        )));
    }
    let record_schema = record_schema(after_snapshot, before_snapshot);

    let after_keys: Vec<Option<&str>> = ops
        .iter()
        .zip(keys)
        .map(|(op, k)| {
            if op == op::DELETE {
                None
            } else {
                Some(k.as_str())
            }
        })
        .collect();
    let before_keys: Vec<Option<&str>> = ops
        .iter()
        .zip(keys)
        .map(|(op, k)| {
            if op == op::INSERT {
                None
            } else {
                Some(k.as_str())
            }
        })
        .collect();

    let after = gather_records_by_key(
        after_snapshot,
        &record_schema,
        &after_keys,
        record_key_field,
    )?;
    let before = gather_records_by_key(
        before_snapshot,
        &record_schema,
        &before_keys,
        record_key_field,
    )?;

    let op_array = StringArray::from(ops.to_vec());
    let ts_array = StringArray::from(vec![commit_time.to_string(); ops.len()]);
    let struct_type = DataType::Struct(record_schema.fields().clone());
    let schema = SchemaRef::from(Schema::new(vec![
        Field::new(CDC_OP_FIELD, DataType::Utf8, false),
        Field::new("ts_ms", DataType::Utf8, false),
        Field::new("before", struct_type.clone(), true),
        Field::new("after", struct_type, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(op_array), Arc::new(ts_array), before, after],
    )
    .map_err(CoreError::ArrowError)
}

/// Extract the `(op, record_key)` columns from a CDC file's records.
pub fn extract_ops_and_keys(cdc_records: &RecordBatch) -> Result<(Vec<String>, Vec<String>)> {
    let string_col = |name: &str| -> Result<Vec<String>> {
        let array = cdc_records
            .column_by_name(name)
            .ok_or_else(|| {
                CoreError::Schema(format!("CDC records are missing the '{name}' column"))
            })?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| CoreError::Schema(format!("CDC '{name}' column is not a string")))?;
        Ok((0..array.len())
            .map(|i| {
                if array.is_valid(i) {
                    array.value(i).to_string()
                } else {
                    String::new()
                }
            })
            .collect())
    };
    Ok((string_col(CDC_OP_FIELD)?, string_col(CDC_RECORD_KEY_FIELD)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::table::HudiTableConfig::{CdcEnabled, CdcSupplementalLoggingMode};
    use arrow_array::Int64Array;

    #[test]
    fn test_is_cdc_enabled_defaults_false() {
        assert!(!is_cdc_enabled(&HudiConfigs::empty()));
        let configs = HudiConfigs::new([(CdcEnabled, "true".to_string())]);
        assert!(is_cdc_enabled(&configs));
    }

    #[test]
    fn test_supplemental_logging_mode_default_and_explicit() {
        assert_eq!(
            supplemental_logging_mode(&HudiConfigs::empty()).unwrap(),
            CdcSupplementalLoggingModeValue::DataBeforeAfter
        );
        let configs = HudiConfigs::new([(CdcSupplementalLoggingMode, "cdc_op_key".to_string())]);
        assert_eq!(
            supplemental_logging_mode(&configs).unwrap(),
            CdcSupplementalLoggingModeValue::OpKeyOnly
        );
    }

    fn commit_metadata_with_cdc(entries: &[(&str, &[&str])]) -> Map<String, Value> {
        // entries: (partition, [cdc file paths])
        let mut partition_to_write_stats = Map::new();
        for (partition, paths) in entries {
            let cdc_stats: Map<String, Value> = paths
                .iter()
                .map(|p| (p.to_string(), Value::from(1)))
                .collect();
            let stat = serde_json::json!({ "fileId": "f0", "cdcStats": cdc_stats });
            partition_to_write_stats.insert(partition.to_string(), Value::Array(vec![stat]));
        }
        let mut metadata = Map::new();
        metadata.insert(
            "partitionToWriteStats".to_string(),
            Value::Object(partition_to_write_stats),
        );
        metadata
    }

    #[test]
    fn test_cdc_file_paths_from_commit_metadata_collects_and_dedups() {
        let metadata = commit_metadata_with_cdc(&[
            (
                "2024/01/01",
                &[".cdc/f0_001.parquet", ".cdc/f0_001.parquet"],
            ),
            ("2024/01/02", &[".cdc/f1_002.parquet"]),
        ]);
        let mut paths = cdc_file_paths_from_commit_metadata(&metadata);
        paths.sort();
        assert_eq!(paths, vec![".cdc/f0_001.parquet", ".cdc/f1_002.parquet"]);
    }

    #[test]
    fn test_cdc_file_paths_supports_cdc_path_field() {
        let stat = serde_json::json!({ "fileId": "f0", "cdcPath": "2024/01/01/.cdc/f0.parquet" });
        let mut metadata = Map::new();
        metadata.insert(
            "partitionToWriteStats".to_string(),
            serde_json::json!({ "2024/01/01": [stat] }),
        );
        let paths = cdc_file_paths_from_commit_metadata(&metadata);
        assert_eq!(paths, vec!["2024/01/01/.cdc/f0.parquet"]);
    }

    fn snapshot(keys: &[&str], values: &[i64]) -> RecordBatch {
        let schema = SchemaRef::from(Schema::new(vec![
            Field::new("_hoodie_record_key", DataType::Utf8, false),
            Field::new("value", DataType::Int64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(keys.to_vec())) as ArrayRef,
                Arc::new(Int64Array::from(values.to_vec())) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn struct_value_col(batch: &RecordBatch, name: &str) -> Int64Array {
        let s = batch
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        s.column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .clone()
    }

    #[test]
    fn test_build_change_records_reconstructs_before_and_after() {
        // After commit T: A=10 (updated), B=20 (inserted). Before (T_prev): A=1, C=3.
        let after = snapshot(&["A", "B"], &[10, 20]);
        let before = snapshot(&["A", "C"], &[1, 3]);

        let ops = vec![
            op::UPDATE.to_string(),
            op::INSERT.to_string(),
            op::DELETE.to_string(),
        ];
        let keys = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let out = build_change_records(
            &ops,
            &keys,
            Some(&after),
            Some(&before),
            "_hoodie_record_key",
            "20240101000000",
        )
        .unwrap();

        assert_eq!(out.num_rows(), 3);
        let op_col = out
            .column_by_name("op")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(op_col.value(0), "u");
        assert_eq!(op_col.value(2), "d");

        let after_vals = struct_value_col(&out, "after");
        // update A -> after 10, insert B -> after 20, delete C -> after null
        assert_eq!(after_vals.value(0), 10);
        assert_eq!(after_vals.value(1), 20);
        assert!(out.column_by_name("after").unwrap().is_null(2));

        let before_vals = struct_value_col(&out, "before");
        // update A -> before 1, insert B -> before null, delete C -> before 3
        assert_eq!(before_vals.value(0), 1);
        assert!(out.column_by_name("before").unwrap().is_null(1));
        assert_eq!(before_vals.value(2), 3);
    }

    #[test]
    fn test_build_change_records_handles_missing_before_snapshot() {
        // First commit (no prior state): inserts only, before all null.
        let after = snapshot(&["A", "B"], &[1, 2]);
        let ops = vec![op::INSERT.to_string(), op::INSERT.to_string()];
        let keys = vec!["A".to_string(), "B".to_string()];
        let out = build_change_records(&ops, &keys, Some(&after), None, "_hoodie_record_key", "t0")
            .unwrap();
        assert_eq!(out.num_rows(), 2);
        let before = out.column_by_name("before").unwrap();
        assert_eq!(before.null_count(), 2);
        let after_vals = struct_value_col(&out, "after");
        assert_eq!(after_vals.value(0), 1);
        assert_eq!(after_vals.value(1), 2);
    }

    #[test]
    fn test_extract_ops_and_keys() {
        let schema = SchemaRef::from(Schema::new(vec![
            Field::new("op", DataType::Utf8, false),
            Field::new("record_key", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["i", "d"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["A", "B"])) as ArrayRef,
            ],
        )
        .unwrap();
        let (ops, keys) = extract_ops_and_keys(&batch).unwrap();
        assert_eq!(ops, vec!["i", "d"]);
        assert_eq!(keys, vec!["A", "B"]);
    }

    #[test]
    fn test_cdc_file_paths_empty_when_no_cdc() {
        let stat = serde_json::json!({ "fileId": "f0", "path": "2024/01/01/f0.parquet" });
        let mut metadata = Map::new();
        metadata.insert(
            "partitionToWriteStats".to_string(),
            serde_json::json!({ "2024/01/01": [stat] }),
        );
        assert!(cdc_file_paths_from_commit_metadata(&metadata).is_empty());
        // Missing partitionToWriteStats yields no paths.
        assert!(cdc_file_paths_from_commit_metadata(&Map::new()).is_empty());
    }
}
