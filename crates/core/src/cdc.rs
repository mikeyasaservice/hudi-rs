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
use serde_json::{Map, Value};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::table::HudiTableConfig::{CdcEnabled, CdcSupplementalLoggingMode};

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
