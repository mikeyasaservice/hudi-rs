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

//! Test-only generator that writes a minimal but valid Copy-on-Write Hudi table to disk so the
//! real read path can be exercised end-to-end without a pre-baked fixture.
//!
//! It writes `.hoodie/hoodie.properties`, one completed `.commit` instant per [`Self::commit`]
//! call (with `partitionToWriteStats` and an optional `cdcStats` map), `.hoodie_partition_metadata`
//! markers, and the Parquet data files. Table schema is derived from the Parquet footers, so a
//! commit's batch schema (including `PARQUET:field_id` metadata) is what the reader sees.

use arrow_array::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// A single file written in a commit.
pub(crate) struct WriteFile<'a> {
    pub partition: &'a str,
    pub file_id: &'a str,
    pub batch: &'a RecordBatch,
    /// CDC files for this write stat: `(relative_path, record_count)`.
    pub cdc_files: Vec<(String, i64)>,
}

pub(crate) struct TableGen {
    _dir: TempDir,
    base: PathBuf,
    props: Vec<(String, String)>,
}

impl TableGen {
    /// Create a generator for a nonpartitioned COW table named `name` in a fresh temp dir.
    pub fn new(name: &str) -> Self {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path().to_path_buf();
        let props = vec![
            ("hoodie.table.type".to_string(), "COPY_ON_WRITE".to_string()),
            ("hoodie.table.version".to_string(), "6".to_string()),
            (
                "hoodie.timeline.layout.version".to_string(),
                "1".to_string(),
            ),
            ("hoodie.table.name".to_string(), name.to_string()),
            (
                "hoodie.table.recordkey.fields".to_string(),
                "id".to_string(),
            ),
            (
                "hoodie.archivelog.folder".to_string(),
                "archived".to_string(),
            ),
            (
                "hoodie.datasource.write.hive_style_partitioning".to_string(),
                "false".to_string(),
            ),
            (
                "hoodie.table.keygenerator.class".to_string(),
                "org.apache.hudi.keygen.NonpartitionedKeyGenerator".to_string(),
            ),
        ];
        Self {
            _dir: dir,
            base,
            props,
        }
    }

    pub fn with_prop(mut self, key: &str, value: &str) -> Self {
        // Replace if present so callers can override defaults.
        self.props.retain(|(k, _)| k != key);
        self.props.push((key.to_string(), value.to_string()));
        self
    }

    pub fn base_path(&self) -> &str {
        self.base.to_str().expect("utf8 base path")
    }

    /// Write `hoodie.properties` and the `.hoodie` directory.
    pub fn init(&self) {
        let hoodie = self.base.join(".hoodie");
        fs::create_dir_all(&hoodie).expect("mkdir .hoodie");
        let mut body = String::from("#hudi-rs test fixture\n");
        for (k, v) in &self.props {
            body.push_str(&format!("{k}={v}\n"));
        }
        fs::write(hoodie.join("hoodie.properties"), body).expect("write properties");
    }

    fn write_partition_metadata(&self, partition: &str, instant: &str) {
        let dir = if partition.is_empty() {
            self.base.clone()
        } else {
            self.base.join(partition)
        };
        fs::create_dir_all(&dir).expect("mkdir partition");
        let body = format!("#partition metadata\ncommitTime={instant}\npartitionDepth=0\n");
        fs::write(dir.join(".hoodie_partition_metadata"), body).expect("write partition meta");
    }

    /// Write one completed commit instant with the given files.
    pub fn commit(&self, instant: &str, op_type: &str, files: &[WriteFile<'_>]) {
        let mut write_stats: std::collections::HashMap<String, Vec<serde_json::Value>> =
            std::collections::HashMap::new();

        for (i, f) in files.iter().enumerate() {
            self.write_partition_metadata(f.partition, instant);
            let write_token = format!("0-{i}-{i}");
            let file_name = format!("{}_{write_token}_{instant}.parquet", f.file_id);
            let rel_path = if f.partition.is_empty() {
                file_name.clone()
            } else {
                format!("{}/{file_name}", f.partition)
            };
            let abs_path = self.base.join(&rel_path);
            if let Some(parent) = abs_path.parent() {
                fs::create_dir_all(parent).expect("mkdir for parquet");
            }
            self.write_parquet(&abs_path, f.batch);
            let size = fs::metadata(&abs_path).map(|m| m.len()).unwrap_or(0) as i64;

            let cdc_stats: serde_json::Value = if f.cdc_files.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::Value::Object(
                    f.cdc_files
                        .iter()
                        .map(|(p, n)| (p.clone(), serde_json::Value::from(*n)))
                        .collect(),
                )
            };

            let stat = serde_json::json!({
                "fileId": f.file_id,
                "path": rel_path,
                "partitionPath": f.partition,
                "numWrites": f.batch.num_rows(),
                "numInserts": f.batch.num_rows(),
                "numUpdateWrites": 0,
                "numDeletes": 0,
                "totalWriteBytes": size,
                "fileSizeInBytes": size,
                "cdcStats": cdc_stats,
            });
            write_stats
                .entry(f.partition.to_string())
                .or_default()
                .push(stat);
        }

        let partition_to_write_stats: serde_json::Map<String, serde_json::Value> = write_stats
            .into_iter()
            .map(|(k, v)| (k, serde_json::Value::Array(v)))
            .collect();
        let metadata = serde_json::json!({
            "partitionToWriteStats": partition_to_write_stats,
            "operationType": op_type,
            "compacted": false,
        });

        let hoodie = self.base.join(".hoodie");
        fs::write(
            hoodie.join(format!("{instant}.commit")),
            serde_json::to_vec_pretty(&metadata).expect("serialize commit"),
        )
        .expect("write commit");
        // Mirror the real triplet; the reader only consumes the completed `.commit`.
        fs::write(hoodie.join(format!("{instant}.commit.requested")), b"").expect("requested");
        fs::write(hoodie.join(format!("{instant}.inflight")), b"").expect("inflight");
    }

    /// Write a standalone Parquet file (e.g. a CDC data file) at `rel_path` under the table base.
    pub fn write_data_file(&self, rel_path: &str, batch: &RecordBatch) {
        let abs = self.base.join(rel_path);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent).expect("mkdir for data file");
        }
        self.write_parquet(&abs, batch);
    }

    fn write_parquet(&self, path: &std::path::Path, batch: &RecordBatch) {
        let file = fs::File::create(path).expect("create parquet");
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).expect("arrow writer");
        writer.write(batch).expect("write batch");
        writer.close().expect("close writer");
    }
}
