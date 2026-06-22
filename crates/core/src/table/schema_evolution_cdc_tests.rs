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

//! End-to-end read-path tests for schema evolution and CDC, exercised against COW tables that the
//! [`super::fixture::TableGen`] generator writes to disk (rather than pre-baked fixtures).

use super::fixture::{TableGen, WriteFile};
use crate::config::read_options::ReadOptions;
use crate::table::Table;
use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringArray, StructArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use std::collections::HashMap;
use std::sync::Arc;

const SKIP_VALIDATION: &str = "hoodie.internal.skip.config.validation";

fn field_with_id(name: &str, dt: DataType, id: i32) -> Field {
    Field::new(name, dt, true).with_metadata(HashMap::from([(
        "PARQUET:field_id".to_string(),
        id.to_string(),
    )]))
}

async fn read_table_with_options<I>(base: &str, opts: I) -> Vec<RecordBatch>
where
    I: IntoIterator<Item = (&'static str, &'static str)>,
{
    let mut options: Vec<(&str, &str)> = vec![(SKIP_VALIDATION, "true")];
    options.extend(opts);
    let table = Table::new_with_options(base, options).await.unwrap();
    table.read(&ReadOptions::new()).await.unwrap()
}

/// Collect every row across batches into `(id -> (column name -> stringified value))`, so tests
/// can assert on values regardless of slice order or null-filled column layout.
fn rows_by_id(batches: &[RecordBatch]) -> HashMap<i32, HashMap<String, Option<String>>> {
    let mut out = HashMap::new();
    for batch in batches {
        let schema = batch.schema();
        let id_idx = schema.index_of("id").unwrap();
        let ids = batch
            .column(id_idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let mut cols = HashMap::new();
            for (c, field) in schema.fields().iter().enumerate() {
                let col = batch.column(c);
                let value = if col.is_null(row) {
                    None
                } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                    Some(a.value(row).to_string())
                } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
                    Some(a.value(row).to_string())
                } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                    Some(a.value(row).to_string())
                } else {
                    Some("<other>".to_string())
                };
                cols.insert(field.name().clone(), value);
            }
            out.insert(ids.value(row), cols);
        }
    }
    out
}

#[tokio::test]
async fn read_reconciles_added_column_across_file_slices() {
    // Commit 1 writes a file group with [id, amount]; commit 2 a second group that also has [city].
    let s1 = SchemaRef::from(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, true),
    ]));
    let b1 = RecordBatch::try_new(
        s1,
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![100])),
        ],
    )
    .unwrap();

    let s2 = SchemaRef::from(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, true),
        Field::new("city", DataType::Utf8, true),
    ]));
    let b2 = RecordBatch::try_new(
        s2,
        vec![
            Arc::new(Int32Array::from(vec![2])),
            Arc::new(Int64Array::from(vec![200])),
            Arc::new(StringArray::from(vec!["paris"])),
        ],
    )
    .unwrap();

    let tg = TableGen::new("evo_add_col");
    tg.init();
    tg.commit(
        "20240101000001",
        "INSERT",
        &[WriteFile {
            partition: "",
            file_id: "fg1",
            batch: &b1,
            cdc_files: vec![],
        }],
    );
    tg.commit(
        "20240101000002",
        "INSERT",
        &[WriteFile {
            partition: "",
            file_id: "fg2",
            batch: &b2,
            cdc_files: vec![],
        }],
    );

    let batches = read_table_with_options(tg.base_path(), []).await;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2);
    // All batches share one reconciled schema, with `city` present.
    for b in &batches {
        assert!(
            b.schema().index_of("city").is_ok(),
            "city must be reconciled in"
        );
    }
    let rows = rows_by_id(&batches);
    assert_eq!(rows[&1]["amount"], Some("100".to_string()));
    assert_eq!(rows[&1]["city"], None); // null-filled for the older file
    assert_eq!(rows[&2]["city"], Some("paris".to_string()));
}

#[tokio::test]
async fn read_tracks_renamed_column_by_field_id() {
    // Same field id (7) for the renamed column; commit 1 oldest so newest name wins.
    let s_old = SchemaRef::from(Schema::new(vec![
        field_with_id("id", DataType::Int32, 1),
        field_with_id("amount", DataType::Int64, 7),
    ]));
    let b_old = RecordBatch::try_new(
        s_old,
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int64Array::from(vec![100])),
        ],
    )
    .unwrap();

    let s_new = SchemaRef::from(Schema::new(vec![
        field_with_id("id", DataType::Int32, 1),
        field_with_id("total_amount", DataType::Int64, 7),
    ]));
    let b_new = RecordBatch::try_new(
        s_new,
        vec![
            Arc::new(Int32Array::from(vec![2])),
            Arc::new(Int64Array::from(vec![200])),
        ],
    )
    .unwrap();

    let tg = TableGen::new("evo_rename");
    tg.init();
    tg.commit(
        "20240101000001",
        "INSERT",
        &[WriteFile {
            partition: "",
            file_id: "fg1",
            batch: &b_old,
            cdc_files: vec![],
        }],
    );
    tg.commit(
        "20240101000002",
        "INSERT",
        &[WriteFile {
            partition: "",
            file_id: "fg2",
            batch: &b_new,
            cdc_files: vec![],
        }],
    );

    let batches = read_table_with_options(tg.base_path(), []).await;
    // The renamed column is one reconciled column keyed by field id, not two name-based columns.
    for b in &batches {
        assert!(
            b.schema().index_of("amount").is_err(),
            "old name should not survive a rename tracked by field id"
        );
        assert!(b.schema().index_of("total_amount").is_ok());
    }
    let rows = rows_by_id(&batches);
    // The row written under the OLD name maps onto the new name by id, value preserved.
    assert_eq!(rows[&1]["total_amount"], Some("100".to_string()));
    assert_eq!(rows[&2]["total_amount"], Some("200".to_string()));
}

const CDC_ENABLED: &str = "hoodie.table.cdc.enabled";
const CDC_MODE: &str = "hoodie.table.cdc.supplemental.logging.mode";

/// A base data record carrying the `_hoodie_record_key` meta field used by CDC reconstruction.
fn keyed_record(record_key: &str, id: i32, value: i64) -> RecordBatch {
    let schema = SchemaRef::from(Schema::new(vec![
        Field::new("_hoodie_record_key", DataType::Utf8, false),
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![record_key])),
            Arc::new(Int32Array::from(vec![id])),
            Arc::new(Int64Array::from(vec![value])),
        ],
    )
    .unwrap()
}

fn struct_i64(batch: &RecordBatch, struct_col: &str, field: &str, row: usize) -> Option<i64> {
    let col = batch.column_by_name(struct_col).unwrap();
    if col.is_null(row) {
        return None;
    }
    let s = col.as_any().downcast_ref::<StructArray>().unwrap();
    Some(
        s.column_by_name(field)
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(row),
    )
}

#[tokio::test]
async fn read_cdc_data_before_after_reads_cdc_files_directly() {
    // The CDC file already holds self-contained change records; read_cdc returns them verbatim.
    let cdc = RecordBatch::try_new(
        SchemaRef::from(Schema::new(vec![
            Field::new("op", DataType::Utf8, false),
            Field::new("record_key", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["i", "u"])),
            Arc::new(StringArray::from(vec!["k1", "k2"])),
        ],
    )
    .unwrap();

    let tg = TableGen::new("cdc_dba")
        .with_prop(CDC_ENABLED, "true")
        .with_prop(CDC_MODE, "cdc_data_before_after");
    tg.init();
    tg.write_data_file(".hoodie/.cdc/cdc1.parquet", &cdc);
    tg.commit(
        "20240101000001",
        "INSERT",
        &[WriteFile {
            partition: "",
            file_id: "fg1",
            batch: &keyed_record("k1", 1, 10),
            cdc_files: vec![(".hoodie/.cdc/cdc1.parquet".to_string(), 2)],
        }],
    );

    let table = Table::new_with_options(tg.base_path(), [(SKIP_VALIDATION, "true")])
        .await
        .unwrap();
    let changes = table.read_cdc(&ReadOptions::new()).await.unwrap();
    let total: usize = changes.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2);
    let ops: Vec<String> = changes
        .iter()
        .flat_map(|b| {
            let a = b
                .column_by_name("op")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            (0..a.len())
                .map(|i| a.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(ops, vec!["i", "u"]);
}

#[tokio::test]
async fn read_cdc_op_key_reconstructs_before_and_after() {
    // Commit 1 inserts k1=10; commit 2 updates it to 20 and logs an op-key CDC record.
    let op_key_cdc = RecordBatch::try_new(
        SchemaRef::from(Schema::new(vec![
            Field::new("op", DataType::Utf8, false),
            Field::new("record_key", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["u"])),
            Arc::new(StringArray::from(vec!["k1"])),
        ],
    )
    .unwrap();

    let tg = TableGen::new("cdc_op_key")
        .with_prop(CDC_ENABLED, "true")
        .with_prop(CDC_MODE, "cdc_op_key");
    tg.init();
    tg.commit(
        "20240101000001",
        "INSERT",
        &[WriteFile {
            partition: "",
            file_id: "fg1",
            batch: &keyed_record("k1", 1, 10),
            cdc_files: vec![],
        }],
    );
    tg.write_data_file(".hoodie/.cdc/cdc2.parquet", &op_key_cdc);
    tg.commit(
        "20240101000002",
        "UPSERT",
        &[WriteFile {
            partition: "",
            file_id: "fg1",
            batch: &keyed_record("k1", 1, 20),
            cdc_files: vec![(".hoodie/.cdc/cdc2.parquet".to_string(), 1)],
        }],
    );

    let table = Table::new_with_options(tg.base_path(), [(SKIP_VALIDATION, "true")])
        .await
        .unwrap();
    let changes = table.read_cdc(&ReadOptions::new()).await.unwrap();
    let total: usize = changes.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1, "one change record for the single update");
    let batch = &changes[0];

    let op = batch
        .column_by_name("op")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(op.value(0), "u");
    // Reconstructed from the snapshots at commit 2 (after) and commit 1 (before).
    assert_eq!(struct_i64(batch, "after", "value", 0), Some(20));
    assert_eq!(struct_i64(batch, "before", "value", 0), Some(10));
}
