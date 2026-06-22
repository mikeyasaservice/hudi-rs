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

use crate::Result;
use crate::error::CoreError;
use arrow::array::ArrayRef;
use arrow::array::RecordBatch;
use arrow::array::StringArray;
use arrow::compute::cast;
use arrow_array::{Array, UInt32Array, new_null_array};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use std::collections::HashMap;

pub trait ColumnAsArray {
    fn get_array(&self, column_name: &str) -> Result<ArrayRef>;

    fn get_string_array(&self, column_name: &str) -> Result<StringArray>;
}

impl ColumnAsArray for RecordBatch {
    fn get_array(&self, column_name: &str) -> Result<ArrayRef> {
        let index = self.schema().index_of(column_name)?;
        let array = self.column(index);
        Ok(array.clone())
    }

    fn get_string_array(&self, column_name: &str) -> Result<StringArray> {
        let array = self.get_array(column_name)?;
        let array = array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                ArrowError::CastError(format!(
                    "Column {column_name} cannot be cast to StringArray."
                ))
            })?;
        Ok(array.clone())
    }
}

pub fn lexsort_to_indices(arrays: &[ArrayRef], desc: bool) -> UInt32Array {
    let fields = arrays
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields).unwrap();
    let rows = converter.convert_columns(arrays).unwrap();
    let mut sort: Vec<_> = rows.iter().enumerate().collect();
    if desc {
        sort.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));
    } else {
        sort.sort_unstable_by(|(_, a), (_, b)| a.cmp(b));
    }
    UInt32Array::from_iter_values(sort.iter().map(|(i, _)| *i as u32))
}

pub fn create_row_converter<I, S>(schema: SchemaRef, column_names: I) -> Result<RowConverter>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let sort_fields: Result<Vec<_>> = column_names
        .into_iter()
        .map(|col| {
            let (_, field) = schema
                .column_with_name(col.as_ref())
                .ok_or_else(|| CoreError::Schema(format!("Column {} not found", col.as_ref())))?;
            Ok(SortField::new(field.data_type().clone()))
        })
        .collect();
    RowConverter::new(sort_fields?).map_err(CoreError::ArrowError)
}

pub fn get_column_arrays<I, S>(batch: &RecordBatch, column_names: I) -> Result<Vec<ArrayRef>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    column_names
        .into_iter()
        .map(|col| {
            batch
                .column_by_name(col.as_ref())
                .cloned()
                .ok_or_else(|| CoreError::Schema(format!("Column {} not found", col.as_ref())))
        })
        .collect()
}

/// Project a [`RecordBatch`] to a subset of columns by name. Returns the batch
/// unchanged when `projection` is `None`. Errors when any name is not present in
/// the batch's schema.
pub fn project_batch_by_names(
    batch: RecordBatch,
    projection: Option<&[String]>,
) -> Result<RecordBatch> {
    let Some(cols) = projection else {
        return Ok(batch);
    };
    let indices: Vec<usize> = cols
        .iter()
        .map(|name| {
            batch
                .schema()
                .index_of(name)
                .map_err(|e| CoreError::Schema(format!("Projection column not found: {e:?}")))
        })
        .collect::<Result<_>>()?;
    batch.project(&indices).map_err(CoreError::ArrowError)
}

fn is_integer_type(d: &DataType) -> bool {
    matches!(
        d,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn is_float_type(d: &DataType) -> bool {
    matches!(d, DataType::Float16 | DataType::Float32 | DataType::Float64)
}

/// Choose a common type for a column that appears with two different types across schemas.
///
/// Numeric occurrences are widened (to `Int64` when both are integers, `Float64` when any is
/// floating). For any other mismatch the newer type (`b`) is chosen; an incompatible change
/// then surfaces as a cast error in [`adapt_batch_to_schema`] rather than silently mis-reading.
fn widen_types(a: &DataType, b: &DataType) -> DataType {
    if a == b {
        return a.clone();
    }
    let a_num = is_integer_type(a) || is_float_type(a);
    let b_num = is_integer_type(b) || is_float_type(b);
    if a_num && b_num {
        if is_float_type(a) || is_float_type(b) {
            DataType::Float64
        } else {
            DataType::Int64
        }
    } else {
        b.clone()
    }
}

/// Arrow field-metadata key under which the Parquet reader records a column's Parquet field id.
/// Hudi's schema-on-read tables write stable field ids, which lets reconciliation track a column
/// across a rename (the id stays the same while the name changes).
const PARQUET_FIELD_ID_KEY: &str = "PARQUET:field_id";

fn field_id(field: &Field) -> Option<String> {
    field.metadata().get(PARQUET_FIELD_ID_KEY).cloned()
}

/// Reconciliation key for a column: its Parquet field id when present (stable across renames),
/// otherwise its name.
#[derive(Hash, PartialEq, Eq, Clone)]
enum ColKey {
    Id(String),
    Name(String),
}

fn col_key(field: &Field) -> ColKey {
    match field_id(field) {
        Some(id) => ColKey::Id(id),
        None => ColKey::Name(field.name().clone()),
    }
}

/// Reconcile a set of (possibly schema-evolved) Arrow schemas into a single target schema for
/// schema-on-read.
///
/// Columns are matched by Parquet field id when present (so a renamed column is tracked across
/// schemas), otherwise by name, and unioned in first-appearance order. The target column name is
/// taken from the last schema in which the column appears (newest-wins, so a rename resolves to
/// the current name when schemas are supplied oldest-first). A field's type is widened across its
/// occurrences (see [`widen_types`]); it is nullable if absent from any input schema or marked
/// nullable in any of them. When every input schema is identical the first schema is returned
/// unchanged, keeping the common no-evolution path a no-op.
pub fn reconcile_schemas(schemas: &[SchemaRef]) -> SchemaRef {
    let Some(first) = schemas.first() else {
        return SchemaRef::from(Schema::empty());
    };
    if schemas.iter().all(|s| s.fields() == first.fields()) {
        return first.clone();
    }

    struct Acc {
        name: String,
        data_type: DataType,
        present: usize,
        nullable: bool,
        metadata: HashMap<String, String>,
    }

    let total = schemas.len();
    let mut order: Vec<ColKey> = Vec::new();
    let mut acc: HashMap<ColKey, Acc> = HashMap::new();
    for schema in schemas {
        for field in schema.fields() {
            let key = col_key(field);
            if let Some(existing) = acc.get_mut(&key) {
                existing.name = field.name().clone(); // newest-wins (handles renames by id)
                existing.data_type = widen_types(&existing.data_type, field.data_type());
                existing.present += 1;
                existing.nullable = existing.nullable || field.is_nullable();
                for (k, v) in field.metadata() {
                    existing.metadata.insert(k.clone(), v.clone());
                }
            } else {
                order.push(key.clone());
                acc.insert(
                    key,
                    Acc {
                        name: field.name().clone(),
                        data_type: field.data_type().clone(),
                        present: 1,
                        nullable: field.is_nullable(),
                        metadata: field.metadata().clone(),
                    },
                );
            }
        }
    }

    let fields: Vec<Field> = order
        .iter()
        .filter_map(|key| {
            acc.get(key).map(|a| {
                let nullable = a.nullable || a.present < total;
                let field = Field::new(&a.name, a.data_type.clone(), nullable);
                if a.metadata.is_empty() {
                    field
                } else {
                    field.with_metadata(a.metadata.clone())
                }
            })
        })
        .collect();
    SchemaRef::from(Schema::new(fields))
}

/// Adapt a [`RecordBatch`] to `target` for schema-on-read: select the target's columns (matching
/// by Parquet field id when present, else by name, so renamed columns still line up; reordering
/// as needed), cast columns whose type differs, and fill columns missing from the batch with
/// nulls. Columns of `batch` absent from `target` are dropped. Returns the batch unchanged when
/// its schema already matches `target`.
pub fn adapt_batch_to_schema(batch: &RecordBatch, target: &SchemaRef) -> Result<RecordBatch> {
    if batch.schema().fields() == target.fields() {
        return Ok(batch.clone());
    }
    let num_rows = batch.num_rows();
    let batch_schema = batch.schema();
    let mut by_id: HashMap<String, usize> = HashMap::new();
    let mut by_name: HashMap<String, usize> = HashMap::new();
    for (i, field) in batch_schema.fields().iter().enumerate() {
        if let Some(id) = field_id(field) {
            by_id.insert(id, i);
        }
        by_name.insert(field.name().clone(), i);
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(target.fields().len());
    for field in target.fields() {
        let idx = field_id(field)
            .and_then(|id| by_id.get(&id).copied())
            .or_else(|| by_name.get(field.name()).copied());
        match idx {
            Some(i) => {
                let col = batch.column(i);
                if col.data_type() == field.data_type() {
                    columns.push(col.clone());
                } else {
                    columns.push(cast(col, field.data_type()).map_err(CoreError::ArrowError)?);
                }
            }
            None => columns.push(new_null_array(field.data_type(), num_rows)),
        }
    }
    RecordBatch::try_new(target.clone(), columns).map_err(CoreError::ArrowError)
}

/// Reconcile a set of batches (read from possibly schema-evolved files) to one common schema, so
/// a single result set has a uniform schema. Returns the batches unchanged when there are fewer
/// than two or they already share a schema; otherwise each is adapted to the reconciled union.
pub fn reconcile_batches(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    if batches.len() < 2 {
        return Ok(batches);
    }
    let schemas: Vec<SchemaRef> = batches.iter().map(|b| b.schema()).collect();
    let target = reconcile_schemas(&schemas);
    if batches
        .iter()
        .all(|b| b.schema().fields() == target.fields())
    {
        return Ok(batches);
    }
    batches
        .iter()
        .map(|b| adapt_batch_to_schema(b, &target))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, Int64Array, StringArray};
    use arrow_array::Float64Array;
    use std::sync::Arc;

    fn schema(fields: &[(&str, DataType, bool)]) -> SchemaRef {
        SchemaRef::from(Schema::new(
            fields
                .iter()
                .map(|(n, t, nullable)| Field::new(*n, t.clone(), *nullable))
                .collect::<Vec<_>>(),
        ))
    }

    fn field_with_id(name: &str, dt: DataType, nullable: bool, id: i32) -> Field {
        Field::new(name, dt, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_KEY.to_string(),
            id.to_string(),
        )]))
    }

    #[test]
    fn test_reconcile_tracks_renamed_column_by_field_id() {
        // Same field id (7), renamed `amount` -> `total_amount`, supplied oldest-first.
        let old = SchemaRef::from(Schema::new(vec![
            field_with_id("id", DataType::Int32, false, 1),
            field_with_id("amount", DataType::Int64, false, 7),
        ]));
        let new = SchemaRef::from(Schema::new(vec![
            field_with_id("id", DataType::Int32, false, 1),
            field_with_id("total_amount", DataType::Int64, false, 7),
        ]));

        let target = reconcile_schemas(&[old.clone(), new]);
        // Union keyed by id, newest name wins -> two columns, the renamed one current.
        let names: Vec<&str> = target.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, ["id", "total_amount"]);

        // A batch written with the old name maps onto the new name by field id, not null-filled.
        let old_batch = RecordBatch::try_new(
            old,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                Arc::new(Int64Array::from(vec![100, 200])) as ArrayRef,
            ],
        )
        .unwrap();
        let adapted = adapt_batch_to_schema(&old_batch, &target).unwrap();
        let total = adapted
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(total.values(), &[100, 200]);
        assert_eq!(adapted.column(1).null_count(), 0);
    }

    #[test]
    fn test_reconcile_schemas_identical_is_noop() {
        let s = schema(&[
            ("id", DataType::Int32, false),
            ("name", DataType::Utf8, true),
        ]);
        let out = reconcile_schemas(&[s.clone(), s.clone()]);
        // Same instance preserved (order + nullability unchanged).
        assert_eq!(out.fields(), s.fields());
        assert!(!out.field(0).is_nullable());
    }

    #[test]
    fn test_reconcile_schemas_added_column_is_nullable_and_appended() {
        let base = schema(&[("id", DataType::Int32, false)]);
        let evolved = schema(&[
            ("id", DataType::Int32, false),
            ("city", DataType::Utf8, false),
        ]);
        let out = reconcile_schemas(&[base, evolved]);
        assert_eq!(out.fields().len(), 2);
        assert_eq!(out.field(1).name(), "city");
        // `city` is absent from the base schema, so it must be nullable in the target.
        assert!(out.field(1).is_nullable());
        // `id` present in both and non-nullable stays non-nullable.
        assert!(!out.field(0).is_nullable());
    }

    #[test]
    fn test_reconcile_schemas_widens_numeric_types() {
        let s32 = schema(&[("v", DataType::Int32, false)]);
        let s64 = schema(&[("v", DataType::Int64, false)]);
        let out = reconcile_schemas(&[s32, s64]);
        assert_eq!(out.field(0).data_type(), &DataType::Int64);

        let sf = schema(&[("v", DataType::Float64, false)]);
        let si = schema(&[("v", DataType::Int32, false)]);
        let out = reconcile_schemas(&[si, sf]);
        assert_eq!(out.field(0).data_type(), &DataType::Float64);
    }

    #[test]
    fn test_adapt_batch_fills_missing_casts_and_reorders() {
        // Batch has [v: Int32, id: Int32]; target wants [id: Int64, v: Int64, city: Utf8].
        let batch = RecordBatch::try_new(
            schema(&[
                ("v", DataType::Int32, false),
                ("id", DataType::Int32, false),
            ]),
            vec![
                Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef,
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            ],
        )
        .unwrap();
        let target = schema(&[
            ("id", DataType::Int64, false),
            ("v", DataType::Int64, false),
            ("city", DataType::Utf8, true),
        ]);

        let adapted = adapt_batch_to_schema(&batch, &target).unwrap();
        assert_eq!(adapted.schema().fields(), target.fields());

        let ids = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 2]);
        let vs = adapted
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(vs.values(), &[10, 20]);
        // Missing column filled with nulls.
        assert_eq!(adapted.column(2).null_count(), 2);
    }

    #[test]
    fn test_reconcile_batches_unifies_evolved_batches() {
        // One batch has [id], another adds [city]; both should end up with [id, city].
        let b1 = RecordBatch::try_new(
            schema(&[("id", DataType::Int32, false)]),
            vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
        )
        .unwrap();
        let b2 = RecordBatch::try_new(
            schema(&[
                ("id", DataType::Int32, false),
                ("city", DataType::Utf8, true),
            ]),
            vec![
                Arc::new(Int32Array::from(vec![3])) as ArrayRef,
                Arc::new(StringArray::from(vec!["paris"])) as ArrayRef,
            ],
        )
        .unwrap();

        let out = reconcile_batches(vec![b1, b2]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].schema().fields(), out[1].schema().fields());
        let out0_schema = out[0].schema();
        let names: Vec<&str> = out0_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, ["id", "city"]);
        // The first batch had no `city`, so it is null-filled.
        assert_eq!(out[0].column(1).null_count(), 2);
    }

    #[test]
    fn test_reconcile_batches_noop_for_uniform_or_single() {
        let s = schema(&[("id", DataType::Int32, false)]);
        let b = RecordBatch::try_new(
            s.clone(),
            vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef],
        )
        .unwrap();
        // Single batch: returned as-is.
        assert_eq!(reconcile_batches(vec![b.clone()]).unwrap().len(), 1);
        // Two identical-schema batches: unchanged.
        let out = reconcile_batches(vec![b.clone(), b]).unwrap();
        assert_eq!(out[0].schema().fields(), s.fields());
    }

    #[test]
    fn test_adapt_batch_identical_schema_is_noop() {
        let s = schema(&[("id", DataType::Int32, false)]);
        let batch = RecordBatch::try_new(
            s.clone(),
            vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef],
        )
        .unwrap();
        let adapted = adapt_batch_to_schema(&batch, &s).unwrap();
        assert_eq!(adapted.schema().fields(), s.fields());
    }

    #[test]
    fn test_basic_int_sort() {
        let arr = Int32Array::from(vec![3, 1, 4, 1, 5]);
        let arrays = vec![Arc::new(arr) as ArrayRef];

        // Test ascending
        let result = lexsort_to_indices(&arrays, false);
        assert_eq!(
            result.values(),
            &[1, 3, 0, 2, 4] // Indices that would sort to [1,1,3,4,5]
        );

        // Test descending
        let result = lexsort_to_indices(&arrays, true);
        assert_eq!(
            result.values(),
            &[4, 2, 0, 1, 3] // Indices that would sort to [5,4,3,1,1]
        );
    }

    #[test]
    fn test_multiple_columns() {
        let arr1 = Int32Array::from(vec![1, 1, 2, 2]);
        let arr2 = StringArray::from(vec!["b", "a", "b", "a"]);
        let arrays = vec![Arc::new(arr1) as ArrayRef, Arc::new(arr2) as ArrayRef];

        let result = lexsort_to_indices(&arrays, false);
        assert_eq!(
            result.values(),
            &[1, 0, 3, 2] // Should sort by first column then second
        );
    }

    #[test]
    fn test_edge_cases() {
        // Empty array
        assert_eq!(lexsort_to_indices(&[], false).len(), 0);

        // Array of empty array
        let arr = Int32Array::from(vec![] as Vec<i32>);
        let arrays = vec![Arc::new(arr) as ArrayRef];
        let result = lexsort_to_indices(&arrays, false);
        assert_eq!(result.len(), 0);

        // Single element
        let arr = Int32Array::from(vec![1]);
        let arrays = vec![Arc::new(arr) as ArrayRef];
        let result = lexsort_to_indices(&arrays, false);
        assert_eq!(result.values(), &[0]);

        // All equal values
        let arr = Int32Array::from(vec![5, 5, 5, 5]);
        let arrays = vec![Arc::new(arr) as ArrayRef];
        let result = lexsort_to_indices(&arrays, false);
        assert_eq!(result.values(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_different_types() {
        let int_arr = Int32Array::from(vec![1, 2, 1]);
        let str_arr = StringArray::from(vec!["a", "b", "c"]);
        let float_arr = Float64Array::from(vec![1.0, 2.0, 3.0]);

        let arrays = vec![
            Arc::new(int_arr) as ArrayRef,
            Arc::new(str_arr) as ArrayRef,
            Arc::new(float_arr) as ArrayRef,
        ];

        let result = lexsort_to_indices(&arrays, false);
        assert_eq!(result.values(), &[0, 2, 1]);
    }

    #[test]
    fn test_project_batch_by_names() {
        use arrow_schema::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef,
                Arc::new(Float64Array::from(vec![1.5, 2.5])) as ArrayRef,
            ],
        )
        .unwrap();

        // None returns input unchanged.
        let same = project_batch_by_names(batch.clone(), None).unwrap();
        assert_eq!(same.num_columns(), 3);
        assert_eq!(same.schema().field(0).name(), "a");

        // Subset projects in the requested order.
        let cols = vec!["c".to_string(), "a".to_string()];
        let projected = project_batch_by_names(batch.clone(), Some(&cols)).unwrap();
        assert_eq!(projected.num_columns(), 2);
        assert_eq!(projected.schema().field(0).name(), "c");
        assert_eq!(projected.schema().field(1).name(), "a");

        // Unknown column errors with a Schema error.
        let bad = vec!["a".to_string(), "missing".to_string()];
        let err = project_batch_by_names(batch, Some(&bad)).unwrap_err();
        assert!(matches!(err, CoreError::Schema(_)));
        assert!(err.to_string().contains("missing"));
    }
}
