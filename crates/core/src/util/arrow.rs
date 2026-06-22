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

/// Reconcile a set of (possibly schema-evolved) Arrow schemas into a single target schema for
/// schema-on-read by column name.
///
/// Fields are unioned in first-appearance order. A field's type is widened across the schemas it
/// appears in (see [`widen_types`]). A field is nullable if it is absent from any input schema or
/// marked nullable in any of them, so missing columns can be null-filled. When every input schema
/// is identical the first schema is returned unchanged, keeping the common no-evolution path a
/// no-op (preserving field order and nullability exactly).
pub fn reconcile_schemas(schemas: &[SchemaRef]) -> SchemaRef {
    let Some(first) = schemas.first() else {
        return SchemaRef::from(Schema::empty());
    };
    if schemas.iter().all(|s| s.fields() == first.fields()) {
        return first.clone();
    }

    struct Acc {
        data_type: DataType,
        present: usize,
        nullable: bool,
    }

    let total = schemas.len();
    let mut order: Vec<String> = Vec::new();
    let mut acc: HashMap<String, Acc> = HashMap::new();
    for schema in schemas {
        for field in schema.fields() {
            let name = field.name();
            if let Some(existing) = acc.get_mut(name) {
                existing.data_type = widen_types(&existing.data_type, field.data_type());
                existing.present += 1;
                existing.nullable = existing.nullable || field.is_nullable();
            } else {
                order.push(name.clone());
                acc.insert(
                    name.clone(),
                    Acc {
                        data_type: field.data_type().clone(),
                        present: 1,
                        nullable: field.is_nullable(),
                    },
                );
            }
        }
    }

    let fields: Vec<Field> = order
        .iter()
        .filter_map(|name| {
            acc.get(name).map(|a| {
                let nullable = a.nullable || a.present < total;
                Field::new(name, a.data_type.clone(), nullable)
            })
        })
        .collect();
    SchemaRef::from(Schema::new(fields))
}

/// Adapt a [`RecordBatch`] to `target` for schema-on-read: select the target's columns by name
/// (reordering as needed), cast columns whose type differs, and fill columns missing from the
/// batch with nulls. Columns of `batch` absent from `target` are dropped. Returns the batch
/// unchanged when its schema already matches `target`.
pub fn adapt_batch_to_schema(batch: &RecordBatch, target: &SchemaRef) -> Result<RecordBatch> {
    if batch.schema().fields() == target.fields() {
        return Ok(batch.clone());
    }
    let num_rows = batch.num_rows();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(target.fields().len());
    for field in target.fields() {
        match batch.schema().index_of(field.name()) {
            Ok(idx) => {
                let col = batch.column(idx);
                if col.data_type() == field.data_type() {
                    columns.push(col.clone());
                } else {
                    columns.push(cast(col, field.data_type()).map_err(CoreError::ArrowError)?);
                }
            }
            Err(_) => columns.push(new_null_array(field.data_type(), num_rows)),
        }
    }
    RecordBatch::try_new(target.clone(), columns).map_err(CoreError::ArrowError)
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
