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
//! Point lookups against the metadata table `record_index` partition.
//!
//! The record-level index (RLI) maps a record key to the location of the data file that
//! holds it. The HFile key in each bucket is the record key itself, and the payload's
//! `recordIndexMetadata` carries the partition path and the file id (as Java UUID
//! most/least-significant bits plus a file index). Looking up a key yields the single file
//! group that can contain the record, so equality predicates on the record key can skip
//! every other file group.

use std::collections::HashMap;

use apache_avro::Schema as AvroSchema;
use apache_avro::types::Value as AvroValue;

use crate::Result;
use crate::error::CoreError;
use crate::hfile::HFileReader;
use crate::metadata::table::column_stats::{
    extract_long, extract_string, find_field, latest_base_files_per_bucket, unwrap_union,
};
use crate::metadata::table::records::decode_avro_value;
use crate::table::Table;

/// The metadata table partition name that stores the record-level index.
pub const RECORD_INDEX_PARTITION_NAME: &str = "record_index";

/// The location of a record's data file, as recorded by the record-level index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordLocation {
    /// The data table partition path holding the record (e.g. `city=san_francisco`).
    pub partition_path: String,
    /// The file id of the file group holding the record (e.g. `<uuid>-0`).
    pub file_id: String,
}

/// Format two Java `UUID` bit halves into the canonical 8-4-4-4-12 hex string.
fn uuid_string_from_bits(high: i64, low: i64) -> String {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&high.to_be_bytes());
    bytes[8..].copy_from_slice(&low.to_be_bytes());
    let mut out = String::with_capacity(36);
    for (i, b) in bytes.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            out.push('-');
        }
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Reconstruct the file id from a `recordIndexMetadata` record.
///
/// When the index uses the default UUID encoding, the file id is the UUID built from
/// `fileIdHighBits`/`fileIdLowBits` suffixed with `fileIndex`. Some indexes instead store a
/// raw `fileId` string, which is used verbatim when present.
fn reconstruct_file_id(ri_fields: &[(String, AvroValue)]) -> Option<String> {
    if let Some(file_id) = find_field(ri_fields, "fileId").and_then(extract_string)
        && !file_id.is_empty()
    {
        return Some(file_id);
    }
    let high = find_field(ri_fields, "fileIdHighBits").and_then(extract_long)?;
    let low = find_field(ri_fields, "fileIdLowBits").and_then(extract_long)?;
    let file_index = find_field(ri_fields, "fileIndex")
        .and_then(extract_long)
        .unwrap_or(0);
    Some(format!("{}-{file_index}", uuid_string_from_bits(high, low)))
}

/// Decode the location from a `record_index` HFile record value.
///
/// Returns `Ok(None)` for tombstones (empty value) or records without a
/// `recordIndexMetadata` payload (e.g. a deleted key).
pub fn decode_record_location_with_schema(
    value: &[u8],
    schema: &AvroSchema,
) -> Result<Option<RecordLocation>> {
    if value.is_empty() {
        return Ok(None);
    }
    let avro_value = decode_avro_value(value, schema)?;
    let AvroValue::Record(top_fields) = &avro_value else {
        return Ok(None);
    };
    let Some(AvroValue::Record(ri_fields)) =
        find_field(top_fields, "recordIndexMetadata").map(unwrap_union)
    else {
        return Ok(None);
    };

    let partition_path = find_field(ri_fields, "partitionName")
        .and_then(extract_string)
        .unwrap_or_default();
    let Some(file_id) = reconstruct_file_id(ri_fields) else {
        return Ok(None);
    };
    Ok(Some(RecordLocation {
        partition_path,
        file_id,
    }))
}

impl Table {
    /// Look up record keys in the metadata table `record_index` partition.
    ///
    /// Returns a map from each found record key to its [`RecordLocation`]. Keys absent from
    /// the index (or whose latest index entry is a delete) are omitted. Returns an empty map
    /// when the metadata table is not enabled or has no record-index partition.
    ///
    /// Must be called on a DATA table, not a METADATA table.
    pub async fn lookup_record_index(
        &self,
        record_keys: &[&str],
    ) -> Result<HashMap<String, RecordLocation>> {
        let metadata_table = self.get_or_init_metadata_table().await?;
        metadata_table
            .fetch_record_index_locations(record_keys)
            .await
    }

    /// Read the `record_index` buckets and resolve the requested keys to file locations.
    ///
    /// A record key lives in exactly one bucket; since the bucket hash is not reproduced
    /// here, every bucket's latest base HFile is probed with a keyed lookup. Must be called
    /// on a METADATA table.
    pub(crate) async fn fetch_record_index_locations(
        &self,
        record_keys: &[&str],
    ) -> Result<HashMap<String, RecordLocation>> {
        if record_keys.is_empty() {
            return Ok(HashMap::new());
        }
        let Some(latest_ts) = self.timeline.get_latest_commit_timestamp_as_option() else {
            return Ok(HashMap::new());
        };

        let storage = self.file_system_view.storage.as_ref();
        let base_files =
            latest_base_files_per_bucket(storage, RECORD_INDEX_PARTITION_NAME, latest_ts).await?;
        if base_files.is_empty() {
            return Ok(HashMap::new());
        }

        // HFile keyed lookup expects sorted keys.
        let mut sorted_keys: Vec<&str> = record_keys.to_vec();
        sorted_keys.sort_unstable();
        sorted_keys.dedup();

        let mut out: HashMap<String, RecordLocation> = HashMap::new();
        for base_file in &base_files {
            let relative_path = format!("{RECORD_INDEX_PARTITION_NAME}/{}", base_file.file_name());
            let mut reader = HFileReader::open(storage, &relative_path)
                .await
                .map_err(|e| {
                    CoreError::MetadataTable(format!(
                        "Failed to open record_index base file {relative_path}: {e:?}"
                    ))
                })?;
            let avro_schema = reader
                .get_avro_schema()
                .map_err(|e| {
                    CoreError::MetadataTable(format!("Failed to get record_index schema: {e:?}"))
                })?
                .cloned();
            let Some(avro_schema) = avro_schema else {
                continue;
            };
            let hits = reader.lookup_records(&sorted_keys).map_err(|e| {
                CoreError::MetadataTable(format!("Failed to look up record_index keys: {e:?}"))
            })?;
            for (key, maybe_record) in hits {
                if let Some(record) = maybe_record
                    && let Some(location) =
                        decode_record_location_with_schema(record.value(), &avro_schema)?
                {
                    out.insert(key, location);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::table::records::parse_avro_schema;
    use hudi_test::QuickstartTripsTable;

    #[test]
    fn test_uuid_string_from_bits() {
        // Verified against the san_francisco file id in the V8 trips fixture.
        assert_eq!(
            uuid_string_from_bits(247114695546521503, -4833926705110317157),
            "036ded81-9ed4-479f-bcea-7145dfa0079b"
        );
    }

    #[test]
    fn test_decode_empty_value_is_none() {
        let schema = parse_avro_schema(
            r#"{"type":"record","name":"T","fields":[{"name":"type","type":"int"}]}"#,
        )
        .unwrap();
        assert!(
            decode_record_location_with_schema(&[], &schema)
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_lookup_record_index_resolves_known_keys() {
        let table_path = QuickstartTripsTable::V8Trips8I3U1D.path_to_mor_avro();
        let table = Table::new(&table_path).await.unwrap();

        // A record key (uuid) known to live in city=san_francisco.
        let key = "334e26e9-8355-45cc-97c6-c31daf0df330";
        let located = table.lookup_record_index(&[key]).await.unwrap();

        let location = located.get(key).expect("record key should be indexed");
        assert_eq!(location.partition_path, "city=san_francisco");
        assert_eq!(location.file_id, "036ded81-9ed4-479f-bcea-7145dfa0079b-0");

        // A key that is not present resolves to nothing.
        let missing = table
            .lookup_record_index(&["00000000-0000-0000-0000-000000000000"])
            .await
            .unwrap();
        assert!(missing.is_empty());
    }
}
