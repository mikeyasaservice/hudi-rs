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

//! Key generator implementations for transforming user filters to partition filters.

pub mod timestamp_based;

use crate::Result;
use crate::config::HudiConfigs;
use crate::config::table::HudiTableConfig::PartitionFields;
use crate::config::table::HudiTableConfig::{KeyGeneratorClass, KeyGeneratorType as KeyGenTypeCfg};
use crate::expr::filter::Filter;

/// The kind of Hudi key generator configured for a table.
///
/// Resolved from `hoodie.table.keygenerator.type` (v8+) and, failing that, inferred from
/// the `hoodie.table.keygenerator.class` Java class name. The `*_AVRO` type variants and the
/// Avro key-generator classes map to the same logical kinds as their non-Avro counterparts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyGeneratorType {
    /// Single record-key field, single partition field.
    Simple,
    /// Multiple record-key and/or partition fields.
    Complex,
    /// Partition path derived by formatting a timestamp source field.
    Timestamp,
    /// Per-partition-field types, e.g. `field1:SIMPLE,field2:TIMESTAMP`.
    Custom,
    /// No partition fields.
    NonPartition,
    /// Global delete (record key only).
    GlobalDelete,
}

impl KeyGeneratorType {
    /// Classify from a `hoodie.table.keygenerator.type` value (case-insensitive).
    pub fn from_type_str(s: &str) -> Option<Self> {
        match s.trim().to_uppercase().as_str() {
            "SIMPLE" | "SIMPLE_AVRO" => Some(Self::Simple),
            "COMPLEX" | "COMPLEX_AVRO" => Some(Self::Complex),
            "TIMESTAMP" | "TIMESTAMP_AVRO" => Some(Self::Timestamp),
            "CUSTOM" | "CUSTOM_AVRO" => Some(Self::Custom),
            "NON_PARTITION" | "NON_PARTITION_AVRO" => Some(Self::NonPartition),
            "GLOBAL_DELETE" | "GLOBAL_DELETE_AVRO" => Some(Self::GlobalDelete),
            _ => None,
        }
    }

    /// Classify from a `hoodie.table.keygenerator.class` Java class name. Checks the most
    /// specific names first so that, e.g., `NonpartitionedKeyGenerator` is not mistaken for
    /// a simple generator by a substring match.
    pub fn from_class_name(s: &str) -> Option<Self> {
        if s.contains("Nonpartitioned") {
            Some(Self::NonPartition)
        } else if s.contains("GlobalDelete") {
            Some(Self::GlobalDelete)
        } else if s.contains("TimestampBased") {
            Some(Self::Timestamp)
        } else if s.contains("Complex") {
            Some(Self::Complex)
        } else if s.contains("Custom") {
            Some(Self::Custom)
        } else if s.contains("Simple") {
            Some(Self::Simple)
        } else {
            None
        }
    }

    /// Resolve the key generator type from table configs, preferring the explicit type config
    /// and falling back to the class name. Returns `None` when neither is set or recognized.
    pub fn resolve(hudi_configs: &HudiConfigs) -> Result<Option<Self>> {
        if let Some(v) = hudi_configs.try_get(KeyGenTypeCfg)? {
            let s: String = v.into();
            if let Some(t) = Self::from_type_str(&s) {
                return Ok(Some(t));
            }
        }
        if let Some(v) = hudi_configs.try_get(KeyGeneratorClass)? {
            let s: String = v.into();
            if let Some(t) = Self::from_class_name(&s) {
                return Ok(Some(t));
            }
        }
        Ok(None)
    }
}

/// Parse a partition-field spec into its column name and optional per-field key-generator type.
///
/// [`KeyGeneratorType::Custom`] encodes the type per field as `field:TYPE`
/// (e.g. `ts:TIMESTAMP`, `city:SIMPLE`); other generators use the bare column name.
pub fn parse_partition_field_spec(spec: &str) -> (&str, Option<KeyGeneratorType>) {
    match spec.split_once(':') {
        Some((field, type_str)) => (field.trim(), KeyGeneratorType::from_type_str(type_str)),
        None => (spec.trim(), None),
    }
}

/// The partition column names declared by `hoodie.table.partition.fields`, with any
/// [`KeyGeneratorType::Custom`] `:TYPE` suffix stripped so the names match table-schema columns.
pub fn partition_column_names(hudi_configs: &HudiConfigs) -> Vec<String> {
    let specs: Vec<String> = hudi_configs.get_or_default(PartitionFields).into();
    specs
        .iter()
        .map(|spec| parse_partition_field_spec(spec).0.to_string())
        .collect()
}

/// Returns true if the table uses a timestamp-based key generator (the whole partition path
/// is a formatted timestamp). Note this is distinct from a [`KeyGeneratorType::Custom`]
/// generator that merely has a timestamp-typed partition field among others.
pub fn is_timestamp_based_keygen(hudi_configs: &HudiConfigs) -> Result<bool> {
    Ok(matches!(
        KeyGeneratorType::resolve(hudi_configs)?,
        Some(KeyGeneratorType::Timestamp)
    ))
}

/// Trait for key generators that can transform user filters on data columns
/// to filters on partition path columns.
pub trait KeyGeneratorFilterTransformer {
    /// Returns the source field name that this key generator operates on.
    fn source_field(&self) -> &str;

    /// Transforms a filter on the source field to one or more filters on partition fields.
    fn transform_filter(&self, filter: &Filter) -> Result<Vec<Filter>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generator_type_from_type_str() {
        assert_eq!(
            KeyGeneratorType::from_type_str("simple"),
            Some(KeyGeneratorType::Simple)
        );
        assert_eq!(
            KeyGeneratorType::from_type_str("COMPLEX"),
            Some(KeyGeneratorType::Complex)
        );
        assert_eq!(
            KeyGeneratorType::from_type_str("timestamp_avro"),
            Some(KeyGeneratorType::Timestamp)
        );
        assert_eq!(
            KeyGeneratorType::from_type_str("CUSTOM"),
            Some(KeyGeneratorType::Custom)
        );
        assert_eq!(
            KeyGeneratorType::from_type_str("non_partition"),
            Some(KeyGeneratorType::NonPartition)
        );
        assert_eq!(
            KeyGeneratorType::from_type_str("GLOBAL_DELETE"),
            Some(KeyGeneratorType::GlobalDelete)
        );
        assert_eq!(KeyGeneratorType::from_type_str("unknown"), None);
    }

    #[test]
    fn test_key_generator_type_from_class_name() {
        let cases = [
            (
                "org.apache.hudi.keygen.SimpleKeyGenerator",
                KeyGeneratorType::Simple,
            ),
            (
                "org.apache.hudi.keygen.ComplexKeyGenerator",
                KeyGeneratorType::Complex,
            ),
            (
                "org.apache.hudi.keygen.TimestampBasedKeyGenerator",
                KeyGeneratorType::Timestamp,
            ),
            (
                "org.apache.hudi.keygen.CustomKeyGenerator",
                KeyGeneratorType::Custom,
            ),
            (
                "org.apache.hudi.keygen.NonpartitionedKeyGenerator",
                KeyGeneratorType::NonPartition,
            ),
            (
                "org.apache.hudi.keygen.GlobalDeleteKeyGenerator",
                KeyGeneratorType::GlobalDelete,
            ),
            // Avro variants map to the same logical kinds.
            (
                "org.apache.hudi.keygen.NonpartitionedAvroKeyGenerator",
                KeyGeneratorType::NonPartition,
            ),
        ];
        for (class_name, expected) in cases {
            assert_eq!(
                KeyGeneratorType::from_class_name(class_name),
                Some(expected)
            );
        }
        assert_eq!(
            KeyGeneratorType::from_class_name("com.example.MyKeyGen"),
            None
        );
    }

    #[test]
    fn test_resolve_prefers_type_then_class() {
        // Explicit type wins.
        let configs = HudiConfigs::new([
            (KeyGenTypeCfg, "COMPLEX".to_string()),
            (
                KeyGeneratorClass,
                "org.apache.hudi.keygen.SimpleKeyGenerator".to_string(),
            ),
        ]);
        assert_eq!(
            KeyGeneratorType::resolve(&configs).unwrap(),
            Some(KeyGeneratorType::Complex)
        );

        // Falls back to class name when type is absent.
        let configs = HudiConfigs::new([(
            KeyGeneratorClass,
            "org.apache.hudi.keygen.TimestampBasedKeyGenerator".to_string(),
        )]);
        assert_eq!(
            KeyGeneratorType::resolve(&configs).unwrap(),
            Some(KeyGeneratorType::Timestamp)
        );

        // Neither configured.
        assert_eq!(
            KeyGeneratorType::resolve(&HudiConfigs::empty()).unwrap(),
            None
        );
    }

    #[test]
    fn test_is_timestamp_based_keygen_only_for_pure_timestamp() {
        let ts = HudiConfigs::new([(KeyGenTypeCfg, "TIMESTAMP".to_string())]);
        assert!(is_timestamp_based_keygen(&ts).unwrap());

        // A custom generator with a timestamp partition field is NOT a pure timestamp keygen.
        let custom = HudiConfigs::new([(KeyGenTypeCfg, "CUSTOM".to_string())]);
        assert!(!is_timestamp_based_keygen(&custom).unwrap());
    }

    #[test]
    fn test_parse_partition_field_spec() {
        assert_eq!(
            parse_partition_field_spec("ts:TIMESTAMP"),
            ("ts", Some(KeyGeneratorType::Timestamp))
        );
        assert_eq!(
            parse_partition_field_spec("city:SIMPLE"),
            ("city", Some(KeyGeneratorType::Simple))
        );
        // Bare column name (non-custom generators).
        assert_eq!(parse_partition_field_spec("city"), ("city", None));
        // Unknown type after the colon yields None for the type but still strips the suffix.
        assert_eq!(parse_partition_field_spec("city:WAT"), ("city", None));
    }

    #[test]
    fn test_partition_column_names_strips_custom_specs() {
        let configs = HudiConfigs::new([(PartitionFields, "ts:TIMESTAMP,city:SIMPLE".to_string())]);
        assert_eq!(
            partition_column_names(&configs),
            vec!["ts".to_string(), "city".to_string()]
        );

        // Bare names are returned unchanged.
        let configs = HudiConfigs::new([(PartitionFields, "year,month".to_string())]);
        assert_eq!(
            partition_column_names(&configs),
            vec!["year".to_string(), "month".to_string()]
        );
    }
}
