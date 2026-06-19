<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# hudi-rs ↔ Apache Hudi Feature Gap Analysis

This document maps what `hudi-rs` implements today against the feature surface of Apache Hudi
proper, and proposes a value/effort-ranked roadmap for closing the gaps in native Rust. It covers
both the read and write paths.

It is a planning artifact, not an API contract — for the supported public reader surface see
[`reader-spec.md`](./reader-spec.md).

_Snapshot date: 2026-06-19. File:line references are accurate as of commit `0910ca7`._

## Contents

1. [Scope](#1-scope)
2. [Current-state inventory](#2-current-state-inventory)
3. [Gap catalog](#3-gap-catalog)
4. [Ranked roadmap](#4-ranked-roadmap)
5. [Out of scope](#5-out-of-scope)
6. [Appendix: in-code markers](#6-appendix-in-code-markers)

## 1. Scope

`hudi-rs` is currently a **read-only** native Rust consumer of Apache Hudi tables: Copy-on-Write
(COW) and Merge-on-Read (MOR) reads via snapshot, time-travel, and incremental queries, with
Python bindings, experimental C++ bindings, and a DataFusion `TableProvider`. Apache Hudi proper is
a full lakehouse storage engine: a large writer path, table services (compaction / clustering /
cleaning / archival / indexing), a multi-partition metadata + index layer, concurrency control, and
richer read semantics (CDC, full schema evolution, multiple merge modes).

This analysis enumerates the gaps and ranks them. Effort is a rough order of magnitude
(**S** ≈ days, **M** ≈ weeks, **L** ≈ months). Value reflects impact on the read-first user base
that exists today.

## 2. Current-state inventory

| Area | Status | Notes |
|------|--------|-------|
| Write path | ✗ Absent | No insert/upsert/delete/bulk_insert; no commit, markers, or `hoodie.properties` init |
| COW reads | ✓ Complete | Snapshot, time-travel, incremental, snapshot streaming |
| MOR reads | ✓ Complete | Base + log merge; read-optimized mode; rollback blocks honored |
| Incremental streaming | ✗ Absent | `read_stream` errors `Unsupported` (`table/mod.rs:821`) |
| Timeline actions | ◑ Partial | Only `commit` / `deltacommit` / `replacecommit` loaded (`timeline/mod.rs`); no clean/compaction/clustering/rollback/savepoint/restore/indexing |
| LSM archived timeline | ✗ Stub | v2 history reader returns empty (`timeline/loader.rs:256`) |
| Merge strategies | ◑ Partial | `AppendOnly`, `OverwriteWithLatest`; **single ordering field only** (`config/table.rs:267`) |
| Metadata table | ◑ Partial | `files` + `column_stats` partitions read (`metadata/table/column_stats.rs`); `bloom_filters`/`record_index`/`partition_stats` enums defined but unused |
| HFile reader | ✓ Complete | `crates/core/src/hfile/`; powers the MDT |
| Data skipping | ◑ Partial | MDT `column_stats` index used at planning time when available (`metadata/table/column_stats.rs`, wired in `table/fs_view.rs`), falling back to per-file Parquet footers (`table/file_pruner.rs`); base-HFile sourced only (post-compaction delta logs not yet read) |
| Partition pruning | ✓ Complete | Hive-style + standard paths (`table/partition.rs`) |
| Key generators | ◑ Partial | `TimestampBased` complete; Simple/Complex detection-only |
| Base file formats | ◑ Partial | Parquet (+ experimental Lance); ORC rejected (`config/table.rs:455`); HFile for MDT only |
| Log-only file groups | ✗ Absent | MOR slice without a base file unsupported (P1 TODOs: `file_group/mod.rs:195`, `file_group/builder.rs:284`, `table/listing.rs:155`) |
| CDC reads | ✗ Absent | `.cdc` log suffix not parsed (`file_group/log_file/mod.rs:76,226`) |
| Schema evolution | ◑ Partial | Resolver present; full add/drop/rename/reorder + type promotion not guaranteed |
| Avro→Arrow | ◑ Partial | Maps, local timestamps, schema refs, nested unions, Float16, MOR-streaming Decimal incomplete (`avro_to_arrow/`) |
| Storage backends | ✓ Complete | file/s3/az/gs via `object_store` |
| DataFusion provider | ✓ Mostly | Filter pushdown; OR predicates skipped |
| Python bindings | ✓ Mostly | Near-full Table API; no filter DSL / config-enum introspection |
| C++ bindings | ◑ Minimal | Only `FileGroupReader`; no Table/timeline/schema/metadata |

## 3. Gap catalog

### Read path

- **MDT column-stats partition → planning-time data skipping.** Highest read-perf lever.
  _Initial implementation landed_ (`metadata/table/column_stats.rs`, wired in `table/fs_view.rs`):
  the `column_stats` base HFiles are decoded into per-file `StatisticsContainer`s and fed to
  `FilePruner`, skipping files without a footer read. Remaining: source stats from
  post-compaction delta logs, and add IN/NOT-IN + decimal support. _High / M._
- **MDT partition-stats partition → partition pruning** without listing. _Med / M._
- **MDT record-level index (RLI) → point lookups** for equality on record keys; large win for
  engines pushing key predicates. _High / M._
- **CDC read queries.** Parse `.cdc` log blocks and expose a change-feed query type. _High / M._
- **Log-only file groups.** MOR slices with no base file currently error; unblocks tables written
  with certain configs. Contained, explicitly flagged P1. _Med / M._
- **Incremental streaming.** Only eager incremental exists today. _Med / S._
- **Full read-side schema evolution** (add/drop/rename/reorder columns, type promotion). _High / M._
- **Additional key generators** (Simple, Complex, NonPartitioned, Custom) including filter
  transforms for partition pruning. _Med / M._
- **More merge modes / payloads:** `EVENT_TIME_ORDERING`, `COMMIT_TIME_ORDERING`, `CUSTOM`,
  partial-update, and **multiple ordering fields** (currently rejected). _Med / M-L._
- **LSM archived-timeline reader** → time travel beyond the active timeline. _Med / M._
- **Broader timeline action coverage** (clean/compaction/clustering/rollback/savepoint/restore/
  indexing) → savepoint- and restore-aware reads. _Med / M._
- **ORC base-file reads; HFile as a base format.** _Med / M._
- **Avro→Arrow completeness** (maps, local timestamps, refs, nested unions, Float16, MOR-stream
  Decimal). _Med / M._
- **Expression/functional index, secondary index, bloom-filters MDT partition.** _Low-Med / M._
- **Bootstrap-table reads** (external-Parquet metadata bootstrap). _Med / M._

### Write path (entirely absent today)

- **Writer foundation:** COW `insert` / `upsert` / `bulk_insert` / `delete` / `insert_overwrite` /
  `delete_partition`; commit protocol, markers, heartbeats, `hoodie.properties` init. _High / L._
- **MOR writes** (log-block append) and **write-side indexing** (bloom/simple/bucket/RLI tagging).
  _High / L._
- **Table services:** compaction, clustering, cleaning, archival, async metadata indexing. _High / L._
- **Concurrency control:** OCC multi-writer and Hudi-1.0 non-blocking concurrency control. _Med / L._
- **Savepoint / restore execution; metadata-table maintenance/writes.** _Med / L._

### Cross-cutting

- **C++ binding parity:** expose Table/timeline/schema/metadata, not just `FileGroupReader`. _Med / M._
- **Python:** surface the filter DSL and config-enum introspection. _Low / S._
- **Windows CI** (`.github/workflows/ci.yml`). _Low / S._

## 4. Ranked roadmap

### Tier 1 — read-side, high value / contained effort (recommended near-term)

MDT column-stats data skipping; MDT partition-stats pruning; record-level index point lookups; CDC
reads; log-only file groups; incremental streaming; full read-side schema evolution; additional key
generators. These deliver the largest query-side wins, build on infrastructure already in the
codebase (HFile reader, MDT record-type enums, file pruner), and keep the project's read-first
focus.

### Tier 2 — read-side, medium value or larger effort

Additional merge modes + multiple ordering fields; LSM archived-timeline reader; broader timeline
action coverage; ORC / HFile base files; Avro→Arrow completeness; expression/secondary/bloom MDT
partitions; bootstrap-table reads.

### Tier 3 — the writer path (largest effort; "bring all of Hudi")

Writer foundation (COW + MOR), write-side indexing, table services, concurrency control, and
savepoint/restore execution. This is the bulk of Apache Hudi by volume and the biggest commitment;
sequence it after the read path is feature-complete, starting with COW bulk_insert/insert as the
smallest viable writer.

## 5. Out of scope

External catalog sync (Hive Metastore / AWS Glue) and engine-specific plugins
(Spark / Flink / Presto / Trino) live outside a native Rust library and are not "bring to Rust"
targets. Daft and Ray already integrate through the existing read APIs.

## 6. Appendix: in-code markers

Representative `TODO` / `unimplemented!` / "not yet supported" markers backing the gaps above:

- `crates/core/src/table/mod.rs:821` — incremental streaming returns `Unsupported`.
- `crates/core/src/metadata/table/mod.rs:112` — "support more partitions. Only 'files' is used".
- `crates/core/src/file_group/mod.rs:195,223`, `file_group/builder.rs:284`,
  `table/listing.rs:155` — log-only file groups (P1).
- `crates/core/src/file_group/log_file/mod.rs:76,226` — `.cdc` suffix support.
- `crates/core/src/config/table.rs:267` — multiple ordering fields rejected.
- `crates/core/src/config/table.rs:455` — ORC base format rejected.
- `crates/core/src/timeline/loader.rs:256` — v2 LSM history reader stubbed (returns empty).
- `crates/core/src/avro_to_arrow/schema.rs`, `arrow_array_reader.rs` — Avro→Arrow type gaps.
- `crates/core/src/statistics/estimator.rs` — stats estimator assumes Parquet base format.
