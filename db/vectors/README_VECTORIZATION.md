# Filing vectorization — Postgres / pgvector

## 1. Configuration

All connection settings live in the `[VECTOR_DB]` section of
`configs/commands_mgr.ini`:

```
[VECTOR_DB]
VECTORS_PG_HOST=localhost
VECTORS_PG_PORT=5433
VECTORS_PG_DB=quant_ml_vectors
VECTORS_PG_USER=bias_research
VECTORS_PG_PWD=YOUR_PWD
VECTORS_PG_SCHEMA=bias_research
```

`MLSettingsLoader` only reads those values. `FilingVectorsManager` is the one
that turns them into a psycopg2 connection, because it is the only layer that
knows the driver.

Dependency: `pip install psycopg2-binary`

## 2. Apply the schema

```
psql -h localhost -p 5433 -U bias_research -d quant_ml_vectors -f db/vectors/01_schema_pgvector.sql
```

The script drops and recreates the three tables, so it is destructive on rerun.

### Tables created (schema `bias_research`)

| Table | Purpose |
|---|---|
| `filing_documents` | One row per filing file vectorized. Unique on (symbol, report_type, fiscal_year, quarter, file_name). Holds `content_hash` (sha256) so an unchanged file is skipped on rerun. |
| `filing_chunks` | One row per chunk, with its `section_label`, text and `embedding vector(768)`. `embedding_model` is part of the unique key, so mpnet and bge can coexist over the same document and be compared. |
| `vectorization_runs` | One row per year processed: counts of found / processed / skipped / failed and chunks persisted. This is the audit trail. |

### Indexes

| Index | Table | Purpose |
|---|---|---|
| `idx_filing_documents_lookup` | filing_documents | lookup by report type, year, quarter, symbol |
| `idx_filing_documents_sector` | filing_documents | sector slicing |
| `idx_filing_chunks_document` | filing_chunks | fetch a document's chunks for one model |
| `idx_filing_chunks_section` | filing_chunks | filter by MD&A / Risk Factors |
| `idx_filing_chunks_embedding` | filing_chunks | **HNSW, cosine** — the semantic search index |
| `idx_vectorization_runs_lookup` | vectorization_runs | run history |

View `v_filing_chunks` joins chunk and document context; `search_similar()` queries it.

**Dimension:** the embedding column is `vector(768)`, which fits `all-mpnet-base-v2` and
`bge-base-en-v1.5`. A 384-dim model (`bge-small`) is rejected by the data access layer
rather than silently truncated.

## 3. Supporting a new document family

`doc_type` selects the extractor through `SectionExtractorRegistry`. Today only
`K_Q_10` is registered. To add earnings call transcripts:

1. Write `TranscriptSectionExtractor(BaseSectionExtractor)` with `DOC_TYPE = "TRANSCRIPT"`,
   implementing `resolve_sub_type()` and `extract_sections()`.
2. Add one line to `_EXTRACTORS` in `section_extractor_registry.py`.

Nothing in the orchestration layer, the processor or the database changes: the
new family is just another `doc_type` in the MCP message.

## 4. Run the vectorization

MCP (same shape as `document_tagging_ranking`):

```json
{
  "report": "vectorize_documents",
  "portfolio": "US_BIGCAP_EX_TEST_SMALL",
  "source": "US_BIGCAP_EX_TEST_SMALL/Q10",
  "year": "2025",
  "quarter": "Q1",
  "sector": "TECH",
  "overwrite": false,
  "tag_model": "sentence-transformers/all-mpnet-base-v2",
  "doc_type": "K_Q_10"
}
```

Console:

```
RunReport report=vectorize_documents portfolio=US_BIGCAP_EX source=US_BIGCAP_EX_SMALL/K10 year=2025 sector=ENERGY tag_model=sentence-transformers/all-mpnet-base-v2 doc_type=K_Q_10
```

Parameters: `sector` is optional (no sector = the whole portfolio); `quarter` only
applies to Q10; `overwrite=true` re-vectorizes files already stored.

## 5. Test end to end

```
python tools/test_vectorization_e2e.py db
python tools/test_vectorization_e2e.py extract output/securities_reports/US_BIGCAP_EX_TEST_SMALL/Q10/2025/AAPL_2025_Q1_10-Q.html
python tools/test_vectorization_e2e.py full output/securities_reports/US_BIGCAP_EX_TEST_SMALL/Q10/2025/AAPL_2025_Q1_10-Q.html
```

Stage `db` needs no model and takes a second. Stage `full` loads the transformer.
All three clean up their own test rows.

## 6. Check what landed

```sql
SET search_path TO bias_research, public;

SELECT embedding_model, report_type, fiscal_year, quarter,
       COUNT(DISTINCT document_id) AS documents, COUNT(*) AS chunks
FROM v_filing_chunks
GROUP BY 1,2,3,4
ORDER BY fiscal_year DESC;

SELECT symbol, section_label, COUNT(*)
FROM v_filing_chunks
GROUP BY 1,2
ORDER BY symbol;

SELECT * FROM vectorization_runs ORDER BY run_id DESC LIMIT 10;
```
