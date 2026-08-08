# FILE: tools/test_vectorization_e2e.py
# End-to-end check of the vectorization stack, in three stages so a failure
# tells you exactly which layer broke.
#
#   1) DB      - connection, pgvector, schema, insert/search round trip
#   2) EXTRACT - the narrative sections pulled out of a real filing
#   3) FULL    - one real filing vectorized and persisted (loads the model)
#
# Usage:
#   python tools/test_vectorization_e2e.py db
#   python tools/test_vectorization_e2e.py extract <path_to_filing.html>
#   python tools/test_vectorization_e2e.py full <path_to_filing.html> [model]

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from data_access_layer.vectors.filing_vectors_manager import FilingVectorsManager

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def resolve_db_config():
    return MLSettingsLoader().load_settings("./configs/commands_mgr.ini")


def test_db():
    manager = FilingVectorsManager(resolve_db_config())
    print(f"[1] Connection OK | {manager.ping()}")

    document_id = manager.upsert_document(
        symbol="__TEST__", cik=0, report_type="K10", fiscal_year=1900, quarter="",
        portfolio="__TEST__", sector_code="__TEST__", source_folder="__TEST__",
        file_name="__TEST__.html", file_path="__TEST__", content_hash="0" * 64,
    )
    print(f"[2] Document upsert OK | document_id={document_id}")

    vector_a = [0.0] * 768
    vector_a[0] = 1.0
    vector_b = [0.0] * 768
    vector_b[1] = 1.0

    manager.persist_chunks(document_id, "__TEST_MODEL__", [
        {"section_label": "ITEM 7 - MD&A", "chunk_index": 0,
         "chunk_text": "first test chunk", "word_count": 3, "embedding": vector_a},
        {"section_label": "ITEM 1A - RISK FACTORS", "chunk_index": 1,
         "chunk_text": "second test chunk", "word_count": 3, "embedding": vector_b},
    ])
    print("[3] Chunk insert OK | 2 chunks")

    results = manager.search_similar(vector_a, "__TEST_MODEL__", top_k=2)
    print(f"[4] Similarity search OK | top hit sim={results[0]['similarity']:.4f} "
          f"section={results[0]['section_label']}")

    if round(results[0]["similarity"], 3) != 1.0:
        raise Exception("Nearest neighbour is not the vector we searched with")

    print(f"[5] Skip check OK | already_vectorized="
          f"{manager.is_already_vectorized('__TEST__', 'K10', 1900, '', '__TEST__.html', '__TEST_MODEL__', '0' * 64)}")

    with manager.connection.cursor() as cursor:
        cursor.execute("DELETE FROM filing_documents WHERE symbol = '__TEST__'")
    manager.connection.commit()
    print("[6] Cleanup OK | test rows removed")

    manager.close()
    print("\nDB STAGE PASSED")


def build_tag_cfg(model):
    from common.util.tagging.tagging_config_dto import TaggingConfigDTO
    return TaggingConfigDTO(
        tag_model=model, tag_file=None, tags_csv=None,
        sim_threshold=None, doc_type=TaggingConfigDTO.DOC_TYPE_K_Q_10,
        tag_json=None, tag_dedup=True,
    )


def test_extract(file_path, doc_type="K_Q_10"):
    from common.util.extractors.section_extractors.section_extractor_registry import (
        SectionExtractorRegistry,
    )
    from common.util.std_in_out.raw_file_reader import RawFileReader

    file_name = os.path.basename(file_path)
    extractor = SectionExtractorRegistry.get(doc_type)
    report_type = extractor.resolve_sub_type(file_name)

    raw = RawFileReader.get_raw_text(file_path)
    blocks = extractor.extract_sections(raw, file_name)

    print(f"File={file_name} | resolved report_type={report_type} | sections={len(blocks)}\n")
    for label, text in blocks.items():
        print(f"  {label} | chars={len(text)} | words={len(text.split())}")
        print(f"    HEAD: {text[:160]}")
        print(f"    TAIL: {text[-160:]}\n")

    if not blocks:
        raise Exception("No sections extracted - check the filing format")

    print("EXTRACT STAGE PASSED")


def test_full(file_path, model):
    from framework.common.logger.logger import Logger
    from logic_layer.rag_corpus_metadata.vectorization.document_vectorization_processor import DocumentVectorizationProcessor

    logger = Logger()
    file_name = os.path.basename(file_path)

    processor = DocumentVectorizationProcessor(logger, build_tag_cfg(model), resolve_db_config())
    report_type, content_hash, chunks = processor.vectorize_file(file_path, file_name)

    if not chunks:
        raise Exception("No chunks produced")

    print(f"\nreport_type={report_type} | hash={content_hash[:12]}... | chunks={len(chunks)} "
          f"| dim={len(chunks[0]['embedding'])}")

    by_section = {}
    for chunk in chunks:
        by_section[chunk["section_label"]] = by_section.get(chunk["section_label"], 0) + 1
    for label, count in by_section.items():
        print(f"  {label}: {count} chunks")

    manager = processor.vectors_mgr
    document_id = manager.upsert_document(
        symbol="__E2E__", cik=0, report_type=report_type, fiscal_year=1900, quarter="",
        portfolio="__E2E__", sector_code=None, source_folder="__E2E__",
        file_name=file_name, file_path=file_path, content_hash=content_hash,
    )
    manager.delete_chunks(document_id, model)
    persisted = manager.persist_chunks(document_id, model, chunks)
    print(f"\nPersisted {persisted} chunks | document_id={document_id}")

    query_vector = processor.encode_query("gross margin expansion and pricing power")
    for row in manager.search_similar(query_vector, model, top_k=3, symbol="__E2E__"):
        print(f"  sim={row['similarity']:.4f} | {row['section_label']} | {row['chunk_text'][:110]}")

    with manager.connection.cursor() as cursor:
        cursor.execute("DELETE FROM filing_documents WHERE symbol = '__E2E__'")
    manager.connection.commit()
    processor.close()

    print("\nFULL STAGE PASSED")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__ or "See the header of this file for usage")
        sys.exit(1)

    stage = sys.argv[1].lower()

    if stage == "db":
        test_db()
    elif stage == "extract":
        test_extract(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "K_Q_10")
    elif stage == "full":
        test_full(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL)
    else:
        print(f"Unknown stage '{stage}'. Use db | extract | full")
        sys.exit(1)
