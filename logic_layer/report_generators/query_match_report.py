import json
import os
from pathlib import Path
import nltk
from common.enums.folders import Folders
from common.enums.report_folder import ReportFolder
from common.util.std_in_out.root_locator import RootLocator, ROOT_DIR
from framework.common.logger.message_type import MessageType


from logic_layer.rag_ingest.util.multi_stage_rag.chunk_generation.transformers.ktransformers_chunk_generator import \
    KTransformersChunkGenerator


class QueryMatchReportK10Q10():

    def __init__(self,logger,portfolio,report_type,dest_folder):
        self.logger=logger
        self.portfolio = portfolio
        self.report_type=report_type
        self.dest_folder=dest_folder

        self.chunk_generator=KTransformersChunkGenerator()

        self.query =None
        from sentence_transformers import SentenceTransformer
        self.bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  #TODO from settings
        self.bi_threshold =0.3  # float – threshold for bi-encoder acceptance

    def run_analysis(self, symbol,query, year,report_type):
        self._log_start(symbol, year)

        self.query=query

        input_dir = self._build_input_dir(symbol, year)
        out_dir = self._build_output_dir(symbol, year)

        files = self._collect_input_files(input_dir, symbol, year,report_type)
        if not files:
            return []

        results = []

        for file in files:
            try:
                text = self._read_file_safe(file)
                if not self._is_text_valid(text, file):
                    continue

                sentences = self._split_sentences(text)
                if not sentences:
                    continue

                bi_result = self._run_bi_encoder_prescan(sentences, file)
                if not bi_result["matched"]:
                    continue

                match = self._build_match_result(symbol, year, file, bi_result)
                results.append(match)

                self._persist_result(match, out_dir, file)

            except Exception as e:
                self._log_error(symbol, file, e)

        self._log_end(symbol, year, len(results))
        return results


    #/////////////// Private Utils #///////////////
    def _build_input_dir(self, symbol, year):
        return (
                ROOT_DIR
                / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                / self.portfolio
                / self.report_type
                / str(year)

        )

    def _build_output_dir(self, symbol, year):
        out_dir = (
                ROOT_DIR
                / Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value
                / self.dest_folder
                / f"{self.report_type}_query_match_analysis"
                / str(year)

        )
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _log_start(self, symbol, year):
        self.logger.do_log(
            f"[QUERY-MATCH] 🚀 Starting analysis | {symbol} {year}",
            MessageType.INFO
        )

    def _log_end(self, symbol, year, count):
        self.logger.do_log(
            f"[QUERY-MATCH] ✅ Finished | {symbol} {year} | matched_docs={count}",
            MessageType.INFO
        )

    def _log_error(self, symbol, file, error):
        self.logger.do_log(
            f"[QUERY-MATCH][ERROR] {symbol} {file.name}: {error}",
            MessageType.ERROR
        )

    def _collect_input_files(self, input_dir, symbol, year, report_type):
        input_dir = Path(input_dir)

        if not input_dir.exists():
            self.logger.do_log(
                f"[QUERY-MATCH][SKIP] No input dir for {symbol} {year}",
                MessageType.DEBUG
            )
            return []

        if report_type == ReportFolder.K10.value:
            pattern = f"{symbol}_{year}_10-K.html"
        elif report_type == ReportFolder.Q10.value:
            pattern = f"{symbol}_{year}_Q*_10-Q.html"
        else:
            raise ValueError(f"Invalid report type: {report_type}")

        files = list(input_dir.glob(pattern))

        if not files:
            self.logger.do_log(
                f"[QUERY-MATCH][SKIP] No files for {symbol} {year}",
                MessageType.DEBUG
            )

        return files

    def _read_file_safe(self, file):
        return file.read_text(encoding="utf-8", errors="ignore")

    def _is_text_valid(self, text, file):
        if not text or len(text) < 500:
            self.logger.do_log(
                f"[QUERY-MATCH][SKIP] Empty or too small file: {file.name}",
                MessageType.DEBUG
            )
            return False
        return True

    def _split_sentences(self, text: str):
        """
        Split raw text into sentences for cheap prescan.
        This method MUST be fast and MUST NOT use embeddings or clustering.
        """
        try:
            sentences = nltk.sent_tokenize(text)

            # Basic sanity check to avoid tokenizer failures
            if sentences and len(sentences) > 3:
                return sentences
        except Exception:
            pass

        # Defensive fallback: simple punctuation-based split
        return [
            s.strip()
            for s in text.split(".")
            if len(s.strip()) > 40
        ]

    def _run_bi_encoder_prescan(self, sentences, file):
        query_vec = self.bi_encoder.encode(
            self.query,
            normalize_embeddings=True
        )

        sent_vecs = self.bi_encoder.encode(
            sentences,
            batch_size=64,
            normalize_embeddings=True
        )

        sims = sent_vecs @ query_vec

        max_score = float(sims.max())
        top_scores = sorted(sims, reverse=True)[:5]
        avg_top5 = float(sum(top_scores) / len(top_scores))

        self.logger.do_log(
            f"[QUERY-MATCH][BI] {file.name} | max={max_score:.3f} avg_top5={avg_top5:.3f}",
            MessageType.DEBUG
        )

        return {
            "matched": max_score >= self.bi_threshold,
            "bi_max_score": max_score,
            "bi_avg_top5": avg_top5
        }

    def _build_match_result(self, symbol, year, file, bi_result):
        return {
            "symbol": symbol,
            "year": year,
            "file": file.name,
            "bi_max_score": bi_result["bi_max_score"],
            "bi_avg_top5": bi_result["bi_avg_top5"],
            "matched": True
        }

    def _persist_result(self, match, out_dir, file):
        out_file = out_dir / f"{file.stem}_bi_match.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(match, f, indent=2)
