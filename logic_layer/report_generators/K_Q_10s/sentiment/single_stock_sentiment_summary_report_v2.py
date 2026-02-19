import json
from pathlib import Path
from typing import Dict, Optional

from common.enums.report_folder import ReportFolder
from common.enums.sec_reports import SECReports
from common.util.extractors.K_Q_10.k_q_10_mdna_extractor import KQ10MDNAExtractor
from framework.common.logger.message_type import MessageType
from logic_layer.report_generators.K_Q_10s.sentiment.base_sentiment_summary_report import SentimentAnalysisBase


class SentimentSingleSecurity(SentimentAnalysisBase):
    """
    Sentiment analysis for a single security.
    Returns JSON result suitable for API responses.
    """

    def __init__(self, logger):
        """Initialize with logger only - no file system dependencies."""
        super().__init__(logger)

    def analyze(
            self,
            symbol: str,
            report_type: str,
            year: int,
            portfolio: str,
            quarter: Optional[int] = None,
            job_id: Optional[str] = None
    ) -> Dict:
        """
        Analyze sentiment for a single security.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            report_type: '10K' or '10Q'
            year: Filing year
            portfolio: Portfolio name for path resolution
            quarter: Required if report_type is '10Q' (1-4)
            job_id: Optional job identifier for logging

        Returns:
            Dict with sentiment analysis results

        Raises:
            FileNotFoundError: If filing not found
            ValueError: If invalid parameters
            Exception: For analysis errors
        """
        symbol = symbol.upper().strip()
        report_type = report_type.upper().strip()

        # Validation
        if report_type not in [ReportFolder.K10.value, ReportFolder.Q10.value]:
            raise ValueError(f"Invalid report_type: {report_type}. Must be '10K' or '10Q'")

        if report_type == ReportFolder.Q10.value and not quarter:
            raise ValueError("Quarter (1-4) is required for 10Q reports")

        if quarter and (quarter < 1 or quarter > 4):
            raise ValueError(f"Invalid quarter: {quarter}. Must be 1-4")

        period_label = f"Q{quarter}" if quarter else f"Y{year}"

        self.logger.do_log(
            f"[SENT-SINGLE] 🚀 Starting analysis: {symbol} {report_type} {year} {period_label}",
            MessageType.INFO,
            job_id
        )

        try:
            # Build filing path
            filing_path = self._build_single_file_path(
                portfolio=portfolio,
                report_type=report_type,
                year=year,
                symbol=symbol,
                quarter=quarter
            )

            self.logger.do_log(
                f"[SENT-SINGLE][{symbol}] 🔍 Looking for: {filing_path}",
                MessageType.DEBUG,
                job_id
            )

            # Check if file exists
            if not filing_path.exists():
                error_msg = f"Filing not found: {filing_path.name}"
                self.logger.do_log(
                    f"[SENT-SINGLE][{symbol}] ❌ {error_msg}",
                    MessageType.ERROR,
                    job_id
                )
                raise FileNotFoundError(error_msg)

            # Extract text from HTML
            self.logger.do_log(
                f"[SENT-SINGLE][{symbol}] 📄 Extracting text from HTML...",
                MessageType.INFO,
                job_id
            )

            # Extract MD&A section
            self.logger.do_log(
                f"[SENT-SINGLE][{symbol}] 🔍 Extracting MD&A section...",
                MessageType.INFO,
                job_id
            )

            mdna_extractor = KQ10MDNAExtractor(self.logger)

            if (report_type == SECReports.K10.value):
                text = self._html_to_text(filing_path)
                mdna = mdna_extractor._extract_10k(text, symbol)
            elif report_type== SECReports.Q10.value:
                text = self._html_to_html(filing_path)
                mdna = mdna_extractor._extract_10q(text, symbol)
            else:
                raise Exception(f"Invalid report type extracting MDNA section:{report_type}")
            #mdna = self._extract_mdna(text, symbol)

            if not mdna or len(mdna.strip()) < 500:
                error_msg = f"MD&A extraction failed - only {len(mdna)} chars extracted"
                self.logger.do_log(
                    f"[SENT-SINGLE][{symbol}] ❌ {error_msg}",
                    MessageType.ERROR,
                    job_id
                )
                raise Exception(error_msg)

            self.logger.do_log(
                f"[SENT-SINGLE][{symbol}] ✅ MD&A extracted: {len(mdna):,} chars",
                MessageType.INFO,
                job_id
            )

            # Score MD&A
            self.logger.do_log(
                f"[SENT-SINGLE][{symbol}] 📊 Analyzing sentiment...",
                MessageType.INFO,
                job_id
            )

            result = self._score_mdna(mdna)

            # Build response
            output = {
                "status": "success",
                "symbol": symbol,
                "year": year,
                "period": period_label,
                "report_type": report_type,
                "quarter": quarter,
                "model_used": "Loughran-McDonald + VADER (calibrated)",
                "filing_path": str(filing_path.name),
                "mdna_length": len(mdna),
                "analysis": {
                    "metrics": result["metrics"],
                    "top_positive": result["top_positive"],
                    "top_negative": result["top_negative"],
                    #"forward_snippets": result["forward_snippets"],
                   # "curated_text": result["curated_text"],
                }
            }

            metrics = result["metrics"]
            self.logger.do_log(
                f"[SENT-SINGLE][{symbol}] ✅ COMPLETE | "
                f"Sentiment={metrics['mdna_sentiment']:.3f} | "
                f"Sentences={metrics['financial_sentences']} | "
                f"Forward={metrics['forward_ratio']:.1%} | "
                f"Hedge={metrics['hedge_ratio']:.1%}",
                MessageType.INFO,
                job_id
            )

            return output

        except FileNotFoundError as e:
            self.logger.do_log(
                f"[SENT-SINGLE][{symbol}] ❌ File not found: {str(e)}",
                MessageType.ERROR,
                job_id
            )
            return {
                "status": "error",
                "error_type": "file_not_found",
                "symbol": symbol,
                "year": year,
                "period": period_label,
                "report_type": report_type,
                "message": str(e)
            }

        except ValueError as e:
            self.logger.do_log(
                f"[SENT-SINGLE][{symbol}] ❌ Validation error: {str(e)}",
                MessageType.ERROR,
                job_id
            )
            return {
                "status": "error",
                "error_type": "validation_error",
                "symbol": symbol,
                "year": year,
                "report_type": report_type,
                "message": str(e)
            }

        except Exception as e:
            self.logger.do_log(
                f"[SENT-SINGLE][{symbol}] 💥 Analysis failed: {str(e)}",
                MessageType.ERROR,
                job_id
            )
            return {
                "status": "error",
                "error_type": "analysis_error",
                "symbol": symbol,
                "year": year,
                "period": period_label,
                "report_type": report_type,
                "message": str(e)
            }