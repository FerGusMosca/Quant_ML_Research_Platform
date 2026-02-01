from enum import Enum

class ReportType(Enum):
    DOWNLOAD_K10 = "download_k10"
    DOWNLOAD_K8 = "download_k8"
    DOWNLOAD_Q10 = "download_q10"
    DOWNLOAD_YEARLY_INCOME_STATEMENT = "download_yearly_income_statement"
    DOWNLOAD_QUARTERLY_INCOME_STATEMENT = "download_quarterly_income_statement"
    DOWNLOAD_LAST_INCOME_STATEMENT = "download_last_income_statement"
    FINVIZ_NEWS_DOWNLOAD = "finviz_news_download"
    PROCESS_FINVIZ_NEWS = "process_finviz_news"
    COMPETITION_SUMMARY_REPORT_Q10 = "competition_summary_report_q10"
    COMPETITION_SUMMARY_REPORT_K10 = "competition_summary_report_k10"
    SENTIMENT_SUMMARY_REPORT_K10 = "sentiment_summary_report_k10"
    SENTIMENT_SUMMARY_REPORT_Q10 = "sentiment_summary_report_q10"
    SENTIMENT_SUMMARY_REPORT_SINGLE_STOCK_K10 = "sentiment_summary_single_security_report_k10"
    SENTIMENT_SUMMARY_REPORT_SINGLE_STOCK_Q10 = "sentiment_summary_single_security_report_q10"
    QUERY_MATCH_REPORT_K10 = "query_match_report_k10"

    FINANCIAL_RATIOS_REPORT_K10 = "financial_ratios_report_k10"
    FINANCIAL_RATIOS_REPORT_Q10 = "financial_ratios_report_q10"
    DOWNLOAD_SECURITIES_REPORTS_CALENDAR = "download_securities_reports_calendar"

    DOCUMENT_TAGGING_RANKING="document_tagging_ranking"
    COMPETITION_GRAPH = "competition_graph"
    DOWNLOAD_13F_REPORTS = "download_13f_reports"
    CREATE_13F_GRAPH = "create_13f_graphs"

    FINANCIAL_RATIOS_REPORT_SA="financial_ratios_report_sa"

    START_MCP = "start_mcp"
