import os, json, re, statistics
from collections import Counter
from datetime import datetime

from common.enums.folders import Folders


class FinvizOfflineSentimentAnalyzer:

    POS_WORDS = ["soar", "surge", "rally", "beat", "record", "profit", "gain", "upgraded", "buying opportunity", "growth"]
    NEG_WORDS = ["drop", "decline", "downgrade", "loss", "concern", "selloff", "bearish", "fall", "miss", "crash"]

    @staticmethod
    def analyze_text(text: str) -> int:
        """Very simple offline sentiment scoring heuristic."""
        if not text:
            return 0

        t = text.lower()
        pos = sum(w in t for w in FinvizOfflineSentimentAnalyzer.POS_WORDS)
        neg = sum(w in t for w in FinvizOfflineSentimentAnalyzer.NEG_WORDS)
        score = pos - neg

        # Scale roughly to -5..+5
        if score > 5: score = 5
        elif score < -5: score = -5
        return score

    @staticmethod
    def process_folder(base_folder: str):
        """
        Process all Finviz *_full_news.json files inside the given folder.
        Performs offline sentiment scoring and produces per-date summaries.
        """
        summaries = []

        for root, _, files in os.walk(base_folder):
            for f in files:
                if f.endswith("_full_news.json"):
                    path = os.path.join(root, f)
                    with open(path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)

                    symbol = data.get("symbol")
                    articles = data.get("articles", [])

                    scores = []
                    for art in articles:
                        txt = (art.get("title") or "") + " " + (art.get("content") or "")
                        s = FinvizOfflineSentimentAnalyzer.analyze_text(txt)
                        art["sentiment_score"] = s
                        scores.append(s)

                    # === Summary statistics ===
                    if scores:
                        mean = round(statistics.mean(scores), 2)
                        stdev = round(statistics.pstdev(scores), 2)
                        hist = dict(Counter(scores))
                    else:
                        mean, stdev, hist = 0, 0, {}

                    # === Select top 3 most positive and 3 most negative articles ===
                    high = sorted(articles, key=lambda x: x["sentiment_score"], reverse=True)[:3]
                    low = sorted(articles, key=lambda x: x["sentiment_score"])[:3]

                    summary = {
                        "symbol": symbol,
                        "date": datetime.today().strftime("%Y-%m-%d"),
                        "avg_sentiment": mean,
                        "sentiment_stdev": stdev,
                        "distribution": hist,
                        "high_sentiment_news": [
                            {
                                "title": a.get("title", ""),
                                "score": a.get("sentiment_score", 0),
                                "link": a.get("link", "")
                            }
                            for a in high
                        ],
                        "low_sentiment_news": [
                            {
                                "title": a.get("title", ""),
                                "score": a.get("sentiment_score", 0),
                                "link": a.get("link", "")
                            }
                            for a in low
                        ]
                    }

                    # === Write individual summary file ===
                    out_path = os.path.join(root, f"{symbol}_sentiment_summary.json")
                    with open(out_path, "w", encoding="utf-8") as out:
                        json.dump(summary, out, indent=2, ensure_ascii=False)

                    summaries.append(summary)
                    print(f"[SentimentAnalyzer][INFO] ✅ Processed {symbol} -> mean {mean} (σ={stdev})")

        return summaries

    @staticmethod
    def process_portfolio(portfolio, symbol, d_from):
        """
        Scans Finviz full_news folders, processes sentiment for all valid JSONs,
        and produces a combined sentiment summary file.
        """

        base_dir = os.path.join(
            Folders.OUTPUT_SECURITIES_REPORTS_FOLDER.value,
            portfolio,
            "Finviz",
            "full_news",
            symbol
        )

        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"[FinvizOfflineSentimentAnalyzer][ERROR] Base path not found: {base_dir}")

        print(f"[FinvizOfflineSentimentAnalyzer][INFO] Processing Finviz news for {symbol} from {d_from.date()}")

        d_from_str = d_from.strftime("%Y-%m-%d")
        now_str = datetime.today().strftime("%Y-%m-%d")

        # === Step 1: Gather candidate year folders ===
        year_folders = [
            os.path.join(base_dir, f)
            for f in os.listdir(base_dir)
            if f.isdigit() and int(f) >= d_from.year
        ]
        if not year_folders:
            print(f"[FinvizOfflineSentimentAnalyzer][WARN] No year folders found from {d_from.year} onwards.")
            return None

        # === Step 2: Collect all valid Finviz JSONs after d_from ===
        candidate_files = []
        for year_dir in sorted(year_folders):
            for root, _, files in os.walk(year_dir):
                for file in files:
                    if file.endswith("_full_news.json"):
                        try:
                            parts = file.split("_")
                            date_str = parts[1]  # expecting YYYY-MM-DD
                            file_date = datetime.strptime(date_str, "%Y-%m-%d")
                            if file_date >= d_from:
                                candidate_files.append(os.path.join(root, file))
                        except Exception:
                            continue

        if not candidate_files:
            print(f"[FinvizOfflineSentimentAnalyzer][WARN] No news files found after {d_from_str}.")
            return None

        print(f"[FinvizOfflineSentimentAnalyzer][INFO] Found {len(candidate_files)} news files to process.")

        # === Step 3: Perform offline sentiment analysis ===
        processed_summaries = []
        for file_path in candidate_files:
            folder = os.path.dirname(file_path)
            folder_summary = FinvizOfflineSentimentAnalyzer.process_folder(folder)
            processed_summaries.extend(folder_summary)

        # === Step 4: Build the combined sentiment summary ===
        combined = {
            "symbol": symbol,
            "from_date": d_from_str,
            "to_date": now_str,
            "total_articles": sum(len(s.get("distribution", [])) for s in processed_summaries),
            "summaries": processed_summaries
        }

        out_file = os.path.join(
            base_dir,
            f"{symbol}_{d_from_str}_to_{now_str}_full_sentiment_summary.json"
        )

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"[FinvizOfflineSentimentAnalyzer][INFO] ✅ Saved combined sentiment summary -> {out_file}")
        return out_file

