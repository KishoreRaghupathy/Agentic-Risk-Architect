"""
data_loader.py — Production-Grade Financial Data Ingestion Module
=================================================================

Agentic Risk Architect · Ingestion Layer
-----------------------------------------
Provides high-integrity data pipelines for:
  • Historical OHLCV market data via yfinance
  • Real-time financial news metadata via NewsAPI

Design Principles
-----------------
- **Fail-Safe Defaults**: Every external call is wrapped in structured error
  handling with domain-specific custom exceptions.
- **Observability**: Python ``logging`` is configured at module level with
  a consistent formatter so logs are instantly greppable in production.
- **Immutability**: Returned DataFrames are copies; callers can never corrupt
  the internal cache.
- **Type Safety**: All public functions carry full PEP 484/604 type hints.

Author  : Kishore Raghupathy
Module  : src.utils.data_loader
Version : 1.0.0
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Environment & Logging Bootstrap
# ──────────────────────────────────────────────
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)


# ──────────────────────────────────────────────
# Custom Exceptions
# ──────────────────────────────────────────────
class DataLoaderError(Exception):
    """Base exception for all data-loader failures."""


class TickerNotFoundError(DataLoaderError):
    """Raised when a given ticker symbol cannot be resolved by yfinance."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        super().__init__(
            f"Ticker '{ticker}' not found. Verify the symbol exists on a "
            f"supported exchange (NYSE, NASDAQ, BSE, NSE, etc.)."
        )


class NewsAPIError(DataLoaderError):
    """Raised when the NewsAPI request fails or returns an error status."""

    def __init__(self, status: str, message: str) -> None:
        self.status = status
        super().__init__(f"NewsAPI error [{status}]: {message}")


class MissingAPIKeyError(DataLoaderError):
    """Raised when a required API key is absent from the environment."""

    def __init__(self, key_name: str) -> None:
        self.key_name = key_name
        super().__init__(
            f"Environment variable '{key_name}' is not set. "
            f"Add it to your .env file or export it in your shell."
        )


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_DEFAULT_HISTORY_YEARS: int = 2
_NEWSAPI_BASE_URL: str = "https://newsapi.org/v2/everything"
_NEWSAPI_TIMEOUT_SECONDS: int = 15
_NEWSAPI_MAX_PAGE_SIZE: int = 100
_VALID_SORT_OPTIONS: set[str] = {"relevancy", "popularity", "publishedAt"}


# ──────────────────────────────────────────────
# Public API — Market Data
# ──────────────────────────────────────────────
def fetch_historical_data(
    ticker: str,
    years: int = _DEFAULT_HISTORY_YEARS,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single equity from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance-compatible ticker symbol (e.g. ``"AAPL"``, ``"RELIANCE.NS"``).
    years : int, optional
        Number of years of history to retrieve. Defaults to **2**.
    interval : str, optional
        Bar interval — ``"1d"`` (default), ``"1wk"``, ``"1mo"``, etc.
    auto_adjust : bool, optional
        If ``True`` (default), adjusts OHLC for splits and dividends.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``Date`` with columns:
        ``Open, High, Low, Close, Volume`` (and ``Adj Close`` when relevant).

    Raises
    ------
    TickerNotFoundError
        If the ticker symbol is invalid or delisted.
    DataLoaderError
        For any unexpected downstream failure (network outage, API change).

    Examples
    --------
    >>> df = fetch_historical_data("AAPL")
    >>> df.shape[1] >= 5
    True
    """
    ticker = ticker.strip().upper()
    logger.info("Fetching %d-year history for '%s' (interval=%s)", years, ticker, interval)

    end_date: datetime = datetime.now(timezone.utc)
    start_date: datetime = end_date - timedelta(days=years * 365)

    try:
        yf_ticker: yf.Ticker = yf.Ticker(ticker)

        # Validate ticker existence — yfinance silently returns empty frames
        # for invalid symbols, so we must check explicitly.
        info: dict[str, Any] = yf_ticker.info
        if not info or info.get("regularMarketPrice") is None:
            raise TickerNotFoundError(ticker)

        df: pd.DataFrame = yf_ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=auto_adjust,
        )

        if df.empty:
            raise TickerNotFoundError(ticker)

        logger.info(
            "Successfully fetched %d rows for '%s' [%s → %s]",
            len(df),
            ticker,
            df.index.min().strftime("%Y-%m-%d"),
            df.index.max().strftime("%Y-%m-%d"),
        )
        return df.copy()

    except TickerNotFoundError:
        raise  # re-raise domain exception as-is

    except Exception as exc:
        logger.exception("Unexpected error while fetching data for '%s'", ticker)
        raise DataLoaderError(
            f"Failed to fetch historical data for '{ticker}': {exc}"
        ) from exc


def fetch_multiple_tickers(
    tickers: list[str],
    years: int = _DEFAULT_HISTORY_YEARS,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch historical data for a batch of tickers.

    Iterates over ``tickers`` and collects each result into a dictionary.
    Individual failures are **logged and skipped** so one bad symbol does not
    block the entire batch.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    years : int, optional
        Years of history per ticker.  Defaults to ``2``.
    interval : str, optional
        Bar interval.  Defaults to ``"1d"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of *successfully resolved* ticker symbols to their DataFrames.

    Examples
    --------
    >>> results = fetch_multiple_tickers(["AAPL", "MSFT", "INVALID_XYZ"])
    >>> "AAPL" in results
    True
    """
    results: dict[str, pd.DataFrame] = {}
    for symbol in tickers:
        try:
            results[symbol.strip().upper()] = fetch_historical_data(
                symbol, years=years, interval=interval
            )
        except DataLoaderError as err:
            logger.warning("Skipping '%s': %s", symbol, err)
    logger.info(
        "Batch fetch complete — %d/%d tickers resolved.",
        len(results),
        len(tickers),
    )
    return results


def get_ticker_metadata(ticker: str) -> dict[str, Any]:
    """Return fundamental metadata for a single ticker.

    Useful for extracting sector, market cap, P/E ratio, and other
    fundamentals that feed into the risk scoring pipeline.

    Parameters
    ----------
    ticker : str
        Yahoo Finance-compatible ticker symbol.

    Returns
    -------
    dict[str, Any]
        Raw ``info`` dictionary from yfinance.

    Raises
    ------
    TickerNotFoundError
        If the ticker cannot be resolved.
    """
    ticker = ticker.strip().upper()
    logger.info("Fetching metadata for '%s'", ticker)

    try:
        yf_ticker: yf.Ticker = yf.Ticker(ticker)
        info: dict[str, Any] = yf_ticker.info or {}

        if not info or info.get("regularMarketPrice") is None:
            raise TickerNotFoundError(ticker)

        logger.info(
            "Metadata received for '%s' — sector=%s, mktCap=%s",
            ticker,
            info.get("sector", "N/A"),
            info.get("marketCap", "N/A"),
        )
        return info

    except TickerNotFoundError:
        raise

    except Exception as exc:
        logger.exception("Unexpected error fetching metadata for '%s'", ticker)
        raise DataLoaderError(
            f"Failed to retrieve metadata for '{ticker}': {exc}"
        ) from exc


# ──────────────────────────────────────────────
# Public API — News Ingestion
# ──────────────────────────────────────────────
def fetch_latest_news(
    query: str,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    sort_by: str = "publishedAt",
    page_size: int = 20,
    language: str = "en",
    api_key: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Fetch the latest financial news articles from NewsAPI.

    Parameters
    ----------
    query : str
        Free-text search query (e.g. ``"AAPL earnings"``).
    from_date : str, optional
        ISO-8601 date (``YYYY-MM-DD``) for the oldest article.
        Defaults to **7 days ago**.
    to_date : str, optional
        ISO-8601 date for the newest article.  Defaults to **today**.
    sort_by : str, optional
        Sort order — one of ``"relevancy"``, ``"popularity"``,
        ``"publishedAt"`` (default).
    page_size : int, optional
        Number of articles to return (1–100).  Defaults to ``20``.
    language : str, optional
        Two-letter ISO-639 language code.  Defaults to ``"en"``.
    api_key : str, optional
        NewsAPI key.  If ``None``, reads from ``NEWS_API_KEY`` env var.

    Returns
    -------
    list[dict[str, Any]]
        List of article dictionaries with keys:
        ``source, author, title, description, url, urlToImage,
        publishedAt, content``.

    Raises
    ------
    MissingAPIKeyError
        If no API key is found in args or environment.
    NewsAPIError
        If the API returns a non-``ok`` status.
    DataLoaderError
        For network-level failures (timeout, DNS, etc.).

    Examples
    --------
    >>> articles = fetch_latest_news("Tesla stock")
    >>> isinstance(articles, list)
    True
    """
    resolved_key: str = api_key or os.getenv("NEWS_API_KEY", "")
    if not resolved_key:
        raise MissingAPIKeyError("NEWS_API_KEY")

    # ── Validate & default dates ─────────────────────────
    today: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week_ago: str = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    if sort_by not in _VALID_SORT_OPTIONS:
        logger.warning(
            "Invalid sort_by='%s'. Falling back to 'publishedAt'.", sort_by
        )
        sort_by = "publishedAt"

    page_size = max(1, min(page_size, _NEWSAPI_MAX_PAGE_SIZE))

    params: dict[str, Any] = {
        "q": query,
        "from": from_date or week_ago,
        "to": to_date or today,
        "sortBy": sort_by,
        "pageSize": page_size,
        "language": language,
        "apiKey": resolved_key,
    }

    logger.info(
        "Querying NewsAPI — q='%s', from=%s, to=%s, pageSize=%d",
        query,
        params["from"],
        params["to"],
        page_size,
    )

    try:
        response: requests.Response = requests.get(
            _NEWSAPI_BASE_URL,
            params=params,
            timeout=_NEWSAPI_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        payload: dict[str, Any] = response.json()

        if payload.get("status") != "ok":
            raise NewsAPIError(
                status=payload.get("code", "unknown"),
                message=payload.get("message", "No detail provided by API."),
            )

        articles: list[dict[str, Any]] = payload.get("articles", [])
        logger.info(
            "NewsAPI returned %d articles for query '%s'.",
            len(articles),
            query,
        )
        return articles

    except requests.exceptions.Timeout:
        logger.error("NewsAPI request timed out after %ds.", _NEWSAPI_TIMEOUT_SECONDS)
        raise DataLoaderError(
            f"NewsAPI request timed out after {_NEWSAPI_TIMEOUT_SECONDS}s."
        )

    except requests.exceptions.ConnectionError as exc:
        logger.error("Network error while contacting NewsAPI: %s", exc)
        raise DataLoaderError(f"Network error contacting NewsAPI: {exc}") from exc

    except NewsAPIError:
        raise

    except Exception as exc:
        logger.exception("Unexpected error during NewsAPI call.")
        raise DataLoaderError(f"NewsAPI call failed: {exc}") from exc


# ──────────────────────────────────────────────
# Public API — Convenience / Composite
# ──────────────────────────────────────────────
def fetch_ticker_with_news(
    ticker: str,
    years: int = _DEFAULT_HISTORY_YEARS,
    news_page_size: int = 10,
) -> dict[str, Any]:
    """One-shot convenience: market data + latest news for a single ticker.

    Combines :func:`fetch_historical_data`, :func:`get_ticker_metadata`,
    and :func:`fetch_latest_news` into a single composite payload suitable
    for the downstream Predictive Engine and Agentic Layer.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    years : int, optional
        History depth.  Defaults to ``2``.
    news_page_size : int, optional
        Number of news articles.  Defaults to ``10``.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        ``ticker, metadata, history, news, fetched_at``.

    Raises
    ------
    TickerNotFoundError
        If the ticker cannot be resolved by yfinance.
    DataLoaderError
        For any sub-component failure.
    """
    ticker = ticker.strip().upper()
    logger.info(
        "Composite fetch started for '%s' (years=%d, news=%d)",
        ticker,
        years,
        news_page_size,
    )

    metadata: dict[str, Any] = get_ticker_metadata(ticker)
    history: pd.DataFrame = fetch_historical_data(ticker, years=years)

    # Build a news query from the company's long name or fall back to ticker
    company_name: str = metadata.get("longName", ticker)
    news: list[dict[str, Any]] = []
    try:
        news = fetch_latest_news(company_name, page_size=news_page_size)
    except DataLoaderError as err:
        # News failure should NOT block market data delivery
        logger.warning("News fetch failed — proceeding without news: %s", err)

    result: dict[str, Any] = {
        "ticker": ticker,
        "metadata": metadata,
        "history": history,
        "news": news,
        "fetched_at": datetime.now(timezone.utc).isoformat() + "Z",
    }
    logger.info("Composite fetch complete for '%s'.", ticker)
    return result


# ──────────────────────────────────────────────
# Module Self-Test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("  Agentic Risk Architect — Data Loader Self-Test")
    print("=" * 72)

    TEST_TICKER: str = "AAPL"

    # 1. Historical data
    print(f"\n[1/4] Fetching 2-year history for {TEST_TICKER}...")
    try:
        df = fetch_historical_data(TEST_TICKER)
        print(f"  [OK]  {len(df)} rows fetched  |  Columns: {list(df.columns)}")
        print(f"  [OK]  Date range: {df.index.min().date()} → {df.index.max().date()}")
    except DataLoaderError as e:
        print(f"  [FAIL]  {e}")

    # 2. Metadata
    print(f"\n[2/4] Fetching metadata for {TEST_TICKER}...")
    try:
        meta = get_ticker_metadata(TEST_TICKER)
        print(f"  [OK]  {meta.get('longName', 'N/A')}  |  Sector: {meta.get('sector', 'N/A')}")
    except DataLoaderError as e:
        print(f"  [FAIL]  {e}")

    # 3. Bad ticker
    print("\n[3/4] Testing invalid ticker 'ZZZZXXX123'...")
    try:
        fetch_historical_data("ZZZZXXX123")
        print("  [FAIL]  No exception raised — unexpected!")
    except TickerNotFoundError as e:
        print(f"  [OK]  Correctly raised TickerNotFoundError: {e}")
    except DataLoaderError as e:
        print(f"  [OK]  Caught DataLoaderError: {e}")

    # 4. NewsAPI (only if key is set)
    print("\n[4/4] Fetching news for 'Apple Inc'...")
    try:
        articles = fetch_latest_news("Apple Inc stock", page_size=3)
        for i, article in enumerate(articles, 1):
            print(f"  [{i}] {article.get('title', 'No title')[:80]}")
    except MissingAPIKeyError:
        print("  [WARN]  NEWS_API_KEY not set — skipping news test.")
    except DataLoaderError as e:
        print(f"  [FAIL]  {e}")

    print("\n" + "=" * 72)
    print("  Self-test complete.")
    print("=" * 72)
