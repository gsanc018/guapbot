"""
guapbot/data/kraken_client.py

Kraken REST API client. The ONLY file in the system that calls Kraken REST.
Handles: trades (tick data), ticker, open orders, account balance.
"""
from __future__ import annotations

import time

import requests

from guapbot.utils.config import settings
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

# Kraken public REST base
_BASE = "https://api.kraken.com/0"

# Asset pairs as Kraken expects them
KRAKEN_PAIRS = {
    "XBTUSD": "XXBTZUSD",
    "ETHUSD": "XETHZUSD",
    "ETHBTC": "XETHXXBT",
}


class KrakenRESTError(Exception):
    """Raised when Kraken API returns an error."""


class KrakenClient:
    """
    Thin, stateless wrapper around the Kraken public and private REST API.

    Public methods work without credentials.
    Private methods (balance, orders) require KRAKEN_API_KEY / KRAKEN_API_SECRET
    to be set in .env.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "GuapBot/1.0"})

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def get_trades(self, pair: str, since: int = 0) -> tuple[list, int]:
        """
        Fetch raw tick trades from Kraken GET /public/Trades.

        Args:
            pair:  One of XBTUSD, ETHUSD, ETHBTC.
            since: Cursor from a previous call's returned 'last' value.
                   Pass 0 to start from the beginning of available trades.
                   NOTE: this is a nanosecond timestamp / trade ID, NOT a Unix
                   second timestamp. Treat it as opaque.

        Returns:
            Tuple of (trades_list, last_cursor) where:
              - trades_list: list of raw trade rows from Kraken, each row is
                [price, volume, time, buy_sell, market_limit, misc, trade_id]
              - last_cursor: int to pass as 'since' on the next call.
                If str(last_cursor) == str(since), the feed has stalled — stop.

        Caller is responsible for aggregating tick rows into OHLCV time bars
        using pd.DataFrame.resample().

        Caller is responsible for the pagination loop:
            since = 0
            all_rows = []
            while True:
                rows, last = client.get_trades(pair=pair, since=since)
                if not rows:
                    break
                all_rows.extend(rows)
                if str(last) == str(since):   # stalled
                    break
                since = last
        Then pass all_rows to aggregate_trades_to_ohlcv() in guapbot/data/aggregator.py.
        See DataManager.backfill_trades() for the full implementation.
        """
        kraken_pair = KRAKEN_PAIRS.get(pair, pair)
        params: dict = {"pair": kraken_pair}
        if since:
            params["since"] = since

        logger.debug("Fetching trades %s since=%s", pair, since)
        data = self._public_get("Trades", params)

        pair_key = [k for k in data if k != "last"][0]
        rows = data[pair_key]
        last_cursor = int(data["last"])

        logger.debug("Fetched %d trades for %s, last cursor=%s", len(rows), pair, last_cursor)
        return rows, last_cursor

    def get_ticker(self, pair: str) -> dict:
        """
        Return current ticker for a pair.

        Returns dict with keys: ask, bid, last, volume_24h
        """
        kraken_pair = KRAKEN_PAIRS.get(pair, pair)
        data = self._public_get("Ticker", {"pair": kraken_pair})
        pair_key = list(data.keys())[0]
        t = data[pair_key]
        return {
            "ask": float(t["a"][0]),
            "bid": float(t["b"][0]),
            "last": float(t["c"][0]),
            "volume_24h": float(t["v"][1]),
        }

    def get_server_time(self) -> int:
        """Return Kraken server Unix timestamp."""
        data = self._public_get("Time", {})
        return int(data["unixtime"])

    # ------------------------------------------------------------------
    # Private endpoints (require API key)
    # ------------------------------------------------------------------

    def get_balance(self) -> dict[str, float]:
        """
        Return account balance as {asset: amount}.
        Requires KRAKEN_API_KEY and KRAKEN_API_SECRET in .env.
        """
        import hashlib
        import hmac
        import base64
        import urllib.parse

        nonce = str(int(time.time() * 1000))
        post_data = {"nonce": nonce}
        encoded = urllib.parse.urlencode(post_data)
        message = f"/0/private/Balance{hashlib.sha256((nonce + encoded).encode()).digest()}"

        api_secret = settings.kraken_api_secret
        if not api_secret:
            raise KrakenRESTError("KRAKEN_API_SECRET not set — cannot fetch balance")

        mac = hmac.new(
            base64.b64decode(api_secret),
            f"/0/private/Balance".encode() + hashlib.sha256((nonce + encoded).encode()).digest(),
            hashlib.sha512,
        )
        sig = base64.b64encode(mac.digest()).decode()

        headers = {
            "API-Key": settings.kraken_api_key,
            "API-Sign": sig,
        }
        resp = self._session.post(
            f"{_BASE}/private/Balance",
            data=post_data,
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            raise KrakenRESTError(f"Balance error: {result['error']}")
        return {k: float(v) for k, v in result["result"].items()}

    def get_open_orders(self) -> list[dict]:
        """
        Return list of open orders.
        Simplified — returns raw Kraken order dicts.
        Full order management is in execution/order_manager.py.
        """
        raise NotImplementedError(
            "get_open_orders() — implement in Session 7 (execution layer)"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _public_get(self, endpoint: str, params: dict) -> dict:
        """GET a public Kraken endpoint with retry logic."""
        url = f"{_BASE}/public/{endpoint}"
        for attempt in range(3):
            try:
                resp = self._session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                result = resp.json()
                if result.get("error"):
                    raise KrakenRESTError(f"{endpoint} error: {result['error']}")
                return result["result"]
            except requests.exceptions.RequestException as exc:
                if attempt == 2:
                    raise KrakenRESTError(f"Request failed after 3 attempts: {exc}") from exc
                wait = 2 ** attempt
                logger.warning("Kraken request failed (attempt %d/3), retrying in %ds", attempt + 1, wait)
                time.sleep(wait)
        raise KrakenRESTError("Unreachable")
