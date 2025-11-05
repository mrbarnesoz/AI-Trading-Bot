"""Execution routing against BitMEX REST API."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str  # "Buy" or "Sell"
    quantity: float
    order_type: str = "Market"
    price: Optional[float] = None
    reduce_only: bool = False
    post_only: bool = False
    time_in_force: str | None = None
    client_order_id: str | None = None


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    status: str
    filled_qty: float
    raw_response: Dict[str, Any]


class ExecutionRouter:
    """Abstract execution router."""

    def submit_order(self, order: OrderRequest) -> OrderResult:
        raise NotImplementedError

    def cancel_all(self, symbol: str | None = None) -> None:
        raise NotImplementedError


class BitmexExecutionRouter(ExecutionRouter):
    """REST execution adapter for BitMEX."""

    MAINNET_ENDPOINT = "https://www.bitmex.com"
    TESTNET_ENDPOINT = "https://testnet.bitmex.com"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        testnet: bool = False,
        dry_run: bool = True,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.TESTNET_ENDPOINT if testnet else self.MAINNET_ENDPOINT
        self.dry_run = dry_run or not (api_key and api_secret)
        self._client = httpx.Client(timeout=timeout)

        if self.dry_run:
            logger.warning("BitMEX execution router running in dry-run mode. No orders will be sent.")

    def submit_order(self, order: OrderRequest) -> OrderResult:
        if self.dry_run:
            fake_id = f"dry-{int(time.time() * 1000)}"
            logger.info("[DryRun] submit_order %s", order)
            return OrderResult(order_id=fake_id, status="accepted", filled_qty=0.0, raw_response={})

        payload: Dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side.capitalize(),
            "orderQty": order.quantity,
            "ordType": order.order_type.capitalize(),
            "reduceOnly": order.reduce_only,
            "execInst": "ParticipateDoNotInitiate" if order.post_only else None,
        }
        if order.price is not None:
            payload["price"] = order.price
        if order.time_in_force:
            payload["timeInForce"] = order.time_in_force
        if order.client_order_id:
            payload["clOrdID"] = order.client_order_id

        response = self._signed_post("/api/v1/order", payload)
        return OrderResult(
            order_id=response.get("orderID", ""),
            status=response.get("ordStatus", "unknown"),
            filled_qty=float(response.get("cumQty", 0.0)),
            raw_response=response,
        )

    def cancel_all(self, symbol: str | None = None) -> None:
        if self.dry_run:
            logger.info("[DryRun] cancel_all symbol=%s", symbol)
            return
        payload: Dict[str, Any] = {}
        if symbol:
            payload["symbol"] = symbol
        self._signed_delete("/api/v1/order/all", payload)

    # ------------------------------------------------------------------#
    # HTTP helpers
    # ------------------------------------------------------------------#
    def _signed_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._signed_request("POST", path, payload)

    def _signed_delete(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._signed_request("DELETE", path, payload)

    def _signed_request(self, method: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not (self.api_key and self.api_secret):
            raise RuntimeError("BitMEX API credentials must be provided for live trading.")
        url = f"{self.base_url}{path}"
        body = json.dumps(payload)
        expires = int(time.time()) + 30
        signature = self._generate_signature(method, path, expires, body)
        headers = {
            "api-key": self.api_key,
            "api-expires": str(expires),
            "api-signature": signature,
            "Content-Type": "application/json",
        }
        response = self._client.request(method, url, headers=headers, content=body)
        response.raise_for_status()
        return response.json()

    def _generate_signature(self, method: str, path: str, expires: int, body: str) -> str:
        message = f"{method.upper()}{path}{expires}{body}".encode("utf-8")
        secret = self.api_secret.encode("utf-8") if self.api_secret else b""
        digest = hmac.new(secret, message, hashlib.sha256).digest()
        return digest.hex()

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort clean-up
        try:
            self.close()
        except Exception:
            pass
