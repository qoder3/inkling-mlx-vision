"""Unit tests for FairQueue FIFO ordering and client disconnect cancellation."""

from __future__ import annotations

import threading
import time

import pytest
from runtime_mlx import server as _server

from inkling_mlx_vision.queue import FairQueue


def test_fair_queue_fifo_order():
    q = FairQueue()
    order: list[int] = []
    t1, p1 = q.enter()
    assert p1 == 1
    t2, p2 = q.enter()
    t3, p3 = q.enter()
    assert (p2, p3) == (2, 3)

    def worker(ticket: int):
        q.wait_turn(ticket, lambda: False)
        order.append(ticket)
        q.leave(ticket)

    th2 = threading.Thread(target=worker, args=(t2,))
    th3 = threading.Thread(target=worker, args=(t3,))
    th2.start()
    th3.start()
    time.sleep(0.05)
    order.append(t1)
    q.leave(t1)
    th2.join(timeout=2)
    th3.join(timeout=2)
    assert order == [t1, t2, t3]


def test_fair_queue_cancels_on_disconnect():
    q = FairQueue()
    t1, _ = q.enter()
    t2, _ = q.enter()
    gone = threading.Event()
    errors: list[Exception] = []

    def waiter():
        try:
            q.wait_turn(t2, gone.is_set)
        except Exception as e:
            errors.append(e)

    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.05)
    gone.set()
    th.join(timeout=2)
    assert len(errors) == 1 and isinstance(errors[0], _server.ClientGone)
    q.leave(t1)
