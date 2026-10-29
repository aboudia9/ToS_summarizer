"""Reads text out of an uploaded file (tos_summarizer/extract.py) in a separate process.

PDF, Word and HTML parsers are big, and malformed files are a classic way to make a
parser hang or eat all the memory. So each upload is parsed in a fresh process with a
CPU-time and memory cap, which is killed if it runs long. A bad file costs a few seconds
and can't touch the server or the models.
"""
import logging
import multiprocessing as mp
import threading

from tos_summarizer.extract import ExtractionError

log = logging.getLogger(__name__)

TIME_LIMIT_SECONDS = 15
MEMORY_LIMIT_BYTES = 1_500_000_000
MAX_CONCURRENT = 2

_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


class Busy(Exception):
    pass


def _parse(conn, data, filename):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (TIME_LIMIT_SECONDS, TIME_LIMIT_SECONDS))
    except (ImportError, ValueError, OSError):
        pass  # macOS doesn't enforce these; the parent's timeout still applies

    from tos_summarizer.extract import extract_text
    try:
        conn.send(("ok", extract_text(data, filename)))
    except ExtractionError as e:
        conn.send(("error", str(e)))
    except MemoryError:
        conn.send(("error", "That file is too complex to read."))
    except Exception as e:
        conn.send(("error", "That file couldn't be read.", repr(e)))


def extract_in_sandbox(data, filename):
    if not _slots.acquire(blocking=False):
        raise Busy()
    try:
        ctx = mp.get_context("spawn")   # a clean interpreter: no models, no server state
        receiver, sender = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_parse, args=(sender, data, filename), daemon=True)
        proc.start()
        sender.close()
        try:
            if not receiver.poll(TIME_LIMIT_SECONDS + 3):
                raise ExtractionError("That file took too long to read.")
            reply = receiver.recv()
        except EOFError:  # killed for going over its CPU or memory limit
            raise ExtractionError("That file is too complex to read.")
        finally:
            if proc.is_alive():
                proc.kill()
            proc.join()
    finally:
        _slots.release()

    if reply[0] != "ok":
        if len(reply) > 2:
            log.warning("Couldn't parse %s: %s", filename.rsplit(".", 1)[-1], reply[2])
        raise ExtractionError(reply[1])
    return reply[1]
