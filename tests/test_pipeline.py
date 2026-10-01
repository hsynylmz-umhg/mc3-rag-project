"""CPU-only contract tests. These do not claim to validate model accuracy."""
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter" / "app"))
import app
from protocol import atomic_json, receive, rpc, send
from retriever import build_index, Retriever, tokens, identifiers
from server import Service


class FakeEncoder:
    """Deterministic lexical vectors isolate retrieval logic from a real model."""
    dimension = 64

    def encode(self, texts, **kwargs):
        import numpy as np
        output = np.zeros((len(texts), self.dimension), dtype="float32")
        for i, text in enumerate(texts):
            for word in tokens(text):
                output[i, hashlib.sha256(word.encode()).digest()[0] % self.dimension] += 1
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        return output / np.maximum(norms, 1)


class RetrievalTests(unittest.TestCase):
    def test_identifier_sentence_punctuation_is_not_part_of_join_key(self):
        self.assertEqual(identifiers("Incident logged against ABC-7392."), ["abc-7392"])
        self.assertEqual(identifiers("ticket: ABC-7392 | fixed: 1.2"), ["abc-7392"])

    def test_persisted_two_hop_retrieval_and_exact_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            corpus = directory / "corpus"
            corpus.mkdir()
            units = [
                {"path": "logs/production.log", "location": "line 1", "text": "Thermal throttle incident. Incident tracked as ABC-7392."},
                {"path": "support/bugs.csv", "location": "row 2", "text": "ticket: ABC-7392 | symptom: ADC high | fixed_in: 7.8.9"},
            ]
            units += [{"path": f"distractors/{i}.txt", "location": "1", "text": "Firmware release thermal throttle production details unknown."} for i in range(20)]
            encoder = FakeEncoder()
            build_index(directory, corpus, units, encoder, generation="test")
            retriever = Retriever(directory, encoder)
            try:
                # Identifier follow-up must find the row absent from question terms.
                # Force the target out of direct retrieval so this cannot pass
                # accidentally just because the synthetic dense encoder found it.
                retriever._lexical = lambda question, limit=50: [0]
                mapped_vectors = retriever.vectors
                retriever.vectors = mapped_vectors.copy()
                mapped_vectors._mmap.close()
                retriever.vectors[:] = 1.0
                retriever.vectors[1] = 0.0
                found = retriever.retrieve("The production log shows a thermal throttle incident. Which firmware release fixed the defect?")
                paths = {c["path"] for c in found}
                self.assertIn("logs/production.log", paths)
                self.assertIn("support/bugs.csv", paths)
                self.assertEqual(retriever.manifest["corpus"], str(corpus.resolve()))
                self.assertTrue(all(c["id"] in retriever.chunks for c in found))
            finally:
                retriever.close()

    def test_empty_index_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            build_index(directory, directory, [], FakeEncoder(), generation="empty")
            retriever = Retriever(directory, FakeEncoder())
            try:
                self.assertEqual(retriever.retrieve("anything"), [])
            finally:
                retriever.close()

    def test_service_rejects_wrong_corpus_and_never_reparses_query(self):
        service = Service(None, None, Path.cwd())
        with self.assertRaises(RuntimeError):
            service.handle({"op": "answer", "query": "x"})


class ContractTests(unittest.TestCase):
    def test_fresh_process_writes_refusal_without_loading_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = {**os.environ, "MC3_OUTPUT_DIR": temporary, "MC3_SOCKET": str(Path(temporary) / "absent.sock")}
            run = subprocess.run([sys.executable, str(ROOT / "starter/app/app.py"), "--corpus", temporary,
                                  "--query-id", "query_09", "--query", "Unavailable fact"],
                                 env=env, capture_output=True, timeout=4)
            self.assertEqual(run.returncode, 0, run.stderr)
            payload = json.loads((Path(temporary) / "query_09_output.json").read_text())
            self.assertEqual(payload, {"answer": "", "citations": [], "confidence": 0.0})
            imports = subprocess.run([sys.executable, "-c", "import app,sys; assert not any(x in sys.modules for x in ('torch','transformers','parser','retriever','numpy'))"],
                                     cwd=ROOT / "starter/app", capture_output=True, timeout=4)
            self.assertEqual(imports.returncode, 0, imports.stderr)

    def test_answer_normalizes_citations_and_rejects_bad_result(self):
        with patch.object(app, "rpc", return_value={"answer": "7.8.9", "citations": ["a.csv", "a.csv"], "confidence": 2}):
            self.assertEqual(app.answer(Path.cwd(), "x"), ("7.8.9", ["a.csv"], 1.0))
        with patch.object(app, "rpc", return_value={"answer": "x", "citations": ["../x"], "confidence": 0.9}):
            with self.assertLogs("mc3", level="ERROR"):
                self.assertEqual(app.answer(Path.cwd(), "x"), ("", [], 0.0))

    def test_atomic_output_replaces_previous_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "x.json"
            atomic_json(output, {"answer": "old"})
            atomic_json(output, {"answer": "new"})
            self.assertEqual(json.loads(output.read_text()), {"answer": "new"})
            self.assertEqual(len(list(Path(temporary).iterdir())), 1)

    @unittest.skipUnless(hasattr(socket, "AF_UNIX"), "Unix sockets unavailable")
    def test_slow_socket_has_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            sock_path = str(Path(temporary) / "slow.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(sock_path)
                listener.listen(1)
                def slow():
                    connection, _ = listener.accept()
                    with connection:
                        receive(connection, time.monotonic() + 1)
                        time.sleep(0.35)
                worker = threading.Thread(target=slow)
                worker.start()
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    rpc(sock_path, {"op": "answer"}, timeout=0.08)
                self.assertLess(time.monotonic() - start, 0.3)
                worker.join()

    def test_oversize_ipc_frame_rejected(self):
        first, second = socket.socketpair()
        with first, second:
            first.sendall(struct.pack("!I", 2000000))
            with self.assertRaises(ValueError):
                receive(second, time.monotonic() + 1)

    def test_fragmented_response_cannot_extend_absolute_deadline(self):
        first, second = socket.socketpair()
        with first, second:
            def drip():
                try:
                    for byte in struct.pack("!I", 10) + b'{"a":true}':
                        first.send(bytes([byte]))
                        time.sleep(0.02)
                except OSError:
                    pass
            thread = threading.Thread(target=drip)
            thread.start()
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                receive(second, started + 0.09)
            self.assertLess(time.monotonic() - started, 0.25)
            thread.join()


if __name__ == "__main__":
    unittest.main()
