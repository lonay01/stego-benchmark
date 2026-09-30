"""BCH postprocessing only; no network, source image or true message in decoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import time
import zipfile
import json

from authornight.storage import EvidenceError, cached, commit, digest, freeze, read, write


class CorrectedBCH:
    """Author packet: 56 data bits + 40 ECC storage bits + 4 zero padding bits."""

    def __init__(self):
        import bchlib
        self.bch = bchlib.BCH(5, 137)
        if (self.bch.ecc_bytes, self.bch.ecc_bits) != (5, 35):
            raise EvidenceError("Unexpected BCH parameters")

    @staticmethod
    def bits(data):
        return [int(bit) for byte in data for bit in f"{byte:08b}"]

    def encode(self, data_bits):
        if len(data_bits) != 56 or any(bit not in (0, 1) for bit in data_bits):
            raise ValueError("Expected 56 binary data positions")
        data = bytearray(int("".join(map(str, data_bits[i:i+8])), 2) for i in range(0, 56, 8))
        return self.bits(data + self.bch.encode(data)) + [0] * 4

    def decode(self, predicted):
        if len(predicted) != 100 or any(bit not in (0, 1) for bit in predicted):
            raise ValueError("Expected 100 predicted binary positions")
        # The trailing four author padding bits are not part of the BCH codeword.
        packet = bytearray(int("".join(map(str, predicted[i:i+8])), 2) for i in range(0, 96, 8))
        data, parity = packet[:7], packet[7:]
        corrected_errors = self.bch.decode(data, parity)
        if corrected_errors < 0:
            return {"accepted": False, "corrected_errors": corrected_errors, "data_bits": None}
        self.bch.correct(data, parity)  # decode() alone does not mutate the received bytes
        return {"accepted": True, "corrected_errors": corrected_errors, "data_bits": self.bits(data)}


def selftest():
    codec, rng = CorrectedBCH(), random.Random(20260930)
    counts = {str(n): 0 for n in range(6)}
    for _ in range(200):
        data = [rng.randrange(2) for _ in range(56)]
        encoded = codec.encode(data)
        for errors in range(6):
            received = list(encoded)
            for position in rng.sample(range(91), errors):
                received[position] ^= 1
            decoded = codec.decode(received)
            if not decoded["accepted"] or decoded["data_bits"] != data:
                raise EvidenceError(f"BCH self-test failed with {errors} flipped bits")
            counts[str(errors)] += 1
    # Exercise each stored position, including the unused parity and packet padding.
    data = [rng.randrange(2) for _ in range(56)]
    encoded = codec.encode(data)
    for position in range(100):
        received = list(encoded)
        received[position] ^= 1
        if codec.decode(received)["data_bits"] != data:
            raise EvidenceError(f"BCH single-bit test failed at position {position}")
    return {"passed": True, "random_trials": counts, "single_position_trials": 100,
            "total_trials": 1300, "guaranteed_radius": 5,
            "note": "More than five errors may be rejected OR miscorrected; acceptance is not proof of exact recovery."}


def evaluate(row, codec):
    expected = row["payload"][:56]
    if codec.encode(expected) != row["payload"]:
        raise EvidenceError("Corrected decoder does not match the original author packet encoding")
    stages = {}
    for stage in ("raw_internal", "author_png"):
        old = row["measurements"][stage]
        decoded = codec.decode(old["bits"])
        exact = bool(decoded["accepted"] and decoded["data_bits"] == expected)
        stages[stage] = {**decoded, "data_exact": exact,
                         "accepted_but_wrong": bool(decoded["accepted"] and not exact),
                         "data_errors": sum(a != b for a, b in zip(decoded["data_bits"], expected)) if decoded["accepted"] else None,
                         "legacy_text_exact": old.get("author_ecc_text_exact"),
                         "raw_payload_ber": old["ber"], "raw_payload_exact": old["exact"]}
    return {"source_record_sha256": row["record_sha256"], "image_id": row["image_id"],
            "repetition": row["repetition"], "stages": stages,
            "note": "Only BCH postprocessing changed; saved logits/bits/PNGs and full-payload BER are unchanged."}


def run(root, deadline=float("inf")):
    root = Path(root)
    plan = read(root / "plan.json")
    freeze(root / "_meta/recovery/bch_selftest.json", selftest())
    codec = CorrectedBCH()
    for path in sorted((root / "sepwater/coco/records").glob("*_author_bch56.json")):
        if time.time() >= deadline:
            return
        row = cached(root, path.relative_to(root).as_posix(), plan["identity"], verify=False)
        relative = f"sepwater/ecc_corrected/{path.stem}.json"
        if cached(root, relative, plan["identity"]):
            continue
        result = evaluate(row, codec)
        commit(root, relative, plan["identity"], result)
    print("SepWater BCH replay complete; original measurements have not changed.", flush=True)


def replay_report_zip(archive_path):
    """Read-only CPU smoke on the actual uploaded evidence; no GPU or model download."""
    tests = selftest()
    codec = CorrectedBCH()
    totals = {stage: {"cases": 0, "data_exact": 0, "rejected": 0, "accepted_but_wrong": 0,
                      "legacy_text_exact": 0} for stage in ("raw_internal", "author_png")}
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("sepwater/coco/records/") or not name.endswith("_author_bch56.json"):
                continue
            row = json.loads(archive.read(name))
            if digest({k: v for k, v in row.items() if k != "record_sha256"}) != row["record_sha256"]:
                raise EvidenceError(f"Modified source record: {name}")
            for stage, score in evaluate(row, codec)["stages"].items():
                totals[stage]["cases"] += 1
                totals[stage]["data_exact"] += int(score["data_exact"])
                totals[stage]["rejected"] += int(not score["accepted"])
                totals[stage]["accepted_but_wrong"] += int(score["accepted_but_wrong"])
                totals[stage]["legacy_text_exact"] += int(score["legacy_text_exact"])
    return {"scope": "Actual saved SepWater bits; corrected BCH only; no GPU inference", "selftest": tests, "totals": totals}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-zip", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = replay_report_zip(args.report_zip)
    write(Path(args.output), result)
    print(json.dumps(result, indent=2))
