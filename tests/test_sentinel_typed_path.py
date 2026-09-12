"""Reject the relabeller skip sentinel before it becomes a positive target.

The former conversion mapped 32002 to 1.0, turning a skipped observation
into an apparently decisive win. Both typed and scalar paths must reject it.
"""

import pytest

from training.labels import SENTINEL_SKIP, SentinelError, cp_to_u, is_sentinel_skip


def test_sentinel_raises_and_does_not_decode():
    assert SENTINEL_SKIP == 32002
    assert is_sentinel_skip(32002)
    with pytest.raises(SentinelError):
        cp_to_u(SENTINEL_SKIP, None)


@pytest.mark.parametrize("calibration_id", [None, "", "sigmoid-400"])
def test_sentinel_raises_on_every_calibration(calibration_id):
    """The guard must not be bypassable by naming a calibration."""
    with pytest.raises(SentinelError):
        cp_to_u(SENTINEL_SKIP, calibration_id)


def test_real_scores_still_decode():
    """The guard must reject ONLY the sentinel, not scores adjacent to it."""
    assert cp_to_u(0, None) == pytest.approx(0.5)
    assert 0.0 <= cp_to_u(30000, None) <= 1.0
    assert 0.0 <= cp_to_u(32001, None) <= 1.0
    assert 0.0 <= cp_to_u(-30000, None) <= 1.0


# --- third path: the encode farm (FR2 blocker 2) ----------------------------
# `build()` filtered sentinels, but `encode_farm.do_datav2` called
# `encode_records` DIRECTLY and did not, producing bound_lo=0.999, wdl=[1,0,0],
# result=2 on a SKIPPED record. The test above could not see that path — which
# is exactly the "one module fixed, the other still live" pattern from ruling 31,
# repeated a third time. The guard now lives in the shared encoder.


def test_encode_records_rejects_sentinel_rows():
    import numpy as np
    from training.f512 import build_datav2 as bd
    from training.f512.rx9records import RECORD_DTYPE

    rec = np.zeros(2, dtype=RECORD_DTYPE)
    rec["score"][0] = 0
    rec["score"][1] = bd.SENTINEL_SKIP
    with pytest.raises(ValueError, match="skip sentinel"):
        bd.encode_records(rec, 1.0)
