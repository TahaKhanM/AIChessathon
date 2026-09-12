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


def test_batch_rejects_sentinel_rows():
    """A mixed batch must not turn a skipped observation into a win target."""
    import chess

    from training.features import FeatureEncoder
    from training.records import make_observation
    from training.train import build_batch

    def record(score):
        return {
            "record_id": f"synthetic-score-{score}",
            "position": {"fen4": " ".join(chess.STARTING_FEN.split()[:4])},
            "observations": [
                make_observation(
                    kind="searched", perspective="side_to_move", score_kind="cp", cp=score
                )
            ],
        }

    encoder = FeatureEncoder()
    ordinary = record(0)
    assert build_batch([ordinary], encoder).u_targets == [[0.5]]
    with pytest.raises(SentinelError, match="skip sentinel"):
        build_batch([ordinary, record(SENTINEL_SKIP)], encoder)
