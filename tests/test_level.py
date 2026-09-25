from karaokifex.level import MOST, QUIET_FLOOR, gains, lift


def test_a_loud_enough_song_is_left_as_it_is():
    assert lift(-9.5) == 0.0
    assert lift(-13.0, -16.0) == 0.0
    assert lift(QUIET_FLOOR) == 0.0
    assert lift(None) == 0.0


def test_a_quiet_song_is_lifted_to_the_floor_but_never_by_more_than_the_most():
    assert lift(-17.0) == 5.0
    assert lift(-21.0, -16.0) == 5.0
    assert lift(-40.0) == MOST


def test_matched_the_karaoke_comes_to_the_song_and_both_are_lifted_alike():
    original, karaoke = gains(-18.0, -21.0, match=True, lift_quiet=True)
    assert original == 6.0
    assert karaoke == 3.0 + 6.0          # to the song, then the same lift


def test_unmatched_each_is_lifted_from_its_own_level():
    assert gains(-18.0, -21.0, match=False, lift_quiet=True) == (6.0, 9.0)
    assert gains(-8.0, -10.0, match=False, lift_quiet=True) == (0.0, 0.0)


def test_without_the_lift_only_the_match_counts():
    assert gains(-22.0, -25.0, match=True, lift_quiet=False) == (0.0, 3.0)
    assert gains(-22.0, None, match=True, lift_quiet=False) == (0.0, 0.0)
