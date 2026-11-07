import unittest

import ffbridge_handicap as handicap


class RuleTests(unittest.TestCase):
    def test_labels_clubs_use(self):
        self.assertEqual(handicap.handicap_rule("OCTOPUS", "2026-10-29"), handicap.PER_PLAYER)
        self.assertEqual(handicap.handicap_rule("Simultané Octopus", "2026-10-29"), handicap.PER_PLAYER)
        self.assertEqual(handicap.handicap_rule("SIMULTANET", "2026-10-30"), handicap.PER_PLAYER)
        self.assertIsNone(handicap.handicap_rule("Atout Simultane", "2026-10-23"))
        self.assertIsNone(handicap.handicap_rule("Rondes de France", "2026-11-03"))

    def test_roy_rene_is_handicap_only_on_the_first_tuesday(self):
        self.assertEqual(handicap.handicap_rule("Roy René", "2026-11-03"), handicap.ROY_RENE)
        self.assertEqual(handicap.handicap_rule("Roy Rene avec handivap", "2026-11-03"), handicap.ROY_RENE)
        self.assertIsNone(handicap.handicap_rule("Roy René", "2026-11-10"))
        self.assertIsNone(handicap.handicap_rule("Roy René", None))


class BonusTests(unittest.TestCase):
    def test_octopus_points_per_player_match_published_bonuses(self):
        # (IVs, bonus FFBridge published) from Octopus 29 Oct 2026, Levallois.
        for ivs, bonus in (
            ([68, 60], 0.0), ([58, 58], 6.0), ([64, 44], 4.0), ([68, 38], 5.0),
            ([56, 48], 7.0), ([40, 34], 10.0), ([32, 24], 14.0), ([26, 22], 15.0),
        ):
            self.assertEqual(handicap.pair_bonus(handicap.PER_PLAYER, ivs), bonus, ivs)

    def test_lowest_ivs_earn_eight_points_each(self):
        self.assertEqual(handicap.pair_bonus(handicap.PER_PLAYER, [20, 20]), 16.0)

    def test_roy_rene_bonus_from_team_iv(self):
        # (IVs, bonus) from Roy René 6 Oct 2026, Levallois.
        for ivs, bonus in (([68, 68], 3.0), ([58, 52], 6.25), ([38, 38], 10.5), ([30, 28], 12.75), ([76, 84], 0.0)):
            self.assertEqual(handicap.pair_bonus(handicap.ROY_RENE, ivs), bonus, ivs)
        self.assertEqual(handicap.pair_bonus(handicap.ROY_RENE, [90, 84]), 0.0)

    def test_a_pair_needs_two_ivs(self):
        with self.assertRaises(ValueError):
            handicap.pair_bonus(handicap.PER_PLAYER, [50])


if __name__ == "__main__":
    unittest.main()
