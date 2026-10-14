import unittest
from datetime import date

import polars as pl

import ffbridge_bridgeplus_postmortem as bridgeplus


FORM = """
<select name='TOURNOI'>
<option selected value='S202719'>&nbsp;Simultané du Roy René n&deg;19 du 13/10/2026&nbsp</option>
<option value='E202701'>&nbsp;École du 12/10/2026&nbsp</option>
<option value='S202718'>&nbsp;Simultané n&deg;18 du 09/10/2026&nbsp</option>
</select>
"""

CLUB = """
<div>09500754-01353276</div>
<a href='https://www.bridgeplus.com/nos-simultanes/resultats/?p=route/S202719/5802079/A/212'>
SALITA Robert - JACOUPY Christian
</a>
<div>01111111-02222222</div>
<a href='https://www.bridgeplus.com/nos-simultanes/resultats/?p=route/S202719/5802079/A/1'>
OTHER Pair
</a>
"""

ROUTE = """
<div>SALITA Robert - JACOUPY Christian (EO212 A)</div>
<div>NS 113 KAMOUN Annie - SENOUSSI Marie-Louise</div>
<div>1</div><div>-140</div><div>376,0 (64,93 %)</div><div>E : 3P=</div><div>♥A</div>
"""

DONNE = """
<div id="donne-board-1" class="donne-board">
<div>Top = 579</div>
<div>Donneur : Nord</div><div>Vulnérable : Personne</div>
<span class="pique"></span><span class="gros">7 2</span>
<span class="coeur"></span><span class="gros">D 10 8 7 3</span>
<span class="carreau"></span><span class="gros">D 5 2</span>
<span class="trefle"></span><span class="gros">A R V</span>
<span class="pique"></span><span class="gros">V 10 6</span>
<span class="coeur"></span><span class="gros">V 9 4 2</span>
<span class="carreau"></span><span class="gros">A 10 8</span>
<span class="trefle"></span><span class="gros">D 7 4</span>
<span class="pique"></span><span class="gros">A R D 8 4</span>
<span class="coeur"></span><span class="gros">5</span>
<span class="carreau"></span><span class="gros">R V 9 7</span>
<span class="trefle"></span><span class="gros">9 6 3</span>
<span class="pique"></span><span class="gros">9 5 3</span>
<span class="coeur"></span><span class="gros">A R 6</span>
<span class="carreau"></span><span class="gros">6 4 3</span>
<span class="trefle"></span><span class="gros">10 8 5 2</span>
</div>
"""


class DisclosureWindowTests(unittest.TestCase):
    def test_ffbridge_redirects_only_inside_the_organizer_window(self):
        website = "https://www.bridgeplus.com/nos-simultanes/"
        session = date(2026, 10, 13)
        self.assertEqual(
            bridgeplus.disclosure_redirect(website, 3, session, date(2026, 10, 14)),
            website,
        )
        self.assertIsNone(
            bridgeplus.disclosure_redirect(website, 3, session, date(2026, 10, 12))
        )
        self.assertIsNone(
            bridgeplus.disclosure_redirect(website, 3, session, date(2026, 10, 16))
        )

    def test_session_id_stays_outside_the_lancelot_range(self):
        self.assertEqual(bridgeplus.bridgeplus_session_id("S202719"), "900202719")


class SheetParserTests(unittest.TestCase):
    def test_tournament_options_keep_roy_rene_dates(self):
        rows = bridgeplus.parse_tournament_options(FORM)
        self.assertEqual(
            rows,
            [
                {"tournament_id": "S202719", "date": "2026-10-13"},
                {"tournament_id": "S202718", "date": "2026-10-09"},
            ],
        )

    def test_club_sheet_finds_the_license_route(self):
        found = bridgeplus.parse_player_route(CLUB, "9500754")
        self.assertEqual(found["license_1"], "9500754")
        self.assertEqual(found["license_2"], "1353276")
        self.assertIn("p=route/S202719/5802079/A/212", found["route_url"])

    def test_route_and_deal_sheets_become_one_board_frame(self):
        frame = bridgeplus.scrape_mldf(
            {
                "route_url": "https://www.bridgeplus.com/x?p=route/S202719/5802079/A/212",
                "date": "2026-10-13",
                "player_id": "246273",
                "player_license_number": "9500754",
                "partner_license_number": "1353276",
                "license_1": "9500754",
                "license_2": "1353276",
                "player_name": "SALITA Robert",
                "partner_name": "JACOUPY Christian",
                "tournament_id": "S202719",
                "organization_id": "5802079",
            },
            route_html=ROUTE,
            donne_html=DONNE,
        )
        row = frame.row(0, named=True)
        self.assertEqual(row["Contract"], "3SE")
        self.assertEqual(row["Result"], 0)
        self.assertEqual(row["Score_NS"], -140)
        self.assertEqual(row["Declarer"], "E")
        self.assertEqual(row["Player_ID_E"], "246273")
        self.assertEqual(row["Player_ID_W"], "1353276")
        self.assertEqual(row["Pair_Direction"], "EW")
        self.assertEqual(row["MP_Top"], 579)
        self.assertAlmostEqual(row["Pct_EW"], 0.6493)
        self.assertTrue(row["PBN"].startswith("N:72.QT873.Q52.AKJ"))
        self.assertEqual(sum(len(part) for part in row["PBN"][2:].replace(" ", ".").split(".")), 52)
        self.assertEqual(frame.schema["Lead"], pl.String)
        self.assertEqual(frame.schema["MP_Top"], pl.Int64)
        self.assertEqual(frame.schema["Score_NS"], pl.Int16)
        self.assertEqual(frame.schema["Score_EW"], pl.Int16)
        self.assertEqual(frame.schema["Player_ID_N"], pl.String)
        self.assertEqual(frame.schema["Player_Name_S"], pl.String)
        prepared = frame.with_columns(
            pl.when(pl.col("Declarer").is_in(["N", "S"]))
            .then(pl.col("Score_NS"))
            .otherwise(pl.col("Score_EW"))
            .alias("Score"),
            pl.concat_list(["Player_ID_N", "Player_ID_S"]).alias("Pair_IDs_NS"),
            pl.concat_list(["Player_Name_N", "Player_Name_S"]).alias("Pair_Names_NS"),
        )
        self.assertEqual(prepared.schema["Score"], pl.Int16)
        self.assertEqual(prepared.schema["Pair_IDs_NS"], pl.List(pl.String))
        self.assertEqual(prepared.schema["Pair_Names_NS"], pl.List(pl.String))


class GapSelectionTests(unittest.TestCase):
    def test_gap_is_the_undisclosed_bridgeplus_event_the_player_played(self):
        pages = {
            bridgeplus.BRIDGEPLUS_FORM_URL: FORM,
            "https://www.bridgeplus.com/nos-simultanes/resultats/?p=club/S202719/5802079": CLUB,
            "https://www.bridgeplus.com/nos-simultanes/resultats/?p=route/S202719/5802079/A/212": ROUTE,
        }

        def fetch(url: str) -> str:
            if url not in pages:
                raise AssertionError(url)
            return pages[url]

        bridgeplus._series_organizers = lambda: [
            {
                "lancelot_series_id": 3,
                "series_id": 5,
                "label": "Roy René",
                "website": "https://www.bridgeplus.com/nos-simultanes/",
                "non_disclosure_days": 3,
            }
        ]
        bridgeplus._lancelot_session_dates = lambda series_id: {"2026-10-09"}
        try:
            rows = bridgeplus.gap_session_entries(
                player_id="246273",
                license_number="9500754",
                sessions=[{"organization_id": "5802079", "organization_name": "Levallois"}],
                today=date(2026, 10, 14),
                fetch=fetch,
            )
        finally:
            # Restore the real lookups for any later test in this process.
            import importlib
            importlib.reload(bridgeplus)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session_id"], "900202719")
        self.assertEqual(rows[0]["date"], "2026-10-13")
        self.assertEqual(rows[0]["source"], "bridgeplus")
        self.assertEqual(rows[0]["player_direction"], "E")
        self.assertNotIn("2026-10-09", [row["date"] for row in rows])


if __name__ == "__main__":
    unittest.main()
