import unittest
from datetime import date
from unittest.mock import patch

import ffbridge_bridgeinter_postmortem as bix
import mlBridge.mlBridgeBILib as bi


def _hand(spades, hearts, diamonds, clubs):
    return (
        f'<td width="34%" class="text1"><img src=images/pique.gif />&nbsp; {spades} <br /> '
        f'<img src=images/coeur.gif />&nbsp; {hearts} <br /> '
        f'<img src=images/carreau.gif />&nbsp; {diamonds} <br /> '
        f'<img src=images/trefle.gif />&nbsp; {clubs}</td>'
    )


# Same layout as donnes.php: North, then West and East on one row, then South.
DEAL_PAGE = (
    '<td class="text1"> Donneur: <!Donneur> Nord</td> '
    '<td align="right" class="text1"> Vulnerabilit&eacute;: <!Vulnerabilite> Personne</td>'
    "<table><tr><td></td>" + _hand("AR872", "9", "V10", "ARDV6") + "<td></td></tr>"
    "<tr>" + _hand("D954", "D10854", "AR8", "3")
    + '<td><img src="images/rosevents.jpg" /></td>'
    + _hand("V6", "ARV76", "D63", "842") + "</tr>"
    "<tr><td></td>" + _hand("103", "32", "97542", "10975") + "<td></td></tr></table>"
    "<tr bgcolor=#CEF376 class=text1><td align=center>420</td><td align=center>&nbsp;</td>"
    "<td align=center>1</td><td align=center>100.0%</td><td align=center> 0.0%</td></tr>"
    "<tr bgcolor=#CEF376 class=text1><td align=center>&nbsp;</td><td align=center>50</td>"
    "<td align=center>3</td><td align=center> 62.3%</td><td align=center> 37.7%</td></tr>"
    "<tr bgcolor=#CEF376 class=text1><td align=center>PASSE</td><td align=center>&nbsp;</td>"
    "<td align=center>1</td><td align=center> 50.0%</td><td align=center> 50.0%</td></tr>"
    "Donne jou&eacute;e 5 fois."
)


def _card_row(board, contract, declarer, lead, result, ns, ew, pct, opponents):
    font = "<td align=center><font face=verdana size=2>{}</font></td>"
    return (
        "<tr>" + "".join(font.format(v) for v in (board, contract, declarer, lead, result))
        + f"<td align=center>{ns or '&nbsp;'}</td><td align=center>{ew or '&nbsp;'}</td>"
        + f"<td align=center> {pct}%</td><td>{opponents or '&nbsp;'}</td></tr>"
    )


SCORECARD = (
    "<tr> <td>Donne</td> </tr>"
    + _card_row(1, "5C", "E", "10T", "-1", "50", "", "27.9", "SININGE Catherine - ZABOTTO Guy")
    + _card_row(2, "4PX", "S", "8P", "=", "590", "", "42.1", "")
    + _card_row(3, "3SA", "N", "4P", "-1", "", "50", "30.9", "AMAR Monique - SUISSA Marie")
    + _card_row(4, "Passe", "", "", "", "", "", "50.0", "")
)


class BridgeInterParsingTests(unittest.TestCase):
    def test_deal_page_hands_frequencies_and_tables(self):
        deal = bi.parse_deal_page(DEAL_PAGE)
        self.assertEqual(deal["dealer"], "N")
        self.assertEqual(deal["vul"], "None")
        self.assertEqual(
            deal["pbn"],
            "N:AK872.9.JT.AKQJ6 J6.AKJ76.Q63.842 T3.32.97542.T975 Q954.QT854.AK8.3",
        )
        self.assertEqual(deal["played"], 5)
        self.assertEqual(deal["frequencies"][2]["score_ns"], 0)  # PASSE
        self.assertEqual(deal["frequencies"][1], {"score_ns": -50, "count": 3, "pct_ns": 62.3, "pct_ew": 37.7})

    def test_scorecard_contracts_leads_scores_and_opponents(self):
        boards = bi.parse_scorecard(SCORECARD)["boards"]
        self.assertEqual(
            [(b["Contract"], b["Result"], b["lead"], b["score_ns"]) for b in boards],
            [("5HE", -1, "TC", 50), ("4SXS", 0, "8S", 590), ("3NN", -1, "4S", -50), ("PASS", None, None, 0)],
        )
        # Opponents are named once per round and carry over to the next board.
        self.assertEqual(boards[1]["opponents"], "SININGE Catherine - ZABOTTO Guy")
        self.assertEqual(boards[0]["pct"], 27.9)

    def test_pair_code(self):
        self.assertEqual(
            bi.parse_pair_code("lo261026LEVA04EO"),
            {"club_code": "LEV", "section": "A", "table": 4, "pair_direction": "EW"},
        )


class BridgeInterSourceTests(unittest.TestCase):
    def test_session_ids_are_numeric_and_above_bridgeplus(self):
        self.assertEqual(bix.bridgeinter_session_id("lo261026"), "922610260")
        self.assertEqual(bix.bridgeinter_session_id("jo261022"), "922610221")
        self.assertGreater(int(bix.bridgeinter_session_id("lo000101")), 910_000_000)

    def test_club_and_player_name_matching(self):
        self.assertTrue(bix.club_matches("Bridge Club Levallois Perret", "Levallois-Perret"))
        self.assertFalse(bix.club_matches("Bridge Club Levallois Perret", "Paris Victoria"))
        rows = [
            {"player1_name": "COLLINS Gilbert", "player2_name": "GELRUBIN Pauline"},
            {"player1_name": "DARMON Robert", "player2_name": "COLLINS Anne"},
        ]
        found = bix.find_pair(rows, "Collins", "Gilbert")
        self.assertEqual((found["player2_name"], found["player_index"]), ("GELRUBIN Pauline", 0))
        self.assertIsNone(bix.find_pair(rows, "Collins", ""))  # two Collins: ambiguous

    def test_build_mldf_uses_the_pairs_percent_and_field_top(self):
        entry = {
            "date": "2026-10-26", "section_name": "A", "pair_direction": "EW", "pair_number": "4",
            "player_id": "56959", "player_name": "COLLINS Gilbert", "player_direction": "E",
            "partner_license_number": None, "partner_name": "GELRUBIN Pauline", "partner_direction": "W",
        }
        card = bi.parse_scorecard(SCORECARD)
        deal = bi.parse_deal_page(DEAL_PAGE)
        deals = {board["board"]: deal for board in card["boards"]}
        df = bix.build_mldf(card, deals, entry=entry, opponent_tables={"SININGE CATHERINE ZABOTTO GUY": 7})
        first = df.row(0, named=True)
        self.assertAlmostEqual(first["Pct_EW"], 0.279)
        self.assertEqual(first["MP_Top"], 4)
        self.assertAlmostEqual(first["MP_EW"], 1.12)
        self.assertEqual((first["Pair_Number_NS"], first["Pair_Number_EW"]), (7, 4))
        self.assertEqual((first["Player_ID_E"], first["Player_Name_W"], first["Player_ID_N"]), ("56959", "GELRUBIN Pauline", None))
        self.assertIsNone(df.row(2, named=True)["Pair_Number_NS"])  # opponent not on the club page

    def test_gap_entries_skip_published_dates_and_other_weekdays(self):
        organizer = {"lancelot_series_id": 27, "series_id": 386, "label": "Simultané Octopus",
                     "website": "http://www.bridgeinter.net/", "non_disclosure_days": 2}
        looked_up = []

        def fake_find(session_date, **kwargs):
            looked_up.append(session_date)
            return {"session_id": "x"}

        with patch.object(bix, "_organizers", return_value=[organizer]), \
                patch.object(bix.bridgeplus, "_lancelot_session_dates", return_value={"2026-10-22"}), \
                patch.object(bix, "_find_entry", side_effect=fake_find):
            rows = bix.gap_session_entries(
                player_id="56959", license_number="1821439", last_name="Collins", first_name="Gilbert",
                sessions=[{"organization_name": "Bridge Club Levallois Perret"}], today=date(2026, 10, 27),
            )
        # Window is Tue 27 and Mon 26; only Monday is an Octopus day.
        self.assertEqual(looked_up, [date(2026, 10, 26)])
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
