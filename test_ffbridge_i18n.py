import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import mlBridge.mlBridgePostmortemLib as pm
from mlBridge.mlBridgePostmortemLib import (
    chrome_text,
    language_from_accept_language,
    localized_field,
    standard_report_title,
)
from mlBridge.postmortem_section_help import section_help_for_players

FAVORITES_PATH = Path(__file__).with_name("default.favorites.json")
FAVORITES = json.loads(FAVORITES_PATH.read_text(encoding="utf-8"))
_FIELDS = re.compile(r"{(\w+)}")


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FfbridgeI18nTests(unittest.TestCase):
    def test_browser_language_uses_the_primary_tag(self):
        self.assertEqual(language_from_accept_language(None), "en")
        self.assertEqual(language_from_accept_language("fr-FR,fr;q=0.9,en;q=0.8"), "fr")
        self.assertEqual(language_from_accept_language("en-US,en;q=0.9,fr;q=0.8"), "en")

    def test_english_report_title_stays_when_language_is_unset(self):
        state = SessionState(
            provisional_results=True,
            player_name="Robert SALITA",
            player_license_number="9500754",
        )
        with patch.object(pm.st, "session_state", state):
            self.assertEqual(
                standard_report_title(),
                "Provisional Results: Bridge Game Postmortem Report "
                "Personalized for Robert SALITA (9500754)",
            )

    def test_french_report_uses_the_favorites_catalog(self):
        state = SessionState(
            report_language="fr",
            provisional_results=True,
            player_name="Robert SALITA",
            player_license_number="9500754",
            favorites=FAVORITES,
        )
        with patch.object(pm.st, "session_state", state):
            title = standard_report_title()
            self.assertTrue(title.startswith("Résultats provisoires : "))
            self.assertIn("Personnalisé pour Robert SALITA (9500754)", title)
            played = FAVORITES["SelectBoxes"]["Vetted_Prompts"]["Boards_We_Played"]
            self.assertEqual(
                localized_field(played, "help"),
                played["i18n"]["fr"]["help"],
            )
            self.assertEqual(chrome_text("go_to_top", "Go to top of report"), "Haut du rapport")

    def test_french_text_keeps_the_same_placeholders(self):
        chrome = FAVORITES["Chrome"]
        french = chrome["i18n"]["fr"]
        english_keys = {key for key in chrome if key != "i18n"}
        self.assertEqual(set(french), english_keys)
        for key in english_keys:
            self.assertEqual(
                _FIELDS.findall(chrome[key]),
                _FIELDS.findall(french[key]),
                key,
            )

    def test_section_help_uses_the_player_name_from_favorites(self):
        best = FAVORITES["SelectBoxes"]["Vetted_Prompts"]["Our_Best_Boards"]
        worst = FAVORITES["SelectBoxes"]["Vetted_Prompts"]["Our_Worst_Boards"]
        declared = FAVORITES["SelectBoxes"]["Vetted_Prompts"]["Boards_I_Declared"]
        played = FAVORITES["SelectBoxes"]["Vetted_Prompts"]["Boards_We_Underperformed_DD"]
        self.assertEqual(best["help"], "**Showing our partnership's best boards.**")
        self.assertEqual(worst["help"], "**Showing our partnership's worst boards.**")
        self.assertEqual(
            best["i18n"]["fr"]["help"],
            "**Meilleures donnes de notre paire.**",
        )
        self.assertEqual(
            worst["i18n"]["fr"]["help"],
            "**Plus mauvaises donnes de notre paire.**",
        )
        self.assertIn("{player1_name}", declared["help"])
        self.assertIn("our partnership played", played["help"])
        self.assertNotIn("we played", played["help"])
        self.assertEqual(
            section_help_for_players(declared["help"], "Robert SALITA", "JACOUPY Christian"),
            "**Showing boards which Robert declared.**",
        )
        self.assertEqual(
            section_help_for_players(
                declared["help"], "Jean DUPONT", "Jean MARTIN"
            ),
            "**Showing boards which Jean DUPONT declared.**",
        )
        self.assertEqual(
            section_help_for_players(
                declared["i18n"]["fr"]["help"],
                "Robert SALITA",
                "JACOUPY Christian",
                language="fr",
            ),
            "**Donnes déclarées par Robert.**",
        )

    def test_french_help_matches_the_partnership_voice(self):
        second_person = re.compile(r"\b(votre|vos|vous)\b", re.IGNORECASE)
        prompts = FAVORITES["SelectBoxes"]["Vetted_Prompts"]
        for fav_id, entry in prompts.items():
            english = str(entry.get("help") or "")
            french = str((entry.get("i18n") or {}).get("fr", {}).get("help") or "")
            if not english or not french:
                continue
            shown = section_help_for_players(
                english, "Robert SALITA", "JACOUPY Christian"
            )
            if "our partnership" in shown or re.search(r"\bour\b", shown):
                self.assertIsNone(
                    second_person.search(french),
                    f"{fav_id}: {french}",
                )

    def test_each_favorite_has_french_title_and_help_and_sql_stays_english(self):
        groups = (
            FAVORITES["Buttons"],
            FAVORITES["SelectBoxes"]["Vetted_Prompts"],
        )
        for group in groups:
            for fav_id, entry in group.items():
                french = entry["i18n"]["fr"]
                self.assertTrue(french.get("title"), fav_id)
                if str(entry.get("help") or "").strip():
                    self.assertTrue(french.get("help"), fav_id)
                if str(entry.get("chip") or "").strip():
                    self.assertTrue(french.get("chip"), fav_id)
                for prompt in entry.get("prompts") or []:
                    if not isinstance(prompt, dict):
                        continue
                    sql = prompt.get("sql") or ""
                    if sql.strip():
                        self.assertIn("SELECT", sql.upper(), fav_id)
                        self.assertNotIn("i18n", prompt)


if __name__ == "__main__":
    unittest.main()
