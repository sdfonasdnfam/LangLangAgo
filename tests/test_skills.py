import tempfile
import unittest
from pathlib import Path

from browser_agent.skills import load_skills, select_skills


class SkillTests(unittest.TestCase):
    def test_matching_loads_multiple_strategies(self):
        skills = load_skills(Path("skills"))
        selected = select_skills("搜索并提取商品价格", skills)
        self.assertEqual({s.name for s in selected}, {"search", "extract"})
        self.assertIn("搜索关键词", next(s.instructions for s in selected if s.name == "search"))

    def test_unmatched_task_uses_general_strategy(self):
        self.assertEqual(select_skills("打开首页", load_skills(Path("skills"))), [])

    def test_keywords_are_case_insensitive(self):
        selected = select_skills("SEARCH documentation", load_skills(Path("skills")))
        self.assertEqual([s.name for s in selected], ["search"])

    def test_malformed_skill_reports_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "broken"
            folder.mkdir()
            (folder / "SKILL.md").write_text("---\nname: broken\n---\nbody", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SKILL.md"):
                load_skills(Path(directory))


if __name__ == "__main__":
    unittest.main()
