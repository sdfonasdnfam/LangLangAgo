"""Local Markdown strategies, selected by explicit keyword matching."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    keywords: list[str]
    instructions: str


def load_skills(directory: Path) -> list[Skill]:
    skills = []
    names = set()
    for path in sorted(directory.glob("*/SKILL.md")):
        try:
            text = path.read_text(encoding="utf-8")
            opening, header, body = text.split("---", 2)
            if opening.strip():
                raise ValueError("文件必须以 YAML front matter 开头")
            meta = yaml.safe_load(header)
            if not isinstance(meta, dict):
                raise ValueError("元数据必须是映射")
            for key in ("name", "description"):
                if not isinstance(meta.get(key), str) or not meta[key].strip():
                    raise ValueError(f"缺少有效的 {key}")
            keywords = meta.get("keywords")
            if (
                not isinstance(keywords, list)
                or not keywords
                or any(not isinstance(k, str) or not k.strip() for k in keywords)
            ):
                raise ValueError("keywords 必须是非空字符串列表")
            if not body.strip():
                raise ValueError("缺少操作策略正文")
            if meta["name"] in names:
                raise ValueError("Skill 名称重复")
            names.add(meta["name"])
            skills.append(Skill(meta["name"], meta["description"], keywords, body.strip()))
        except (ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    return skills


def select_skills(task: str, skills: list[Skill]) -> list[Skill]:
    text = task.casefold()
    return [s for s in skills if any(k.casefold() in text for k in s.keywords)]
