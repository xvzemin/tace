import re
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives


_KEY = re.compile(r"^(?P<indent> *)(?P<key>[^\s:#][^:]*)\s*:")
_COMMENTED_KEY = re.compile(r"^# (?P<indent> *)(?P<key>[^\s:#][^:]*)\s*:")


def _section(lines, key, start, end, parent_indent):
    candidates = []
    for index in range(start, end):
        match = _KEY.match(lines[index])
        if match is None:
            continue
        indent = len(match.group("indent"))
        if indent > parent_indent and match.group("key").strip() == key:
            candidates.append((indent, index))
    if not candidates:
        raise ValueError(f"YAML key {key!r} was not found")

    indent, section_start = min(candidates)
    section_end = end
    for index in range(section_start + 1, end):
        match = _KEY.match(lines[index]) or _COMMENTED_KEY.match(lines[index])
        if match is not None and len(match.group("indent")) <= indent:
            section_end = index
            break
    return section_start, section_end, indent


def _extract_yaml_section(path, yaml_path):
    lines = path.read_text(encoding="utf-8").splitlines()
    keys = yaml_path.split(".")
    start, end, indent = 0, len(lines), -1
    for key in keys:
        start, end, indent = _section(lines, key, start, end, indent)

    selected = lines[start:end]
    while selected and selected[-1].lstrip().startswith("#"):
        selected.pop()
    if selected and not selected[-1].strip():
        selected.pop()
    selected = [line[indent:] if line.strip() else "" for line in selected]
    for key in reversed(keys[:-1]):
        selected = [f"  {line}" if line else "" for line in selected]
        selected.insert(0, f"{key}:")
    return "\n".join(selected).rstrip() + "\n"


class YAMLConfigDirective(Directive):
    required_arguments = 1
    option_spec = {"path": directives.unchanged_required}

    def run(self):
        source = Path(self.state.document.current_source)
        yaml_file = (source.parent / self.arguments[0]).resolve()
        self.state.document.settings.env.note_dependency(str(yaml_file))
        try:
            text = _extract_yaml_section(yaml_file, self.options["path"])
        except (OSError, ValueError) as error:
            raise self.error(str(error)) from error

        block = nodes.literal_block(text, text)
        block["language"] = "yaml"
        return [block]


def setup(app):
    app.add_directive("yaml-config", YAMLConfigDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
