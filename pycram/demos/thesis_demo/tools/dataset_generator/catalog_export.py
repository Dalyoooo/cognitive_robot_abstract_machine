import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from thesis_demo.tools.dataset_generator.catalog import (
    DEFAULT_CATALOG_PATH,
    PACKAGE_DIR,
    SCHEMA_VERSION,
    SemanticCatalog,
    TypeSpec,
)

DEFAULT_POLICY_PATH = PACKAGE_DIR / "assets" / "catalog_policy.json"

_SEMDT_SOURCE_DIR = Path(
    "semantic_digital_twin/src/semantic_digital_twin/semantic_annotations"
)


def default_source_paths():
    source_dir = PACKAGE_DIR.parents[4] / _SEMDT_SOURCE_DIR
    annotations = source_dir / "semantic_annotations.py"
    mixins = source_dir / "mixins.py"
    if annotations.is_file() and mixins.is_file():
        return annotations, mixins
    raise FileNotFoundError(f"semDT annotation sources not found under {source_dir}")


def camel_to_words(type_name):
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", type_name)
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", words)
    return words.replace("_", " ").casefold()


@dataclass(frozen=True)
class _ClassInfo:
    name: str
    parents: tuple[str, ...]
    aliases: tuple[str, ...]


def _base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


def _literal_strings(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return ()
    if not isinstance(value, (set, frozenset, list, tuple)):
        return ()
    return tuple(sorted(str(item) for item in value if isinstance(item, str)))


def _parse_classes(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        parents = []
        for base in node.bases:
            name = _base_name(base)
            if name is not None:
                parents.append(name)

        aliases = ()
        for statement in node.body:
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            defines_synonyms = False
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "_synonyms":
                    defines_synonyms = True
                    break
            if defines_synonyms and statement.value is not None:
                aliases = _literal_strings(statement.value)
        result[node.name] = _ClassInfo(node.name, tuple(parents), aliases)
    return result


def _ancestors(name, graph):
    result = []

    def visit(current, visiting):
        if current in visiting:
            raise ValueError(f"inheritance cycle involving {current}")
        visiting.add(current)
        for parent in graph.get(current, ()):
            if parent not in {"ABC", "object"} and parent not in result:
                result.append(parent)
            if parent in graph:
                visit(parent, visiting)
        visiting.remove(current)

    visit(name, set())
    return tuple(result)


def _load_policy(path):
    with path.open(encoding="utf-8") as policy_file:
        policy = json.load(policy_file)
    required = {
        "roles",
        "capabilities",
        "requires_open_close",
        "storage_endpoints",
    }
    missing = required - set(policy)
    if missing:
        raise ValueError(f"catalog policy is missing keys: {sorted(missing)}")
    return policy


@dataclass(frozen=True)
class _CapabilityRule:
    capabilities: frozenset
    include: frozenset
    exclude: frozenset

    @classmethod
    def from_policy(cls, value):
        return cls(
            capabilities=frozenset(value.get("capabilities", ())),
            include=frozenset(value.get("include", ())),
            exclude=frozenset(value.get("exclude", ())),
        )

    def applies_to(self, name, capabilities):
        matched = name in self.include or bool(
            self.capabilities.intersection(capabilities)
        )
        return matched and name not in self.exclude


@dataclass(frozen=True)
class CatalogPolicy:
    role_precedence: tuple
    role_roots: dict
    role_overrides: dict
    capability_roots: dict
    aliases: dict
    natural_labels: dict
    concept_only: frozenset
    excluded: dict
    open_close: _CapabilityRule
    storage: _CapabilityRule

    @classmethod
    def load(cls, path):
        policy = _load_policy(Path(path))
        roles = policy["roles"]
        return cls(
            role_precedence=tuple(roles["precedence"]),
            role_roots=roles["roots"],
            role_overrides=roles.get("overrides", {}),
            capability_roots=policy["capabilities"],
            aliases=policy.get("aliases", {}),
            natural_labels=policy.get("natural_labels", {}),
            concept_only=frozenset(policy.get("concept_only", ())),
            excluded=policy.get("excluded", {}),
            open_close=_CapabilityRule.from_policy(policy["requires_open_close"]),
            storage=_CapabilityRule.from_policy(policy["storage_endpoints"]),
        )

    def capabilities_for(self, lineage):
        return tuple(
            capability
            for capability, roots in sorted(self.capability_roots.items())
            if lineage.intersection(roots)
        )

    def role_for(self, name, lineage):
        if name in self.role_overrides:
            return self.role_overrides[name]
        for candidate in self.role_precedence:
            if lineage.intersection(self.role_roots.get(candidate, ())):
                return candidate
        return None

    def build_spec(self, name, info, ancestors):
        lineage = {name, *ancestors}
        capabilities = self.capabilities_for(lineage)
        role = self.role_for(name, lineage)
        sampleable = (
            role is not None
            and name not in self.concept_only
            and self.excluded.get(name) is None
        )
        return TypeSpec(
            name=name,
            natural_label=str(self.natural_labels.get(name, camel_to_words(name))),
            aliases=tuple(
                sorted(
                    set(info.aliases).union(
                        str(item) for item in self.aliases.get(name, ())
                    )
                )
            ),
            parents=info.parents,
            ancestors=ancestors,
            capabilities=capabilities,
            role=role,
            sampleable=sampleable,
            requires_open_close=self.open_close.applies_to(name, capabilities),
            can_store_objects=self.storage.applies_to(name, capabilities),
        )


def _resolve_sources(annotations_path, mixins_path):
    if annotations_path is None or mixins_path is None:
        default_annotations, default_mixins = default_source_paths()
        annotations_path = annotations_path or default_annotations
        mixins_path = mixins_path or default_mixins
    return Path(annotations_path), Path(mixins_path)


def build_catalog_data(
    *,
    annotations_path=None,
    mixins_path=None,
    policy_path=None,
):
    annotations, mixins = _resolve_sources(annotations_path, mixins_path)
    policy = CatalogPolicy.load(policy_path or DEFAULT_POLICY_PATH)
    annotation_classes = _parse_classes(annotations)
    all_classes = {**_parse_classes(mixins), **annotation_classes}
    graph = {name: info.parents for name, info in all_classes.items()}
    specs = {
        name: policy.build_spec(name, info, _ancestors(name, graph)).to_dict()
        for name, info in sorted(annotation_classes.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": ["semantic_annotations.py", "mixins.py"],
        "types": specs,
    }


def export_catalog(
    output_path=DEFAULT_CATALOG_PATH,
    *,
    annotations_path=None,
    mixins_path=None,
    policy_path=None,
):
    output = Path(output_path)
    data = build_catalog_data(
        annotations_path=annotations_path,
        mixins_path=mixins_path,
        policy_path=policy_path,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return SemanticCatalog.load(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--mixins", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    args = parser.parse_args()
    catalog = export_catalog(
        args.output,
        annotations_path=args.annotations,
        mixins_path=args.mixins,
        policy_path=args.policy,
    )
    print(f"wrote {len(catalog.types)} types to {args.output}")


if __name__ == "__main__":
    main()
