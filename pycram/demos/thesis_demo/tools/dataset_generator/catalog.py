import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .domain import unique

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = PACKAGE_DIR / "assets" / "catalog_policy.json"
DEFAULT_CATALOG_PATH = PACKAGE_DIR / "assets" / "catalog.json"

_SEMDT_SOURCE_DIR = Path(
    "semantic_digital_twin/src/semantic_digital_twin/semantic_annotations"
)


def default_source_paths():
    """
    Find the annotation and mixin files in a nearby CRAM checkout.

    Returns:
        A pair containing the semDT annotation and mixin source paths.

    Raises:
        FileNotFoundError: If no nearby CRAM checkout contains both files.
    """
    source_dir = PACKAGE_DIR.parents[4] / _SEMDT_SOURCE_DIR
    annotations = source_dir / "semantic_annotations.py"
    mixins = source_dir / "mixins.py"
    if annotations.is_file() and mixins.is_file():
        return annotations, mixins
    raise FileNotFoundError(f"semDT annotation sources not found under {source_dir}")


def _normalise_label(value):
    """Normalise spelling while retaining meaningful words."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def camel_to_words(type_name):
    """
    Convert a camel-case type name to lowercase words.

    Args:
        type_name: Type name to convert.

    Returns:
        The type name as lowercase, space-separated words.
    """
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", type_name)
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", words)
    return words.replace("_", " ").casefold()


@dataclass(frozen=True)
class TypeSpec:
    """One semDT annotation type as used by the dataset generator."""

    name: str
    qualified_name: str
    natural_label: str
    aliases: tuple[str, ...]
    parents: tuple[str, ...]
    ancestors: tuple[str, ...]
    capabilities: tuple[str, ...]
    role: str | None
    sampleable: bool
    excluded_reason: str | None
    requires_open_close: bool
    can_store_objects: bool

    @classmethod
    def from_dict(cls, value):
        """
        Build a type specification from exported catalog data.

        Args:
            value: Mapping containing one serialized type specification.

        Returns:
            The parsed immutable type specification.

        Raises:
            KeyError: If a required catalog field is missing.
        """
        return cls(
            name=str(value["name"]),
            qualified_name=str(value["qualified_name"]),
            natural_label=str(value["natural_label"]),
            aliases=tuple(str(item) for item in value.get("aliases", ())),
            parents=tuple(str(item) for item in value.get("parents", ())),
            ancestors=tuple(str(item) for item in value.get("ancestors", ())),
            capabilities=tuple(str(item) for item in value.get("capabilities", ())),
            role=value.get("role"),
            sampleable=bool(value.get("sampleable", True)),
            excluded_reason=value.get("excluded_reason"),
            requires_open_close=bool(value.get("requires_open_close", False)),
            can_store_objects=bool(value.get("can_store_objects", False)),
        )

    def to_dict(self):
        """
        Convert the type specification to JSON-compatible data.

        Returns:
            A dictionary containing every exported type field.
        """
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "natural_label": self.natural_label,
            "aliases": list(self.aliases),
            "parents": list(self.parents),
            "ancestors": list(self.ancestors),
            "capabilities": list(self.capabilities),
            "role": self.role,
            "sampleable": self.sampleable,
            "excluded_reason": self.excluded_reason,
            "requires_open_close": self.requires_open_close,
            "can_store_objects": self.can_store_objects,
        }


@dataclass(frozen=True)
class SemanticCatalog:
    """Read-only query interface over an exported semDT catalog."""

    types: Mapping[str, TypeSpec]
    schema_version: int = 1

    @classmethod
    def load(cls, path):
        """
        Load an exported semantic catalog from JSON.

        Args:
            path: Path to the exported catalog file.

        Returns:
            The loaded semantic catalog.

        Raises:
            OSError: If the catalog file cannot be read.
            json.JSONDecodeError: If the catalog file is not valid JSON.
        """
        with Path(path).open(encoding="utf-8") as catalog_file:
            value = json.load(catalog_file)
        specs = {
            name: TypeSpec.from_dict(spec)
            for name, spec in sorted(value.get("types", {}).items())
        }
        return cls(
            types=specs,
            schema_version=int(value.get("schema_version", 1)),
        )

    def __contains__(self, type_name):
        return type_name in self.types

    def __getitem__(self, type_name):
        try:
            return self.types[type_name]
        except KeyError as error:
            raise KeyError(f"unknown semDT type: {type_name}") from error

    def type_names_for_role(self, role, *, include_unsampleable=False):
        """
        Return deterministic type names for a planner role.

        Args:
            role: Planner role whose types should be returned.
            include_unsampleable: Whether to include policy-excluded types.

        Returns:
            A sorted tuple of matching semDT type names.
        """
        return tuple(
            name
            for name, spec in sorted(self.types.items())
            if spec.role == role and (include_unsampleable or spec.sampleable)
        )

    def natural_label(self, type_name):
        """
        Return the configured natural-language label for a type.

        Args:
            type_name: semDT type name to look up.

        Returns:
            The natural-language label for the type.

        Raises:
            KeyError: If the type is not present in the catalog.
        """
        return self[type_name].natural_label

    def labels_for(self, type_name, include_ancestors=True):
        """
        Return labels and aliases for a type and optionally its ancestors.

        Args:
            type_name: semDT type name to describe.
            include_ancestors: Whether to include labels from ancestor types.

        Returns:
            An order-preserving tuple of natural labels and aliases.

        Raises:
            KeyError: If the type is not present in the catalog.
        """
        spec = self[type_name]
        labels = [spec.natural_label, *spec.aliases]
        if include_ancestors:
            for ancestor in spec.ancestors:
                ancestor_spec = self.types.get(ancestor)
                if ancestor_spec is not None:
                    labels.extend((ancestor_spec.natural_label, *ancestor_spec.aliases))
        return unique(label for label in labels if label)

    def reference_labels(self, type_name):
        """
        Return labels suitable for generated user-facing references.

        Args:
            type_name: semDT type name to describe.

        Returns:
            Labels and aliases defined directly for the requested type.

        Raises:
            KeyError: If the type is not present in the catalog.
        """
        return self.labels_for(type_name, include_ancestors=False)

    def matches(self, type_name, label):
        """
        Check whether a label names a type or one of its ancestors.

        Args:
            type_name: semDT type name to compare.
            label: Natural-language label to match.

        Returns:
            True if the normalized label matches the type hierarchy.

        Raises:
            KeyError: If the type is not present in the catalog.
        """
        wanted = _normalise_label(label)
        return any(
            _normalise_label(candidate) == wanted
            for candidate in self.labels_for(type_name, include_ancestors=True)
        )

    def has_capability(self, type_name, capability):
        """Return whether the type has the capability (used by catalog tests)."""
        return capability in self[type_name].capabilities

    def requires_open_close(self, type_name):
        """
        Check whether accessing a type requires open and close actions.

        Args:
            type_name: semDT type name to inspect.

        Returns:
            True if generated access plans must open and close the type.

        Raises:
            KeyError: If the type is not present in the catalog.
        """
        return self[type_name].requires_open_close

    def can_store_objects(self, type_name):
        """
        Check whether a type is a valid inside-relation endpoint.

        Args:
            type_name: semDT type name to inspect.

        Returns:
            True if instances of the type can store objects.

        Raises:
            KeyError: If the type is not present in the catalog.
        """
        return self[type_name].can_store_objects


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
    """Return parents before grandparents, with deterministic de-duplication."""
    result = []

    def visit(current, visiting):
        """
        Add one class and its parents to the ancestor result.

        Args:
            current: Class name whose parents should be visited.
            visiting: Class names in the active traversal path.

        Raises:
            ValueError: If the inheritance graph contains a cycle.
        """
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


def build_catalog_data(
    *,
    annotations_path=None,
    mixins_path=None,
    policy_path=None,
):
    """
    Build deterministic catalog data from semDT source files.

    Args:
        annotations_path: Optional semantic-annotation source path.
        mixins_path: Optional semantic-annotation mixin source path.
        policy_path: Optional dataset-generator catalog policy path.

    Returns:
        JSON-compatible catalog data containing every annotation type.

    Raises:
        FileNotFoundError: If required source or policy files cannot be found.
        ValueError: If the policy or inheritance graph is invalid.
    """
    if annotations_path is None or mixins_path is None:
        default_annotations, default_mixins = default_source_paths()
        annotations_path = annotations_path or default_annotations
        mixins_path = mixins_path or default_mixins
    annotations = Path(annotations_path)
    mixins = Path(mixins_path)
    selected_policy_path = Path(policy_path or DEFAULT_POLICY_PATH)

    policy = _load_policy(selected_policy_path)
    annotation_classes = _parse_classes(annotations)
    mixin_classes = _parse_classes(mixins)
    all_classes = {**mixin_classes, **annotation_classes}
    graph = {name: info.parents for name, info in all_classes.items()}

    role_policy = policy["roles"]
    role_precedence = tuple(role_policy["precedence"])
    role_roots = role_policy["roots"]
    role_overrides = role_policy.get("overrides", {})
    capability_roots = policy["capabilities"]
    aliases = policy.get("aliases", {})
    natural_labels = policy.get("natural_labels", {})
    concept_only = set(policy.get("concept_only", ()))
    excluded = policy.get("excluded", {})
    open_close_policy = policy["requires_open_close"]
    open_close_capabilities = set(open_close_policy.get("capabilities", ()))
    open_close_include = set(open_close_policy.get("include", ()))
    open_close_exclude = set(open_close_policy.get("exclude", ()))
    storage_policy = policy["storage_endpoints"]
    storage_capabilities = set(storage_policy.get("capabilities", ()))
    storage_include = set(storage_policy.get("include", ()))
    storage_exclude = set(storage_policy.get("exclude", ()))

    specs = {}
    for name, info in sorted(annotation_classes.items()):
        ancestors = _ancestors(name, graph)
        lineage = {name, *ancestors}
        capabilities = tuple(
            capability
            for capability, roots in sorted(capability_roots.items())
            if lineage.intersection(roots)
        )

        if name in role_overrides:
            role = role_overrides[name]
        else:
            role = None
            for candidate in role_precedence:
                roots = role_roots.get(candidate, ())
                if lineage.intersection(roots):
                    role = candidate
                    break

        excluded_reason = excluded.get(name)
        sampleable = (
            role is not None and name not in concept_only and excluded_reason is None
        )
        type_aliases = sorted(
            set(info.aliases).union(str(item) for item in aliases.get(name, ()))
        )
        requires_open_close = (
            name in open_close_include
            or bool(open_close_capabilities.intersection(capabilities))
        ) and name not in open_close_exclude
        can_store_objects = (
            name in storage_include
            or bool(storage_capabilities.intersection(capabilities))
        ) and name not in storage_exclude

        if excluded_reason is not None:
            saved_excluded_reason = str(excluded_reason)
        elif name in concept_only:
            saved_excluded_reason = "concept-only type"
        else:
            saved_excluded_reason = None

        spec = TypeSpec(
            name=name,
            qualified_name=(
                "semantic_digital_twin.semantic_annotations."
                f"semantic_annotations.{name}"
            ),
            natural_label=str(natural_labels.get(name, camel_to_words(name))),
            aliases=tuple(type_aliases),
            parents=info.parents,
            ancestors=ancestors,
            capabilities=capabilities,
            role=role,
            sampleable=sampleable,
            excluded_reason=saved_excluded_reason,
            requires_open_close=requires_open_close,
            can_store_objects=can_store_objects,
        )
        specs[name] = spec.to_dict()

    return {
        "schema_version": 1,
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
    """
    Export catalog data and load its query interface.

    Args:
        output_path: Path where the catalog JSON should be written.
        annotations_path: Optional semantic-annotation source path.
        mixins_path: Optional semantic-annotation mixin source path.
        policy_path: Optional dataset-generator catalog policy path.

    Returns:
        The exported catalog as a semantic query interface.

    Raises:
        OSError: If source files cannot be read or output cannot be written.
        ValueError: If the policy or inheritance graph is invalid.
    """
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


def _main():
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
    _main()
