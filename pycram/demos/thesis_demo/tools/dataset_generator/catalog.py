import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .domain import unique

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = PACKAGE_DIR / "assets" / "catalog.json"

SCHEMA_VERSION = 2


class CatalogSchemaError(Exception):
    pass


def _normalise_label(value):
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


@dataclass(frozen=True)
class TypeSpec:

    name: str
    natural_label: str
    aliases: tuple[str, ...]
    parents: tuple[str, ...]
    ancestors: tuple[str, ...]
    capabilities: tuple[str, ...]
    role: str | None
    sampleable: bool
    requires_open_close: bool
    can_store_objects: bool

    @classmethod
    def from_dict(cls, value):
        return cls(
            name=str(value["name"]),
            natural_label=str(value["natural_label"]),
            aliases=tuple(str(item) for item in value.get("aliases", ())),
            parents=tuple(str(item) for item in value.get("parents", ())),
            ancestors=tuple(str(item) for item in value.get("ancestors", ())),
            capabilities=tuple(str(item) for item in value.get("capabilities", ())),
            role=value.get("role"),
            sampleable=bool(value.get("sampleable", True)),
            requires_open_close=bool(value.get("requires_open_close", False)),
            can_store_objects=bool(value.get("can_store_objects", False)),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "natural_label": self.natural_label,
            "aliases": list(self.aliases),
            "parents": list(self.parents),
            "ancestors": list(self.ancestors),
            "capabilities": list(self.capabilities),
            "role": self.role,
            "sampleable": self.sampleable,
            "requires_open_close": self.requires_open_close,
            "can_store_objects": self.can_store_objects,
        }


@dataclass(frozen=True)
class SemanticCatalog:

    types: Mapping[str, TypeSpec]

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as catalog_file:
            value = json.load(catalog_file)
        schema_version = int(value.get("schema_version", 1))
        if schema_version != SCHEMA_VERSION:
            raise CatalogSchemaError(
                f"{path} has schema_version {schema_version}, "
                f"expected {SCHEMA_VERSION}. Re-export it with: "
                f"python -m thesis_demo.tools.dataset_generator.catalog_export"
            )
        specs = {
            name: TypeSpec.from_dict(spec)
            for name, spec in sorted(value.get("types", {}).items())
        }
        return cls(types=specs)

    def __contains__(self, type_name):
        return type_name in self.types

    def __getitem__(self, type_name):
        try:
            return self.types[type_name]
        except KeyError as error:
            raise KeyError(f"unknown semDT type: {type_name}") from error

    def type_names_for_role(self, role, *, include_unsampleable=False):
        return tuple(
            name
            for name, spec in sorted(self.types.items())
            if spec.role == role and (include_unsampleable or spec.sampleable)
        )

    def natural_label(self, type_name):
        return self[type_name].natural_label

    def labels_for(self, type_name, include_ancestors=True):
        spec = self[type_name]
        labels = [spec.natural_label, *spec.aliases]
        if include_ancestors:
            for ancestor in spec.ancestors:
                ancestor_spec = self.types.get(ancestor)
                if ancestor_spec is not None:
                    labels.extend((ancestor_spec.natural_label, *ancestor_spec.aliases))
        return unique(label for label in labels if label)

    def reference_labels(self, type_name):
        return self.labels_for(type_name, include_ancestors=False)

    def matches(self, type_name, label):
        wanted = _normalise_label(label)
        return any(
            _normalise_label(candidate) == wanted
            for candidate in self.labels_for(type_name, include_ancestors=True)
        )


    def requires_open_close(self, type_name):
        return self[type_name].requires_open_close

    def can_store_objects(self, type_name):
        return self[type_name].can_store_objects
