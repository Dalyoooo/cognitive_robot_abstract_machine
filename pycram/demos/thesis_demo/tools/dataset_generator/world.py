import json
import re
from dataclasses import dataclass
from pathlib import Path

from thesis_demo.tools.dataset_generator.domain import (
    AREAS,
    POSITIONS,
    WorldContext,
    normalise_words,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_HOLDOUT_PATH = PACKAGE_DIR / "assets" / "holdout.json"


def _compact_text(value):
    return "".join(normalise_words(value))


def _contains_term(text, term):
    words = re.findall(r"[a-z0-9]+", term.casefold())
    if not words:
        return False
    body = r"[\s_-]+".join(map(re.escape, words))
    return re.search(rf"(?<![a-z0-9]){body}(?![a-z0-9])", text.casefold()) is not None


@dataclass(frozen=True)
class HoldoutPolicy:
    objects: frozenset[str]
    instances: frozenset[str]
    types: frozenset[str]
    terms: frozenset[str]

    @classmethod
    def load(cls, path=DEFAULT_HOLDOUT_PATH):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            objects=frozenset(map(str, value.get("objects", ()))),
            instances=frozenset(map(str, value.get("instances", ()))),
            types=frozenset(map(str, value.get("types", ()))),
            terms=frozenset(map(str, value.get("terms", ()))),
        )

    def forbidden_terms(self, catalog):
        labels = set(self.terms) | set(self.objects)
        for type_name in self.types:
            if type_name in catalog:
                labels.update(catalog.labels_for(type_name, include_ancestors=False))
        return frozenset(labels)

    def allows_type(self, type_name):
        return type_name not in self.types

    def assert_clean(self, context, catalog):
        names = set(context.objects) | set(context.places)
        leaked_names = names & (set(self.objects) | set(self.instances))
        if leaked_names:
            raise ValueError(
                f"evaluation names leaked into training: {sorted(leaked_names)}"
            )
        leaked_types = set(context.types.values()) & set(self.types)
        if leaked_types:
            raise ValueError(
                f"evaluation types leaked into training: {sorted(leaked_types)}"
            )
        context_terms = {
            _compact_text(value) for value in (*names, *context.types.values())
        }
        context_terms.discard("")
        leaked_terms = {
            term
            for term in self.forbidden_terms(catalog)
            if _compact_text(term) in context_terms
        }
        if leaked_terms:
            raise ValueError(
                f"evaluation terms leaked into training: {sorted(leaked_terms)}"
            )

    def assert_text_clean(self, text, catalog):
        forbidden = (
            set(self.objects) | set(self.instances) | set(self.forbidden_terms(catalog))
        )
        leaked = {term for term in forbidden if _contains_term(text, term)}
        if leaked:
            raise ValueError(f"evaluation terms leaked into language: {sorted(leaked)}")


def _safe_token(label):
    token = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    return token or "entity"


def _assert_unprefixed_names(context):
    prefixed = sorted(name for name in context.objects + context.places if "/" in name)
    if prefixed:
        raise ValueError(f"generated planner names must not contain '/': {prefixed}")


class WorldComposer:
    def __init__(self, catalog, holdout):
        self.catalog = catalog
        self.holdout = holdout

    def _available_types(self, role):
        return [
            name
            for name in self.catalog.type_names_for_role(role)
            if self.holdout.allows_type(name)
        ]

    def _select_types(
        self,
        role,
        count,
        rng,
        *,
        repeated_type_count=1,
    ):
        candidates = self._available_types(role)
        if count and not candidates:
            raise ValueError(f"semDT catalog has no sampleable {role} types")

        if repeated_type_count >= 2 and count >= 2:
            duplicate_candidates = candidates
            if role == "container":
                duplicate_candidates = [
                    name
                    for name in candidates
                    if self.catalog.can_store_objects(name)
                    and self.catalog.requires_open_close(name)
                ]
            duplicated = rng.choice(duplicate_candidates)
            remaining = [name for name in candidates if name != duplicated]
            rng.shuffle(remaining)
            repetitions = min(count, repeated_type_count)
            selected = [duplicated] * repetitions
            selected.extend(remaining[: count - repetitions])
        else:
            selected = list(candidates)
            rng.shuffle(selected)
            selected = selected[:count]

        while len(selected) < count:
            selected.append(rng.choice(candidates))
        rng.shuffle(selected)
        return tuple(selected)

    def _place_rows(self, type_names, rng, used):
        rows = []
        for type_name in type_names:
            token = _safe_token(self.catalog.natural_label(type_name))
            base = f"{rng.choice(AREAS)}_{rng.choice(POSITIONS)}_{token}"
            name = self._unique_name(base, used)
            rows.append((name, type_name))
        return rows

    def _simple_rows(self, type_names, used):
        rows = []
        for type_name in type_names:
            label = self.catalog.natural_label(type_name)
            name = self._unique_name(_safe_token(label), used)
            rows.append((name, type_name))
        return rows

    def _unique_name(self, base, used):
        name = base
        suffix = 2
        forbidden = set(self.holdout.instances) | set(self.holdout.objects)
        while name in used or name in forbidden:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        return name

    def compose(
        self,
        rng,
        *,
        object_count=10,
        surface_count=10,
        container_count=12,
        furniture_count=5,
        room_count=6,
        duplicate_count=2,
    ):
        used = set()
        object_types = self._select_types("object", object_count, rng)
        repeated_type_count = max(2, duplicate_count)
        surface_types = self._select_types(
            "surface",
            surface_count,
            rng,
            repeated_type_count=repeated_type_count,
        )
        container_types = self._select_types(
            "container",
            container_count,
            rng,
            repeated_type_count=repeated_type_count,
        )
        furniture_types = self._select_types("furniture", furniture_count, rng)
        room_types = self._select_types("room", room_count, rng)

        objects = self._simple_rows(object_types, used)
        surfaces = self._place_rows(surface_types, rng, used)
        containers = self._place_rows(container_types, rng, used)
        furniture = self._place_rows(furniture_types, rng, used)
        rooms = self._simple_rows(room_types, used)

        surface_names = [name for name, _type in surfaces]
        storage_names = [
            name
            for name, type_name in containers
            if self.catalog.can_store_objects(type_name)
        ]
        openable_names = [
            name
            for name, type_name in containers
            if self.catalog.requires_open_close(type_name)
        ]
        object_places = surface_names + storage_names
        if not object_places:
            raise ValueError(
                "world composition requires a surface or storage container"
            )

        object_locations = {}
        for object_name, _type_name in objects:
            object_locations[object_name] = (rng.choice(object_places),)

        all_rows = objects + surfaces + containers + furniture + rooms
        types = {name: type_name for name, type_name in all_rows}
        result = WorldContext(
            objects=tuple(name for name, _type in objects),
            object_locations=object_locations,
            surfaces=tuple(name for name, _type in surfaces),
            containers=tuple(name for name, _type in containers),
            openables=tuple(openable_names),
            furniture=tuple(name for name, _type in furniture),
            rooms=tuple(name for name, _type in rooms),
            types=types,
        )
        result.validate()
        _assert_unprefixed_names(result)
        self.holdout.assert_clean(result, self.catalog)
        return result
