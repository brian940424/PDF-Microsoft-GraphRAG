"""Generate questions aimed at one GraphRAG retrieval strategy each (F-67).

The old automatic generator sampled one page per question, so every question was a single-page
Basic lookup, and the manual medium／cross_section modes paired unrelated pages into "What is A?
And B?" questions. Here related source text is found first and then handed to a strategy-specific
prompt (conditions and counter-examples from docs/07_出題指引_四種檢索策略.md):

- Basic: one page, or two consecutive pages of the same section (`section_path`).
- Local: two or three entities joined by GraphRAG relationships, with the source pages of the
  relationships and of each entity; may span documents.
- DRIFT: a chain of "Refer to XX-n" cross references (two or three hops) whose pages are in
  different documents or not adjacent; the question must not reveal the intermediate conclusion.
- Global: the pages where one recurring heading appears across manuals. Global reference answers
  can only cover the supplied pages, so these questions need human review.

Every generated question carries `design_strategy` and the `answer_scope` inferred from its answer
sources (F-66).
"""

from __future__ import annotations

import inspect
import json
import math
import random
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cancellation import JobStopped, raise_if_stopped
from .connections import ConnectionSettings
from .cross_references import page_label_matches, referenced_pages
from .projects import ProjectError, ProjectStore
from .question_generation import QuestionGenerationService, normalize_question
from .question_sets import DESIGN_STRATEGIES, DESIGN_STRATEGY_LABELS, GoldEvidence, infer_answer_scope
from .source_sampling import SourceSample


# Default mix from the question guideline (Basic 20, Local 8, DRIFT 7, Global 5 of 40 questions).
DEFAULT_STRATEGY_MIX: dict[str, float] = {"basic": 50.0, "local": 20.0, "drift": 17.5, "global": 12.5}
MIN_SOURCE_CHARACTERS = 150
MAX_SOURCE_CHARACTERS = 1500
BUNDLES_PER_CALL = 3
# Entities mentioned in more text units than this are hubs ("NISSAN", "SRS"); relationships through
# them are generic, so they are not used for Local questions (SM3E: 99% of entities are in <= 8 units).
HUB_TEXT_UNITS = 8
MAX_LOCAL_SAMPLES = 4
MAX_GLOBAL_SAMPLES = 5
# Entities named after work steps or events ("BRAKE BLEEDING", "BASIC INSPECTION", "COLLISION") are
# what GraphRAG extracts most often from service manuals; the model skips most of them for Local, so
# relationships between named parts and functions are tried first.
PROCESS_WORDS = re.compile(
    r"\b(INSPECTION|INSPECT|CHECK|CHECKING|PROCEDURE|WORKFLOW|REPAIR|REPLACEMENT|REPLACE|INSTALLATION|INSTALL|"
    r"REMOVAL|REMOVE|DIAGNOSIS|DIAGNOSTIC|BLEEDING|DRAINING|REFILLING|FILLING|ADJUSTMENT|CONFIRMATION|"
    r"COLLISION|INJURY|DEPLOYMENT|INFLATION|MAINTENANCE|SERVICE|OPERATION|WORK|PRECAUTION|MANUAL)\b"
)
HEADING_PATTERN = re.compile(r"^[A-Z][A-Z0-9 ()/&,'\"“”\-–.:]{15,89}$")


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """Related source text for one question of one strategy; `hint` explains why the pages belong together."""

    bundle_id: str
    strategy: str
    samples: tuple[SourceSample, ...]
    hint: str = ""


@dataclass(frozen=True, slots=True)
class StrategyQuestion:
    question: str
    reference_answer: str
    design_strategy: str
    answer_scope: str
    question_source_evidence: tuple[GoldEvidence, ...]
    answer_source_evidence: tuple[GoldEvidence, ...]
    bundle_id: str
    hint: str = ""
    reasoning_chain: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()


@dataclass(slots=True)
class StrategyGenerationResult:
    questions: list[StrategyQuestion] = field(default_factory=list)
    quotas: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


def normalize_strategy_mix(mix: Mapping[str, float] | None) -> dict[str, float]:
    """Validate a strategy → share mapping (any positive scale); missing strategies count as 0."""
    values = dict(DEFAULT_STRATEGY_MIX if mix is None else mix)
    unknown = set(values) - set(DESIGN_STRATEGIES)
    if unknown:
        raise ProjectError("不支援的出題策略：" + "、".join(sorted(unknown)))
    result = {}
    for strategy in DESIGN_STRATEGIES:
        value = values.get(strategy, 0)
        try:
            number = float(value or 0)
        except (TypeError, ValueError) as exc:
            raise ProjectError(f"{DESIGN_STRATEGY_LABELS[strategy]} 配比必須是數字") from exc
        if not math.isfinite(number) or number < 0:
            raise ProjectError(f"{DESIGN_STRATEGY_LABELS[strategy]} 配比不可小於 0")
        result[strategy] = number
    if not any(result.values()):
        raise ProjectError("策略配比至少要有一項大於 0")
    return result


def allocate_quotas(total: int, mix: Mapping[str, float]) -> dict[str, int]:
    """Split `total` questions by `mix` with the largest-remainder method; ties go to the earlier strategy."""
    shares = normalize_strategy_mix(mix)
    weight = sum(shares.values())
    exact = {strategy: total * share / weight for strategy, share in shares.items()}
    quotas = {strategy: math.floor(value) for strategy, value in exact.items()}
    order = sorted(DESIGN_STRATEGIES, key=lambda item: (-(exact[item] - quotas[item]), DESIGN_STRATEGIES.index(item)))
    for strategy in order[: total - sum(quotas.values())]:
        quotas[strategy] += 1
    return quotas


class StrategyBundleBuilder:
    """Find related source text for each strategy from processed pages and the GraphRAG index."""

    def __init__(
        self,
        projects: ProjectStore,
        label_matches: Callable[[str, str, int], bool] | None = None,
    ) -> None:
        self.projects = projects
        self._label_matches = label_matches or self._pdf_label_matches

    def build(
        self, project_id: str, strategy: str, samples: Sequence[SourceSample], seed: int = 0,
    ) -> list[SourceBundle]:
        usable = [item for item in samples if item.character_count >= MIN_SOURCE_CHARACTERS]
        rng = random.Random(f"{seed}-{strategy}")
        builder = {
            "basic": self._basic, "local": self._local, "drift": self._drift, "global": self._global,
        }[strategy]
        return builder(project_id, usable, rng)

    # Basic ---------------------------------------------------------------------------------
    def _basic(self, project_id: str, samples: list[SourceSample], rng: random.Random) -> list[SourceBundle]:
        singles = [SourceBundle(f"basic-{item.sample_id}", "basic", (item,)) for item in samples]
        by_page = self._best_by_page(samples)
        pairs = []
        for (document_id, page), first in sorted(by_page.items()):
            second = by_page.get((document_id, page + 1))
            if second is not None and (first.section_path or first.section_id) == (second.section_path or second.section_id):
                pairs.append(SourceBundle(
                    f"basic-{first.sample_id}+{second.page}", "basic", (first, second),
                    f"Consecutive pages {page} and {page + 1} of {document_id}, section {first.section_path or first.section_id}.",
                ))
        rng.shuffle(singles)
        rng.shuffle(pairs)
        # About a third cross-page, as in the guideline's Basic mix (14 single-page, 6 cross-page).
        mixed: list[SourceBundle] = []
        while singles or pairs:
            mixed.extend(singles[:2])
            del singles[:2]
            mixed.extend(pairs[:1])
            del pairs[:1]
        return self._spread_by_document(mixed)

    # Local ---------------------------------------------------------------------------------
    def _local(self, project_id: str, samples: list[SourceSample], rng: random.Random) -> list[SourceBundle]:
        graph = self._graph(project_id)
        if graph is None:
            return []
        entities, relationships, chunk_by_unit = graph
        sample_by_chunk = {item.chunk_id: item for item in samples}

        def chunks(unit_ids: Sequence[str]) -> list[SourceSample]:
            found = [sample_by_chunk.get(chunk_by_unit.get(str(unit), "")) for unit in unit_ids]
            return list(dict.fromkeys(item for item in found if item is not None))

        # Hubs and places or people (dealer addresses, plants) give no component relationship.
        hub = {title for title, (units, kind) in entities.items() if len(units) > HUB_TEXT_UNITS or kind in {"GEO", "PERSON"}}
        entities = {title: units for title, (units, _kind) in entities.items()}
        neighbours: dict[str, list[dict[str, Any]]] = {}
        for item in relationships:
            neighbours.setdefault(item["source"], []).append(item)
            neighbours.setdefault(item["target"], []).append(item)
        candidates = [
            item for item in relationships
            if item["source"] not in hub and item["target"] not in hub and item["source"] != item["target"]
        ]
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: -float(item.get("weight") or 0))
        bundles = []
        used_pairs: set[frozenset[str]] = set()
        for relationship in candidates:
            pair = frozenset((relationship["source"], relationship["target"]))
            if pair in used_pairs:
                continue
            selected = chunks(relationship["text_unit_ids"])[:2]
            for title in (relationship["source"], relationship["target"]):
                for sample in chunks(entities.get(title, ())):
                    if sample not in selected and all(sample.page != item.page or sample.document_id != item.document_id
                                                      for item in selected):
                        selected.append(sample)
                        break
            names = [relationship["source"], relationship["target"]]
            relations = [relationship]
            third = next((
                item for title in names for item in neighbours.get(title, ())
                if item is not relationship and not ({item["source"], item["target"]} & hub)
                and {item["source"], item["target"]} - set(names)
            ), None)
            if third is not None and len(selected) < MAX_LOCAL_SAMPLES:
                extra = chunks(third["text_unit_ids"])
                if extra:
                    names.append(next(iter({third["source"], third["target"]} - set(names[:2]))))
                    relations.append(third)
                    if extra[0] not in selected:
                        selected.append(extra[0])
            selected = selected[:MAX_LOCAL_SAMPLES]
            if not selected:
                continue
            used_pairs.add(pair)
            hint = "Entities: " + ", ".join(names) + ". Graph relationships (summaries, not source text): " + " | ".join(
                f"{item['source']} — {item['target']}: {str(item.get('description') or '')[:300]}" for item in relations
            )
            bundles.append(SourceBundle(f"local-{len(bundles) + 1}", "local", tuple(selected), hint))
        # Relationships whose sources span pages first: a single page usually holds one procedure,
        # which makes a Basic question (the guideline's Local mix is 2 single-page of 8).
        tiers: dict[tuple[bool, bool], list[SourceBundle]] = {}
        for bundle in bundles:
            tiers.setdefault((
                bool(PROCESS_WORDS.search(bundle.hint.split(". Graph relationships", 1)[0])),
                len({(item.document_id, item.page) for item in bundle.samples}) < 2,
            ), []).append(bundle)
        return [item for key in sorted(tiers) for item in self._spread_by_document(tiers[key])]

    def _graph(self, project_id: str):
        """Entities (title → (text-unit ids, type)), relationships and text-unit → chunk map; None before indexing."""
        output = self.projects.path_for(project_id) / "graphrag" / "output"
        metadata = output / "source_metadata.jsonl"
        if not (output / "entities.parquet").exists() or not (output / "relationships.parquet").exists() or not metadata.exists():
            return None
        import pandas as pd

        entity_frame = pd.read_parquet(output / "entities.parquet")
        relationship_frame = pd.read_parquet(
            output / "relationships.parquet", columns=["source", "target", "description", "weight", "text_unit_ids"],
        )
        kinds = entity_frame["type"] if "type" in entity_frame else [""] * len(entity_frame)
        entities = {
            str(title): ([str(unit) for unit in (units if units is not None else [])], str(kind or "").upper())
            for title, units, kind in zip(entity_frame["title"], entity_frame["text_unit_ids"], kinds)
        }
        relationships = [
            {
                "source": str(row.source), "target": str(row.target), "description": row.description,
                "weight": row.weight,
                "text_unit_ids": [str(unit) for unit in (row.text_unit_ids if row.text_unit_ids is not None else [])],
            }
            for row in relationship_frame.itertuples()
        ]
        chunk_by_unit = {}
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                chunk_by_unit[str(item["text_unit_id"])] = str(item.get("chunk_id", ""))
        return entities, relationships, chunk_by_unit

    # DRIFT ---------------------------------------------------------------------------------
    def _drift(self, project_id: str, samples: list[SourceSample], rng: random.Random) -> list[SourceBundle]:
        by_page = self._best_by_page(samples)
        documents = {item.document_id for item in samples}
        label_cache: dict[tuple[str, int], bool] = {}

        def hops(sample: SourceSample, visited: set[tuple[str, int]]) -> list[SourceSample]:
            found = []
            for location in referenced_pages([sample.text], documents):
                if location in visited or location not in by_page:
                    continue
                if location not in label_cache:
                    label_cache[location] = self._label_matches(project_id, *location)
                if label_cache[location]:
                    found.append(by_page[location])
            return found

        chains: list[tuple[SourceSample, ...]] = []
        for start in samples:
            origin = {(start.document_id, start.page)}
            for first in hops(start, origin):
                if not self._far_apart(start, first):
                    continue
                second = next(iter(hops(first, origin | {(first.document_id, first.page)})), None)
                chains.append((start, first, second) if second is not None else (start, first))
        rng.shuffle(chains)
        # Cross-document chains first, then three-hop chains.
        chains.sort(key=lambda chain: (len({item.document_id for item in chain}) < 2, len(chain) < 3))
        bundles, seen = [], set()
        for chain in chains:
            # Many pages refer to the same procedure; keep one chain per referenced tail so the
            # questions do not all end on the same pages.
            key = tuple((item.document_id, item.page) for item in chain[1:])
            if key in seen:
                continue
            seen.add(key)
            hint = "Reference chain: " + " → ".join(f"{item.document_id} p.{item.page}" for item in chain) + (
                ". Each page refers to the next one with \"Refer to XX-n\"."
            )
            bundles.append(SourceBundle(f"drift-{len(bundles) + 1}", "drift", chain, hint))
        return bundles

    @staticmethod
    def _far_apart(left: SourceSample, right: SourceSample) -> bool:
        return left.document_id != right.document_id or abs(left.page - right.page) > 1

    def _pdf_label_matches(self, project_id: str, document_id: str, page: int) -> bool:
        pdf = self.projects.path_for(project_id) / "source" / document_id
        return page_label_matches(pdf, page, f"{Path(document_id).stem}-{page}")

    # Global --------------------------------------------------------------------------------
    def _global(self, project_id: str, samples: list[SourceSample], rng: random.Random) -> list[SourceBundle]:
        occurrences: dict[str, dict[str, list[SourceSample]]] = {}
        for sample in samples:
            for heading in self._headings(sample.text):
                occurrences.setdefault(heading, {}).setdefault(sample.document_id, []).append(sample)
        documents = {item.document_id for item in samples}
        bundles, seen = [], set()
        ranked = sorted(occurrences.items(), key=lambda item: (-len(item[1]), item[0]))
        for heading, by_document in ranked:
            if len(documents) > 1:
                if len(by_document) < 2:
                    continue
                selected = [pages[0] for pages in by_document.values()]
            else:
                pages = sorted(next(iter(by_document.values())), key=lambda item: item.page)
                # Within one manual: at least three non-adjacent pages share the heading.
                selected = [item for index, item in enumerate(pages) if index == 0 or item.page - pages[index - 1].page > 1]
                if len(selected) < 3:
                    continue
            if len(selected) > MAX_GLOBAL_SAMPLES:
                selected = rng.sample(selected, MAX_GLOBAL_SAMPLES)
            selected.sort(key=lambda item: (item.document_id.casefold(), item.page))
            key = frozenset(item.sample_id for item in selected)
            if key in seen:
                continue
            seen.add(key)
            hint = f"Recurring heading \"{heading}\" appears in " + ", ".join(
                f"{document} ({len(pages)} page(s))" for document, pages in sorted(by_document.items())
            ) + "; one occurrence per document is supplied."
            bundles.append(SourceBundle(f"global-{len(bundles) + 1}", "global", tuple(selected), hint))
        return bundles

    @staticmethod
    def _headings(text: str) -> set[str]:
        headings = set()
        for line in text.splitlines():
            value = " ".join(line.split())
            if "...." in value or "INFOID" in value or not HEADING_PATTERN.match(value):
                continue
            # Three or more words: "INSTALLATION" or "REMOVAL AND INSTALLATION" are on most pages and
            # give no topic to synthesise, while "PRECAUTIONS WHEN USING POWER TOOLS ..." does.
            if len(value.split()) >= 4:
                headings.add(value.rstrip(":"))
        return headings

    # Shared --------------------------------------------------------------------------------
    @staticmethod
    def _best_by_page(samples: Sequence[SourceSample]) -> dict[tuple[str, int], SourceSample]:
        """The longest sample of each page stands for that page."""
        result: dict[tuple[str, int], SourceSample] = {}
        for item in samples:
            key = (item.document_id, item.page)
            if key not in result or item.character_count > result[key].character_count:
                result[key] = item
        return result

    @staticmethod
    def _spread_by_document(bundles: Sequence[SourceBundle]) -> list[SourceBundle]:
        """Round-robin over the first document of each bundle so every PDF gets questions early."""
        queues: dict[str, list[SourceBundle]] = {}
        for bundle in bundles:
            queues.setdefault(bundle.samples[0].document_id, []).append(bundle)
        result = []
        while any(queues.values()):
            for document in sorted(queues, key=str.casefold):
                if queues[document]:
                    result.append(queues[document].pop(0))
        return result


COMMON_INSTRUCTIONS = (
    "You write evaluation questions for a benchmark that compares GraphRAG retrieval strategies over "
    "automotive service manuals. Each question targets one strategy. Use only the supplied source text and "
    "do not add outside knowledge.\n"
    "Write every question and reference answer in Traditional Chinese. Part names, abbreviations, DTCs, units, "
    "and OEM terminology may stay in English.\n"
    "Ask in a natural user voice; do not copy section titles or source sentences verbatim.\n"
    "Each question asks about one thing. Never join two unrelated sub-questions such as \"What is A? And what "
    "is B?\".\n"
    "Reference answers must be complete: keep every required step in order, preconditions, values with units "
    "and conditions, cautions and warnings; when the source differs by model or engine, name the model in the "
    "question or list each variant in the answer.\n"
)

STRATEGY_INSTRUCTIONS = {
    "basic": (
        "Strategy: Basic (single-point lookup). Basic retrieves the passages most similar to the question and "
        "answers from them.\n"
        "Conditions (all required): the whole answer is in one section of one document, on one page or two "
        "adjacent pages; it asks for concrete content such as procedure steps, a setting path, a specification "
        "value, a warning or caution, a DTC meaning, a definition, or applicable conditions; it has one clear "
        "answer that needs no inference.\n"
        "When a bundle has two consecutive pages, cite both only if the same topic continues across them; "
        "otherwise write the question from one page.\n"
        "Not Basic: an answer that needs another chapter (DRIFT); \"what types are there overall\" (Global).\n"
        "Counter-example: \"When replacing brake fluid, what are the refill steps?\" has several steps but they "
        "are all in one section, so it is Basic, not DRIFT.\n"
        "Optional difficulty: when the page has similar items (front and rear brakes, two engines), name which "
        "one the question asks about.\n"
    ),
    "local": (
        "Strategy: Local (entity relationship). Local finds the entities in the question and expands along "
        "their graph relationships.\n"
        "Conditions (all required): the question names two or three specific components, functions, signals or "
        "systems (use the entities in the hint), such as parts, control units, switches, sensors, or operating "
        "functions; it asks about their relationship, such as dependency, "
        "exclusion, influence, interaction, sharing, or a prerequisite; the answer draws on what the sources "
        "say about each entity, not on one block of steps.\n"
        "The reference answer states each entity's role in the relationship and the conditions under which the "
        "relationship holds. The hint's relationship summaries only point to the topic; every fact in the "
        "answer must come from the source text.\n"
        "Not Local: one entity with its procedure or values (Basic); a multi-step causal chain (DRIFT).\n"
        "Counter-example: \"What are the maximum runout and thickness variation of the front disc?\" reads two "
        "values from one specification table, so it is Basic, not Local.\n"
        "Skip the bundle when the hint's entities are events, document sections, workflows, or general concepts "
        "(for example COLLISION, BASIC INSPECTION) rather than named components or functions, or when the sources "
        "do not describe how the entities relate.\n"
        "Return the entity names the question is about in \"entities\".\n"
    ),
    "drift": (
        "Strategy: DRIFT (multi-step reasoning). DRIFT builds context from community reports, then asks "
        "follow-up questions and searches locally round by round.\n"
        "The sources form a reference chain across sections (see the hint). Conditions (all required): the "
        "question starts from a situation the user faces; answering it needs an intermediate conclusion from "
        "one source that decides where to look next, and the final answer comes from another source; no single "
        "source answers it completely.\n"
        "Never state the intermediate conclusion in the question, including the referenced component, procedure "
        "or page; otherwise the question becomes a single lookup. The question may describe only the starting "
        "situation from the first source of the chain. For example, if the first page says \"remove the brake "
        "pedal, refer to BR-18\", the question must not mention the brake pedal; ask what else must be done or "
        "checked after the starting task instead.\n"
        "The reference answer follows the reasoning order: intermediate conclusion → final answer, each step "
        "supported by a source. answer_source_sample_ids must include sources from at least two different "
        "documents or non-adjacent pages.\n"
        "Return \"reasoning_chain\": the ordered steps, each \"where to look → what is found\", at least two "
        "steps, in Traditional Chinese.\n"
        "Counter-example: \"What is the detection condition of DTC B2562?\" is answered on one page, so it is "
        "Basic, not DRIFT.\n"
    ),
    "global": (
        "Strategy: Global (corpus-wide synthesis). Global map-reduces community reports over the whole graph "
        "and does not cite pages.\n"
        "The sources are the occurrences of one recurring topic across the manuals (see the hint). Conditions "
        "(all required): the question needs all of these sources, not one passage; it asks for an overview, "
        "classification, aggregation, or comparison, such as which manuals state a rule and what the rule has "
        "in common.\n"
        "The reference answer has 3 to 8 bullet points, one sentence each, with only facts from the sources.\n"
        "Not Global: a question that one ready-made list on one page fully answers (Basic).\n"
        "Cite every source the answer uses.\n"
    ),
}


class StrategyQuestionGenerator:
    """Build source bundles per strategy and ask the generation model for one question per bundle."""

    def __init__(
        self,
        projects: ProjectStore,
        connections: ConnectionSettings,
        client: Callable[..., object],
        builder: StrategyBundleBuilder | None = None,
    ) -> None:
        self.projects = projects
        self.connections = connections
        self.client = client
        self.builder = builder or StrategyBundleBuilder(projects)
        self._lock = threading.Lock()

    def generate(
        self,
        project_id: str,
        samples: Sequence[SourceSample],
        total: int,
        mix: Mapping[str, float] | None,
        model: str,
        reasoning_effort: str = "medium",
        parallel: bool = False,
        seed: int = 0,
        progress_callback: Callable[[float, str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> StrategyGenerationResult:
        if total < 1:
            raise ProjectError("生成題數必須是正整數")
        if model not in self.connections.available_chat_models():
            raise ProjectError(f"不支援的題目生成模型：{model}")
        quotas = allocate_quotas(total, mix)
        result = StrategyGenerationResult()
        bundles = {
            strategy: self.builder.build(project_id, strategy, samples, seed) if quotas[strategy] else []
            for strategy in DESIGN_STRATEGIES
        }
        # Strategies without enough related source text hand their questions to Basic.
        for strategy in ("local", "drift", "global"):
            shortfall = quotas[strategy] - len(bundles[strategy])
            if shortfall > 0:
                quotas[strategy] -= shortfall
                quotas["basic"] += shortfall
                result.notes.append(
                    f"{DESIGN_STRATEGY_LABELS[strategy]} 只找到 {len(bundles[strategy])} 組相關原文，"
                    f"{shortfall} 題改出 Basic"
                )
        if quotas["basic"] and not bundles["basic"]:
            bundles["basic"] = self.builder.build(project_id, "basic", samples, seed)
        result.quotas = dict(quotas)
        api_key = self.connections.apply_to_environment(project_id)
        accepted_questions: set[str] = set()
        finished = {strategy: 0 for strategy in DESIGN_STRATEGIES}

        def report() -> None:
            if progress_callback:
                done = sum(finished.values())
                progress_callback(min(0.95, done / total), f"已生成 {done}/{total} 題")

        def run(strategy: str) -> None:
            queue = list(bundles[strategy])
            failures = 0
            while finished[strategy] < quotas[strategy] and queue and failures < 3:
                raise_if_stopped(stop_event)
                needed = quotas[strategy] - finished[strategy]
                batch, queue = queue[:min(BUNDLES_PER_CALL, needed + 1)], queue[min(BUNDLES_PER_CALL, needed + 1):]
                with self._lock:
                    excluded = sorted(accepted_questions)[-30:]
                prompt = self.build_prompt(strategy, batch, excluded)
                try:
                    response = self._call(api_key, model, prompt, reasoning_effort)
                    questions, skipped, errors = self.parse_response(response, strategy, batch)
                except ProjectError as exc:
                    failures += 1
                    with self._lock:
                        result.errors.append(f"{DESIGN_STRATEGY_LABELS[strategy]}：{exc}")
                    continue
                with self._lock:
                    result.skipped.extend(skipped)
                    result.errors.extend(f"{DESIGN_STRATEGY_LABELS[strategy]}：{item}" for item in errors)
                    for question in questions:
                        normalized = normalize_question(question.question)
                        if finished[strategy] >= quotas[strategy] or not normalized or normalized in accepted_questions:
                            continue
                        accepted_questions.add(normalized)
                        result.questions.append(question)
                        finished[strategy] += 1
                    report()

        strategies = [strategy for strategy in DESIGN_STRATEGIES if quotas[strategy]]
        report()
        if parallel and len(strategies) > 1:
            with ThreadPoolExecutor(max_workers=len(strategies)) as pool:
                futures = [pool.submit(run, strategy) for strategy in strategies]
                for future in futures:
                    try:
                        future.result()
                    except JobStopped:
                        continue  # keep what the other strategies made
                    except Exception as exc:
                        result.errors.append(str(exc))
        else:
            for strategy in strategies:
                try:
                    run(strategy)
                except JobStopped:
                    break
                except Exception as exc:
                    result.errors.append(str(exc))
        for strategy in strategies:
            if finished[strategy] < quotas[strategy]:
                result.notes.append(
                    f"{DESIGN_STRATEGY_LABELS[strategy]} 完成 {finished[strategy]}/{quotas[strategy]} 題"
                )
        order = {strategy: index for index, strategy in enumerate(DESIGN_STRATEGIES)}
        result.questions.sort(key=lambda item: order[item.design_strategy])
        return result

    def _call(self, api_key: str, model: str, prompt: str, reasoning_effort: str) -> object:
        args = (self.connections.get_api_base_url(), api_key, model, prompt)
        try:
            if "reasoning_effort" in inspect.signature(self.client).parameters:
                return self.client(*args, reasoning_effort=self.connections.reasoning_effort_for(model, reasoning_effort))
            return self.client(*args)
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"題目生成 API 呼叫失敗：{exc}") from exc

    @staticmethod
    def build_prompt(strategy: str, bundles: Sequence[SourceBundle], excluded: Sequence[str] = ()) -> str:
        payload = [
            {
                "bundle_id": bundle.bundle_id,
                "hint": bundle.hint,
                "sources": [
                    {
                        "sample_id": item.sample_id, "document_id": item.document_id, "page": item.page,
                        "section": item.section_path or item.section_id, "text": item.text[:MAX_SOURCE_CHARACTERS],
                    }
                    for item in bundle.samples
                ],
            }
            for bundle in bundles
        ]
        extra_fields = {
            "local": ',"entities":["..."]',
            "drift": ',"reasoning_chain":["..."]',
        }.get(strategy, "")
        prompt = (
            COMMON_INSTRUCTIONS
            + STRATEGY_INSTRUCTIONS[strategy]
            + "Write exactly one question for each bundle, using only that bundle's sources. If a bundle cannot "
            "support a question that meets these conditions, return {\"bundle_id\":\"...\",\"skip\":\"<reason>\"} "
            "for it instead of forcing one.\n"
            "question_source_sample_ids lists the sources the question is based on; answer_source_sample_ids lists "
            "every source the reference answer uses. Do not put sample ids, page numbers in brackets, or other "
            "citation markers in the question or reference answer.\n"
            "Output only a JSON object in this format: "
            '{"questions":[{"bundle_id":"...","question":"...","reference_answer":"...",'
            '"question_source_sample_ids":["..."],"answer_source_sample_ids":["..."]' + extra_fields + "}]}\n"
        )
        if excluded:
            prompt += (
                "Do not repeat these existing questions or merely reword them: "
                + json.dumps(list(excluded), ensure_ascii=False) + "\n"
            )
        return prompt + f"Bundles: {json.dumps(payload, ensure_ascii=False)}"

    @staticmethod
    def parse_response(
        response: object, strategy: str, bundles: Sequence[SourceBundle],
    ) -> tuple[list[StrategyQuestion], list[dict[str, str]], list[str]]:
        """Valid questions, skipped bundles, and reasons for rejected items; one bad item never drops the others."""
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as exc:
                raise ProjectError("題目生成結果不是有效 JSON") from exc
        if not isinstance(response, dict) or not isinstance(response.get("questions"), list):
            raise ProjectError("題目生成結果缺少 questions 陣列")
        by_id = {bundle.bundle_id: bundle for bundle in bundles}
        questions, skipped, errors = [], [], []
        used: set[str] = set()
        for index, raw in enumerate(response["questions"], start=1):
            label = f"第 {index} 筆"
            if not isinstance(raw, dict) or raw.get("bundle_id") not in by_id:
                errors.append(f"{label}缺少有效的 bundle_id")
                continue
            bundle = by_id[raw["bundle_id"]]
            if bundle.bundle_id in used:
                errors.append(f"{label}重複使用 {bundle.bundle_id}")
                continue
            used.add(bundle.bundle_id)
            if raw.get("skip"):
                skipped.append({"bundle_id": bundle.bundle_id, "strategy": strategy, "reason": str(raw["skip"])})
                continue
            try:
                questions.append(StrategyQuestionGenerator._question(raw, strategy, bundle))
            except ProjectError as exc:
                errors.append(f"{label}（{bundle.bundle_id}）{exc}")
        return questions, skipped, errors

    @staticmethod
    def _question(raw: Mapping[str, Any], strategy: str, bundle: SourceBundle) -> StrategyQuestion:
        question, answer = raw.get("question"), raw.get("reference_answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise ProjectError("缺少問題或參考答案")
        if not QuestionGenerationService._contains_chinese(question) or not QuestionGenerationService._contains_chinese(answer):
            raise ProjectError("未使用繁體中文")
        source_by_id = {item.sample_id: item for item in bundle.samples}

        def sources(key: str) -> list[SourceSample]:
            values = raw.get(key)
            if not isinstance(values, list) or not values or any(value not in source_by_id for value in values):
                raise ProjectError(f"{key} 必須引用該組原文的 sample_id")
            return [source_by_id[value] for value in dict.fromkeys(values)]

        question = _strip_markers(question, source_by_id)
        answer = _strip_markers(answer, source_by_id)
        question_sources = sources("question_source_sample_ids")
        answer_sources = sources("answer_source_sample_ids")
        documents = {item.document_id for item in answer_sources}
        reasoning_chain: tuple[str, ...] = ()
        entities: tuple[str, ...] = ()
        if strategy == "basic":
            pages = sorted({item.page for item in answer_sources})
            if len(documents) > 1 or pages[-1] - pages[0] > 1:
                raise ProjectError("Basic 題的答案必須在同一份文件的同一頁或相鄰兩頁")
        elif strategy == "local":
            entities = tuple(str(item).strip() for item in raw.get("entities") or [] if str(item).strip())
            if len(entities) < 2:
                raise ProjectError("Local 題必須列出至少兩個實體")
        elif strategy == "drift":
            reasoning_chain = tuple(str(item).strip() for item in raw.get("reasoning_chain") or [] if str(item).strip())
            if len(reasoning_chain) < 2:
                raise ProjectError("DRIFT 題的推理鏈至少要兩步")
            if not any(
                StrategyBundleBuilder._far_apart(left, right)
                for position, left in enumerate(answer_sources) for right in answer_sources[position + 1:]
            ):
                raise ProjectError("DRIFT 題的答案來源必須跨文件或包含不相鄰的頁")
        elif strategy == "global":
            if len(answer_sources) < 2:
                raise ProjectError("Global 題的答案必須引用至少兩筆原文")
        answer_evidence = QuestionGenerationService._gold_evidence(answer_sources)
        return StrategyQuestion(
            question=question.strip(),
            reference_answer=answer.strip(),
            design_strategy=strategy,
            answer_scope=infer_answer_scope(answer_evidence),
            question_source_evidence=QuestionGenerationService._gold_evidence(question_sources),
            answer_source_evidence=answer_evidence,
            bundle_id=bundle.bundle_id,
            hint=bundle.hint,
            reasoning_chain=reasoning_chain,
            entities=entities,
        )


def _strip_markers(text: str, sample_ids: Mapping[str, object]) -> str:
    """Remove 【sample-id】／[sample-id] citation markers the model may copy from the source list."""
    if not sample_ids:
        return text
    ids = "|".join(re.escape(item) for item in sorted(sample_ids, key=len, reverse=True))
    cleaned = re.sub(rf"\s*[【\[][^】\]\n]*(?:{ids})[^】\]\n]*[】\]]", "", text)
    return re.sub(r"[ \t]+\n", "\n", cleaned).strip()
