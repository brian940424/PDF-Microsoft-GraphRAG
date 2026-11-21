"""Entity types and extraction prompt for technical manuals (F-69).

GraphRAG's default extraction (organization, person, geo, event) turned the SM3E service
manuals into 1,616 entities of which 94% were EVENT names such as ``STEERING SWITCH
REPLACEMENT``; components, codes and materials were rarely entities, and 48% of the text units
linked to no entity at all, so Local search could never reach them. This module points
``extract_graph`` at manual-oriented entity types and an extraction prompt that names parts,
codes and values in the manual's own wording and extracts something from every passage.

The prompt is written next to the settings and referenced by absolute path (GraphRAG falls back
to its default when a relative prompt path does not resolve).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

GRAPH_EXTRACTION_VERSION = "manual-entities-v1"
GRAPH_EXTRACTION_PROMPT_PATH = Path("prompts") / "manual_extraction" / "extract_graph.txt"
ENTITY_TYPES = (
    "COMPONENT", "SYSTEM", "FUNCTION", "PROCEDURE", "DIAGNOSTIC_CODE", "SIGNAL", "MATERIAL", "SPECIFICATION",
)
# One extra pass recovers entities the first pass missed; the prompt also asks for at least one entity per passage.
MAX_GLEANINGS = 1

EXTRACT_GRAPH_PROMPT = """
-Goal-
Given a passage from a technical manual (for example a vehicle service manual or a product user manual) and a list of entity types, identify all entities of those types in the passage and all relationships among the identified entities. The graph is used to find the passage again when someone asks about a part, code, function, procedure or value it describes.

-Entity types-
- COMPONENT: a physical part, unit, module, sensor, switch, actuator, connector, harness, fuse, relay, panel, bracket or tool. Examples: STEERING SWITCH, TCM, ABS ACTUATOR AND ELECTRIC UNIT (CONTROL UNIT), CONNECTOR E24, REAR PANEL ASSEMBLY.
- SYSTEM: a system made of several components. Examples: VDC SYSTEM, INTELLIGENT KEY SYSTEM, REAR VIEW CAMERA SYSTEM, AUDIO SYSTEM.
- FUNCTION: a function, mode, feature or setting of a system. Examples: TCS FUNCTION, FAIL-SAFE MODE, AUTO DOOR LOCK, BATTERY SAVER.
- PROCEDURE: a named procedure or task the manual gives, such as a removal, installation, inspection, adjustment, diagnosis or registration. Name it after its object and action as the manual titles it. Examples: STEERING SWITCH REMOVAL AND INSTALLATION, BRAKE FLUID LEVEL SWITCH COMPONENT INSPECTION.
- DIAGNOSTIC_CODE: a diagnostic trouble code with its display name. Examples: C1154 PNP SWITCH, C1155 BR FLUID LEVEL LOW, U1000 CAN COMM CIRCUIT.
- SIGNAL: a signal, communication line or data item passed between components. Examples: STEERING SWITCH SIGNAL, VEHICLE SPEED SIGNAL, CAN COMMUNICATION.
- MATERIAL: a material, fluid, adhesive, sealant or consumable. Examples: SP130, ULTRA HIGH STRENGTH STEEL, BRAKE FLUID, PC/PMMA.
- SPECIFICATION: a named specification or value item. Examples: STEERING SWITCH BOLT TIGHTENING TORQUE, BRAKE PEDAL HEIGHT.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: the entity's name in capital letters, written as the manual writes it (keep codes, connector numbers, part numbers and abbreviations exactly). Name a component, system, code or material by itself: write STEERING SWITCH, not STEERING SWITCH REPLACEMENT or STEERING SWITCH CHECK. Use the same name every time the same thing appears.
- entity_type: One of the following types: [{entity_types}]
- entity_description: what the passage says about the entity: its role, location, conditions, values, steps or cautions, in the passage's terms
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

Extract every component, code, connector, material and value item the passage names, including those that appear only in tables, terminal lists, cautions and notes. A system description page with no actions still has entities: the components and signals it describes. Every passage that contains manual content has at least one entity. Do not extract page headers, page numbers, figure identifiers or the manufacturer or brand name alone. Do not make separate entities for single terminal numbers, wire colors, switch positions (such as ON or OFF) or GROUND; put them in the description of the component or connector they belong to.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other in the passage, for example: one component sends a signal to another, a component is powered by or grounded through another, a component is part of a system, a code is detected by a component or points at a faulty component, a procedure requires removing another part first or refers to another procedure, a material is used on a component, a value applies to a component.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: the relation as the passage states it, including its direction and conditions
- relationship_strength: a numeric score from 1 to 10 indicating how directly the passage states the relation
 Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

######################
-Examples-
######################
Example 1:
Entity_types: COMPONENT,SYSTEM,SIGNAL
Text:
AV > SYSTEM DESCRIPTION > AUDIO SYSTEM > STEERING SWITCH
The steering switch signal is sent to the combination meter. The combination meter transmits the signal to the AV control unit via CAN communication.
######################
Output:
("entity"<|>STEERING SWITCH<|>COMPONENT<|>Steering switch whose operation signal is sent to the combination meter)
##
("entity"<|>COMBINATION METER<|>COMPONENT<|>Receives the steering switch signal and transmits it to the AV control unit via CAN communication)
##
("entity"<|>AV CONTROL UNIT<|>COMPONENT<|>Receives the steering switch signal from the combination meter via CAN communication)
##
("entity"<|>STEERING SWITCH SIGNAL<|>SIGNAL<|>Signal from the steering switch, passed through the combination meter to the AV control unit)
##
("entity"<|>CAN COMMUNICATION<|>SIGNAL<|>Communication line the combination meter uses to send the steering switch signal to the AV control unit)
##
("entity"<|>AUDIO SYSTEM<|>SYSTEM<|>System whose steering switch input is described in this passage)
##
("relationship"<|>STEERING SWITCH<|>COMBINATION METER<|>The steering switch sends its signal to the combination meter<|>9)
##
("relationship"<|>COMBINATION METER<|>AV CONTROL UNIT<|>The combination meter transmits the steering switch signal to the AV control unit via CAN communication<|>9)
##
("relationship"<|>STEERING SWITCH<|>AUDIO SYSTEM<|>The steering switch is an input of the audio system<|>6)
<|COMPLETE|>

######################
Example 2:
Entity_types: COMPONENT,DIAGNOSTIC_CODE,PROCEDURE
Text:
BRC > C1155 BR FLUID LEVEL LOW > DTC Logic
| C1155 | BR FLUID LEVEL LOW | Brake fluid level is low or communication line between the ABS actuator and electric unit (control unit) and brake fluid level switch is open or shorted. | Harness or connector, Brake fluid level switch |
2. CHECK BRAKE FLUID LEVEL SWITCH
Perform the brake fluid level switch component inspection. Refer to BRC-89, "Component Inspection".
######################
Output:
("entity"<|>C1155 BR FLUID LEVEL LOW<|>DIAGNOSTIC_CODE<|>DTC C1155 is detected when brake fluid level is low or the line between the ABS actuator and electric unit (control unit) and the brake fluid level switch is open or shorted; possible causes are the harness or connector and the brake fluid level switch)
##
("entity"<|>BRAKE FLUID LEVEL SWITCH<|>COMPONENT<|>Switch that reports low brake fluid level; a possible cause of C1155, checked by its component inspection)
##
("entity"<|>ABS ACTUATOR AND ELECTRIC UNIT (CONTROL UNIT)<|>COMPONENT<|>Control unit that detects C1155 over its line to the brake fluid level switch)
##
("entity"<|>BRAKE FLUID LEVEL SWITCH COMPONENT INSPECTION<|>PROCEDURE<|>Inspection of the brake fluid level switch referred to at BRC-89 during C1155 diagnosis)
##
("relationship"<|>C1155 BR FLUID LEVEL LOW<|>BRAKE FLUID LEVEL SWITCH<|>C1155 is caused by low brake fluid level or a fault in the brake fluid level switch or its circuit<|>9)
##
("relationship"<|>ABS ACTUATOR AND ELECTRIC UNIT (CONTROL UNIT)<|>C1155 BR FLUID LEVEL LOW<|>The ABS actuator and electric unit (control unit) detects C1155<|>8)
##
("relationship"<|>C1155 BR FLUID LEVEL LOW<|>BRAKE FLUID LEVEL SWITCH COMPONENT INSPECTION<|>Diagnosis of C1155 step 2 refers to the brake fluid level switch component inspection<|>7)
<|COMPLETE|>

######################
Example 3:
Entity_types: COMPONENT,MATERIAL,PROCEDURE,SPECIFICATION
Text:
BRM > ROOF PANEL > REMOVAL
Before removing the roof panel, remove the headlining. Apply SP130 to the bonding surface. Tightening torque: 8.5 N·m.
######################
Output:
("entity"<|>ROOF PANEL<|>COMPONENT<|>Body panel removed after the headlining; SP130 is applied to its bonding surface)
##
("entity"<|>HEADLINING<|>COMPONENT<|>Interior trim that must be removed before the roof panel)
##
("entity"<|>SP130<|>MATERIAL<|>Material applied to the bonding surface during roof panel work)
##
("entity"<|>ROOF PANEL REMOVAL<|>PROCEDURE<|>Removal of the roof panel, which requires removing the headlining first)
##
("entity"<|>ROOF PANEL TIGHTENING TORQUE<|>SPECIFICATION<|>Tightening torque of 8.5 N·m given in the roof panel procedure)
##
("relationship"<|>ROOF PANEL REMOVAL<|>HEADLINING<|>The headlining must be removed before removing the roof panel<|>9)
##
("relationship"<|>SP130<|>ROOF PANEL<|>SP130 is applied to the roof panel bonding surface<|>8)
##
("relationship"<|>ROOF PANEL TIGHTENING TORQUE<|>ROOF PANEL<|>The 8.5 N·m tightening torque applies to the roof panel<|>7)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""


def configure_graph_extraction(settings: dict[str, Any], graph_root: Path) -> None:
    """Write the extraction prompt under ``graph_root`` and point ``extract_graph`` at it."""
    path = (graph_root / GRAPH_EXTRACTION_PROMPT_PATH).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text(encoding="utf-8") != EXTRACT_GRAPH_PROMPT:
        path.write_text(EXTRACT_GRAPH_PROMPT, encoding="utf-8")
    section = settings.get("extract_graph")
    if not isinstance(section, dict):
        section = settings["extract_graph"] = {}
    section["prompt"] = str(path)
    section["entity_types"] = list(ENTITY_TYPES)
    section["max_gleanings"] = MAX_GLEANINGS
