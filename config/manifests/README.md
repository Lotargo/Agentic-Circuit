# Persona manifests

Runtime prompt assembly uses only:

- `personality_core.md` — invariant identity and behavioral boundaries of Liza;
- `prisms/<name>.md` — shared emotional expression layer;
- `synthesis_meta.md` — trust hierarchy and synthesis rules.

The role/direction of thought lives in `config/agents/*.yaml`. Creative, pragmatic, effective, critic and synthesis prompts must not redefine Liza as a different person.

The older per-agent directories (`creative-1/`, `creative-2/`, `pragmatic-*`, `effective-*`) are legacy copies from the first implementation and are **not loaded by runtime**. They should not be edited or treated as active instructions; they are retained temporarily only to make migration history explicit and can be removed after the next clean release.

## Authoring rule

A prism may change tone, rhythm and attention, but must not change:

- facts and uncertainty;
- competence and effort;
- respect for the user;
- safety and boundaries;
- the core personality described in `personality_core.md`.
