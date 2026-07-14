# Persona manifests

Runtime prompt assembly uses only:

- `personality_core.md` — invariant identity and behavioral boundaries of Liza;
- `prisms/<name>.md` — shared emotional expression layer;
- `synthesis_meta.md` — trust hierarchy and synthesis rules.

The role and direction of thought live in `config/agents/*.yaml`. Creative, pragmatic, effective, critic and synthesis prompts must not redefine Liza as a different person.

The former per-agent copies of every prism were removed. They duplicated the same template dozens of times, drifted independently and made emotion replace identity. A prism now has one canonical text shared by all directions; the agent YAML controls how that direction thinks.

## Authoring rule

A prism may change tone, rhythm and attention, but must not change:

- facts and uncertainty;
- competence and effort;
- respect for the user;
- safety and boundaries;
- the core personality described in `personality_core.md`.

A direction instruction may change the method of analysis, but must not invent another biography, relationship or personality.
