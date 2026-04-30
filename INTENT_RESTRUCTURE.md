# Intent Taxonomy Restructure — Rationale

## Overview

The original intent specification (`intents.txt`) was reorganized into a structured YAML taxonomy (`intents.yaml`). This document explains what changed, why it was necessary, and how the new format benefits the system going forward.

---

## What the Original File Contained

The original file organized user phrases under package names (e.g. `BENEVOLENCE`, `MONEYSHARE`) using full natural language sentences as the primary identifier, with an optional `Alias` column for alternate phrasing.

**Example from the original:**

| Package    | Intent                          | Alias                        |
|------------|---------------------------------|------------------------------|
| MONEYSHARE | I want to avoid a bank fee      |                              |
| MONEYSHARE | I need a small loan             |                              |
| MATH       | compute xplusonesquared         | Compute xsquared plus one    |

---

## The Problem

### 1. No Actual Intent Labels

In NLU (Natural Language Understanding) systems, an **intent** is a short, machine-readable label that represents *what the user wants to do* — for example `avoid_fee` or `request_loan`. The original file uses full sentences where the label should be, which means:

- There is no stable, referenceable name for any given intent
- Two sentences that mean the same thing appear as two different intents
- Code, APIs, and routing logic have no clean string to match against

### 2. Conflation of Labels and Training Data

The sentences in the `Intent` column serve two completely different purposes in the original file:

- As an **intent identifier** (what the intent *is called*)
- As a **training utterance** (an example of what a user might say)

These are distinct concepts. A sentence like `"I am hungry"` is a training example — it teaches the system what the `request_food` intent sounds like. It should never be the name of the intent itself. Conflating the two makes the taxonomy brittle: renaming a phrase breaks the intent's identity.

### 3. Duplicate Meaning Without Grouping

`"I'm hungry"` and `"I am hungry"` appear as separate rows under `FOODSHARE`, when they are clearly two utterances for the same intent. The original format has no mechanism to express that these map to a single action. This grows into a maintenance problem as more phrasings are added over time.

### 4. No Structure for Aliasing

The `Alias` column is inconsistently populated and only appears for a few entries. In practice it was doing the job that a proper `utterances` list should do — providing alternate phrasings for the same intent.

---

## What We Changed

The new `intents.yaml` introduces a three-level hierarchy:

```
domain → intent_label → utterances
```

- **Domain**: the package or feature area (e.g. `moneyshare`, `shopping_assistant`)
- **Intent label**: a concise, snake_case action name (e.g. `avoid_fee`, `add_item`)
- **Utterances**: a list of example phrases a user might say to trigger that intent

**Equivalent example in the new format:**

```yaml
moneyshare:
  avoid_fee:
    utterances:
      - "I want to avoid a bank fee"
  request_loan:
    utterances:
      - "I need a small loan"
```

---

## Why This Matters for an LLM Integration

When an LLM is used to classify user input against a set of intents, it needs:

1. **A stable label to return** — the model's output needs to be a value your code can act on. `avoid_fee` is actionable; `"I want to avoid a bank fee"` is not.
2. **Examples to reason from** — utterances serve as few-shot context that guides the model toward the correct classification.
3. **A single source of truth** — keeping all phrasings for one intent together under one label means updating coverage requires editing one place, not hunting through a flat list.

The restructured file feeds directly into the `/classify` endpoint, which builds a system prompt from the taxonomy at runtime. Adding a new utterance to `intents.yaml` immediately improves classification accuracy with no code changes.

---

## Summary of Benefits

| Concern                  | Original Format         | New Format                        |
|--------------------------|-------------------------|-----------------------------------|
| Machine-readable labels  | None                    | Snake_case intent labels          |
| Grouping of synonyms     | Separate rows           | Unified utterances list           |
| Extensibility            | Append rows to flat CSV | Add utterances under existing key |
| LLM/API integration      | Not directly usable     | Loads directly into system prompt |
| Aliasing                 | Partial, inconsistent   | First-class `aliases` field       |
| Readability              | Moderate                | Explicit hierarchy                |

---

## Recommendation

The taxonomy is currently sparse in a few areas worth addressing before production:

- `math.compute_expression` — only covers two very specific phrasings; common variants like "calculate", "what is", or "evaluate" are missing
- `flow_planner.create_flow_plan` — the single utterance references "tomorrow" specifically, which will miss any other time reference
- `remembot` — has no coverage for listing or deleting memories, only storing and recalling

These can be expanded directly in `intents.yaml` without any engineering changes.
