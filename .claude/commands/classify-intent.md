Read the file `intents.yaml` in the current working directory.

Using the intent taxonomy from that file, classify the following user utterance:

"$ARGUMENTS"

Return your answer as:
- **Domain**: the top-level domain key (e.g. `shopping_assistant`)
- **Intent**: the action label within that domain (e.g. `add_item`)
- **Confidence**: high / medium / low
- **Reasoning**: one sentence explaining why

If the utterance matches no known intent, return `domain: unknown, intent: unknown` and suggest where in the taxonomy it might belong or whether a new intent is needed.
