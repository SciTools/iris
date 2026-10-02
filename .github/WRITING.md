# Writing for a Reader in a Hurry

Clear writing is familiar writing, and familiar writing is not jarring. A
reader who is not stopped by the phrasing reaches the meaning, and a reader
who understands you can decide whether to believe you. Unfamiliar prose fails
before any of that. It gets re-read or skipped, and its content is never
judged.

One thing sits alongside. A reader who cannot tell what you verified from what
you guessed has no way to check you, so they discount all of it rather than
the part they caught.

Most of what follows is prohibitions, because telling a writer to "be clear"
changes nothing. Where a rule has an edge it does not cover, ask what a reader
in a hurry would take away.

Trust comes from being checkable, not from sounding careful. Say what you ran.
Say what you did not. Neither needs more than a sentence.

**Scope.** The sentence rules govern every piece of prose you write:
docstrings, comments, commit messages, pull request bodies, issues, review
comments, specifications and plans. The GitHub rules govern only what is
posted to GitHub.

For **where** prose belongs in library code — comment, docstring, test or
module docstring — see [`lib/iris/AGENTS.md`](../lib/iris/AGENTS.md). This
file covers how the sentence is built, not where it lives.


## Use Words That Are Already In Use

Search for the repository's own term before coining one. A word invented for a
single sentence reads as foreign even when it is accurate, and a reader who
meets three of them stops trusting the register.

- Grep first. If Iris calls it a *cell measure*, do not call it a *per-cell
  weight*.
- Do not introduce a term of art that appears exactly once in the repository.
- Conventional verbs are fine: a loader *expects*, a test *pins*, a function
  *assumes*. Novel metaphors are not. A method is not *half of a class*, and
  reading a property is not a *write-path act*.


## Sentences

- **Put the conclusion in the first clause** of every sentence and every
  paragraph. The reader stops partway through. What survives truncation should
  still be the point.
- **One sentence, one fact.** A qualifier that matters gets its own sentence.
  A reader in a hurry takes the main clause and drops the rest, so anything in
  a subordinate clause is not really there.
- **Split, never compress.** When a sentence is too long, divide it or cut a
  fact. Making it denser is the failure this file exists to prevent.
- **An em dash rescuing a sentence means the sentence is overloaded.** Use a
  full stop.
- **Name what you did instead of hedging.** "This should work" tells the
  reader nothing. "I ran it on both branches" tells them what to believe.
  Write "I did not test this" when you did not.
- **No rhetorical contrast unless the reader would assume the rejected side.**
  "Returns a view rather than a copy" stays, because a reader would assume a
  copy. "Passed through, not reached around" goes, because nobody proposed
  reaching around.
- **Do not answer an objection nobody made.** A sentence that would read
  naturally as a reply in a review thread belongs in the review thread.
- **Write the fact, not your side of the argument about it.** Three tells that
  you have written a reply: a negation whose positive was never stated; a
  "therefore" or "so" whose premise is absent; a term of art used once.


## What Stays True

Prose is not corrected by the thing it describes. A docstring gets fixed by
the next change to the file. A commit message and a pull request body are
never fixed at all.

- **Past tense about a change is durable. Present tense about a state is
  not.** "I changed the guard to a membership test" stays true. "The guard is
  a membership test" is true until someone changes it. They cost the same.
- **Never cite what decays silently**: a line number in another file, a count
  of call sites, a time relative to writing ("currently", "this PR", "the new
  behaviour"). Nothing checks any of them, and your "now" is never the
  reader's.
- **Never cite an artefact of how the work was done**: a review finding, a
  task or plan number, a phase, a checklist item. These resolve to nothing for
  a later reader. Cite an issue, a pull request, a design document, or a
  section of the convention.
- **Do not enumerate test names in a pull request body.** They get renamed and
  deleted, and nothing flags the text that named them. Say the behaviour is
  covered.


## Posting to GitHub

Formatting belongs to the destination. This section applies to pull request
bodies, issues, and comments on either. It applies nowhere else.

- **No hard line wrapping. One paragraph is one line.** GitHub wraps to the
  reader's window. Breaks at 80 columns produce a ragged column that looks
  twice as long as it is, and they follow anyone who quotes you.
- **Write real characters.** `--` renders as two hyphens, not an em dash.
- **Open with the model in parentheses**, as the root
  [`AGENTS.md`](../AGENTS.md) requires.

### Pull request bodies

- **The first three sentences must be enough to decide whether to read on.**
  Say what changed, what breaks if you are wrong, and what you want from the
  reader.
- **Everything above the first collapsed section fits on one screen.**
- **Say where you would look first.** Name the two or three places you are
  least confident about. It aims the reviewer's attention, and it is the
  cheapest thing you can do for their trust.
- **The body describes the branch, not the review.** Edit it as the branch
  changes, and let the threads carry the history.
- **Record what you rejected.** That has no home in the tree, and it is the
  part of a body most worth reading later.

### Collapsed sections

- **Fold evidence. Never fold a claim the reviewer has to weigh.** Test
  output, tracebacks, and enumerations behind a visible headline are evidence.
  A breaking change, a known gap, or something you did not test is a claim. A
  folded admission is a deniable one, and that costs more than verbosity.
- **The summary states the conclusion, so that opening it is optional.** A
  reader skimming every summary line should come away with the argument. "Why
  the write lock moved" fails that. "The write lock is built on first use,
  because Dask's process scheduler rejects it at construction" passes.
- **Leave a blank line after `</summary>`**, or the Markdown inside will not
  render.

### Alerts

- **An alert is for something the reader must act on, or must not miss.**
  Never for emphasis. Their whole value is scarcity.
- **Two per body at most**, both above the first collapsed section. Wanting a
  third means none of them were alerts.
- **`[!IMPORTANT]` for a decision you need. `[!WARNING]` for something that
  will break for someone.** Do not use `[!NOTE]`, `[!TIP]` or `[!CAUTION]` in
  text you write; they decay into decoration. Something merely worth knowing
  is a sentence.

### Permalinks

Press <kbd>y</kbd> on a file view to pin the URL to a commit. A permalink
never rots, which makes it the answer to "never cite a line number".

- **On its own line it renders the code inline.** Use that where the code is
  the point.
- **Inside a sentence it stays a link.** Use that in a bullet list, where a
  preview would wreck the scanning the list exists for.
- **Three or four per body.** Linking every call site turns the page into a
  code dump.


## Everything That Is Not GitHub

Specifications, plans, and anything under `docs/` are read in other renderers,
by people with more patience. Keep them to plain Markdown: no alerts, no
collapsed sections, no `<kbd>`. Footnotes are fine here and nowhere else.

Their length is not a fault. They are read once, in full, before work starts.
Every sentence rule above still applies to every line of them.
