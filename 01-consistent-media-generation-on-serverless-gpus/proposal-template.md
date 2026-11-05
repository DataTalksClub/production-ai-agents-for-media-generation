# Capstone proposal: one page

Use the [project requirements](../projects/README.md) as the assessment source of truth.

- **Problem and user:** Who needs these assets, and what task becomes easier?
- **Records:** What fields does each input contain? Where will the data come from?
- **Two GPU media models:** Name exact candidates, modalities, licences and why each is needed.
- **Workflow:** Brief → agent decision → GPU tools → checks → saved artifacts. Mark the asynchronous step.
- **Quality:** Two observable acceptance criteria and an initial evaluation set.
- **Budget:** Planned records and media counts, rate sources, measured vs estimated times, startup/retry/idle assumptions and maximum spend.
- **Delivery:** Demo, code, evidence, cleanup and remaining risks.

## Example direction: astronomy story cards

An astronomy club organizer needs an illustration and a spoken invitation for each session. Each record contains a fictional mascot, activity, audience age range and invitation text. Sana produces the illustration; Kokoro is a candidate for narration after checking its model and voice terms. The agent drafts the prompt and narration, calls both tools, and records assets in Object Storage. A finite Job handles a small prepared set; the reviewer checks the required object is visible and the activity name is pronounced correctly.

The budget is not yet established. The first experiment measures one image and one narration, then estimates the planned batch plus startup and retry allowance. Model 2 latency, voice choice and live budget remain unverified. This is an example proposal direction, not a completed capstone.
