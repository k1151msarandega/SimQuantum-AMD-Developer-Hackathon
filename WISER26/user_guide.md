## User guide

### Before you start
Run this from [`notebooks/00_launch_app.ipynb`](notebooks/00_launch_app.ipynb) -- see "Install & run" above. That notebook installs everything, smoke-tests the pipeline with no UI, then launches the app through a tunnel and prints a URL to click.

### Live Console tab
This is the actual deliverable: a continuously-running instrument session, not a "pick settings, click Run, wait" flow.

1. **Pick a config** in the sidebar (Quick/CPU teaching demo/Full) and click **Start**. A background session begins immediately -- frames start streaming from QArray, and the device feed and potential well begin updating live.
2. **Watch it run on autopilot** first. The frame counter, Vx/Vy readout, and both panels update continuously without you touching anything -- this is the scripted trajectory driving the device.
3. **Take manual control**: check "Manual Vx" and/or "Manual Vy" in the sidebar, type a voltage, click **Apply**. The device immediately starts responding to your value instead of the script -- you'll see the device feed and potential well shift accordingly within a second or two. Click "Release both to autopilot" to hand control back to the script.
4. **Pause/Resume**: Pause freezes frame consumption without losing any state (staleness log, tier tally, thresholds all stay exactly where they were) -- useful for stopping to look at a specific moment. Resume picks up exactly where it left off.
5. **Read the diagnostics** under "Twin health / diagnostics": current tier (FULL/CHEAP/SKIP -- see Learning objective #5), a drift flag, the running tier tally, and a staleness-over-time chart. If the LLM supervisor is on, its most recent reasoning and a collapsible log appear here too.
6. **Stop** ends the session for real -- a subsequent Start begins a fresh trajectory from frame 0, not a resume.

### Mode comparison tab
The original, simpler flow: pick a pipeline mode (serial/batched/batched_triage/batched_triage_llm) and a config, click **Run**, and watch it run to completion. This is the more direct way to see the serial-vs-batched-vs-triage story side by side (see Learning objectives #3-4) without the Live Console's extra controls in the way.

### What to expect (read this before assuming something's broken)
- **Tiers may stay at FULL the whole run.** This is a known, documented tuning gap (see [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md)'s "Open items") -- the triage thresholds were tuned against the original GPU-batched timing, not this CPU port's, and haven't been re-measured yet. It is not a bug in the triage logic itself.
- **The potential well can look nearly flat** at some voltages and clearly curved (a real saddle shape) at others -- this is real physics (see `viz/potential_well.py`'s module docstring for the numeric reasoning), not inconsistent rendering.
- **Occasional `Failed to fetch dynamically imported module` errors** are a known `localtunnel`-through-Colab fragility, not an app bug -- a hard refresh usually clears it. See "Future improvements" below for the plan to remove this dependency entirely.
