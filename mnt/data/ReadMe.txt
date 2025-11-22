===============================================================================
Multi-Agent Pac-Men: Negotiation, Learning, and Coordination in Shared Corridors
===============================================================================

PROJECT OVERVIEW
----------------
This project reimagines the classic Pac-Man game as a multi-agent coordination
and negotiation system, where several autonomous Pac-Men share a single maze,
competing and cooperating under limited spatial and energetic resources.

Each agent seeks to maximize its score by collecting pellets and power pellets
while maintaining energy, avoiding ghosts, and negotiating access to shared
corridors. When multiple agents attempt to move into the same corridor, a
negotiation protocol determines which one proceeds.

The framework models real-world coordination problems faced by multi-robot
systems, autonomous vehicles, and distributed AI agents operating under
resource contention and fairness constraints.

-------------------------------------------------------------------------------
ENVIRONMENT
-------------------------------------------------------------------------------
• Maze Grid
  - Large 2D grid maze with dynamic entities.
  - Walls (#) block movement.
  - Pellets (.) and Power Pellets (o) yield score and restore energy.
  - Critical Corridors (C) represent shared-resource bottlenecks that trigger
    negotiations.

• Movement
  - Grid-based, four-directional (no diagonals).
  - BFS pathfinding with random exploration to avoid deadlocks.

• Ghosts
  - Three mobile hazards patrol dynamically.
  - Cause energy loss and stalls upon contact.
  - Intelligent roaming prevents stasis and ensures continuous threat.
  - Automatic respawn after cooldown.

-------------------------------------------------------------------------------
AGENTS
-------------------------------------------------------------------------------
Three autonomous Pac-Men with distinct behavioral roles:

  A1 (Yellow)  : Keyboard-controllable leader
  A2 (Blue)    : Coordination and timing specialist
  A3 (Green)   : Adaptive learner with probabilistic negotiation

Each agent maintains:
  - Score        : Increases when collecting pellets and power pellets.
  - Energy       : Decreases from conflicts or ghost collisions; regenerates
                   with successful actions.
  - Wait Counter : Increments when blocked or delayed.
  - Negotiation Memory : Stores favor balance and acceptance probability.
  - Pathfinding  : BFS with local obstacle avoidance and stochastic exploration.

Objective:
  Maximize total score and fairness while minimizing conflicts, waiting, and
  energy loss through cooperative negotiation and adaptive decision-making.

-------------------------------------------------------------------------------
CONFLICT & NEGOTIATION MODEL
-------------------------------------------------------------------------------
Conflict Trigger:
  A conflict occurs when two or more agents attempt to enter the same critical
  corridor tile (C).

Resolution Process:
  1. Conflict detected by the shared route lock system.
  2. Agents engage in a negotiation protocol.
  3. One agent gains corridor access; others yield and wait.
  4. Energy, fairness, and wait metrics are updated, and results logged.

This models distributed negotiation and mutual exclusion, a key concept in
robotics, networking, and AI coordination.

-------------------------------------------------------------------------------
NEGOTIATION PROTOCOLS
-------------------------------------------------------------------------------
Two protocols govern how conflicts are resolved:

1. PRIORITY Protocol
   - Deterministic, hierarchy-based.
   - The highest-score or pre-assigned agent wins immediately.
   - Models static authority and fast but potentially unfair arbitration.

2. ALT-OFFER Protocol
   - Alternating-offer negotiation system.
   - Agents exchange PROPOSE, ACCEPT, and REJECT messages with associated
     utilities, probabilities, and favors.
   - Promotes fairness and dynamic cooperation.
   - Encodes utility-based reasoning similar to Rubinstein bargaining.

In-Game Switching:
   P  = PRIORITY Mode
   O  = ALT-OFFER Mode

-------------------------------------------------------------------------------
LEARNING & ADAPTATION
-------------------------------------------------------------------------------
• Each agent maintains an acceptance probability and favor balance that evolve
  through repeated negotiations.

• Learned state persists across runs in "agent_state.json", allowing agents to
  gradually adapt toward cooperative and efficient behavior.

• Favors and acceptance tendencies act as learned trust mechanisms between
  agents.

-------------------------------------------------------------------------------
SYNCHRONIZATION & COORDINATION
-------------------------------------------------------------------------------
• Agents coordinate movement to prevent corridor deadlocks and ghost collisions.
• Shared corridors use lock/unlock mechanisms with visible ownership indicators.
• Ghosts move intelligently and respawn without freezing.
• Agents reroute dynamically to minimize waiting and maintain flow balance.

-------------------------------------------------------------------------------
NEGOTIATION TRANSCRIPTS & LOGGING
-------------------------------------------------------------------------------
All negotiations are recorded in IEEE-ready structured logs for analysis.

  negotiation_log.csv
     - Summary of negotiation outcomes (step, protocol, agents, winner, etc.)

  metrics_log.csv
     - Step-by-step simulation metrics (conflicts, fairness, wait times, etc.)

  negotiation_transcripts.json
     - Full transcript of each negotiation, including:
       • Protocol, step, and corridor coordinates
       • Involved agents
       • Round-by-round utility and probability annotations
       • Fairness delta and energy snapshots

Example JSON entry:
{
  "conflict_id": 12,
  "step": 180,
  "protocol": "ALT-OFFER",
  "corridor": [10, 6],
  "agents": [1, 2],
  "winner": 1,
  "success": true,
  "rounds": 2,
  "transcript": [
    "R1:t=0.03 | A1→A2: PROPOSE (offer=1, util=0.81, favors=2, energy=94)",
    "R1:t=0.05 | A2→A1: REJECT  (prob=0.62, util=0.41, energy=88)",
    "R2:t=0.04 | A2→A1: PROPOSE (offer=0, util=0.74, favors=1, energy=87)",
    "R2:t=0.06 | A1→A2: ACCEPT  (prob=0.63, util=0.79, energy=92)"
  ]
}

Press E to export all logs (CSV + JSON) into "simulation_logs.zip".

-------------------------------------------------------------------------------
SIMULATION METRICS
-------------------------------------------------------------------------------
Tracked and updated live in-game:

  - Conflicts Detected     : Number of simultaneous access attempts
  - Successful Negotiations : Resolved without fallback
  - Negotiation Rounds     : Average rounds per conflict (from transcripts)
  - Average Waiting Time   : Mean agent delay
  - Fairness Index         : Score balance among agents
  - Utility Scores         : Logged from each negotiation exchange

-------------------------------------------------------------------------------
USER INTERFACE & VISUALIZATION
-------------------------------------------------------------------------------
  ✓ Dynamic fullscreen scaling and layout adaptation
  ✓ Animated protocol badge (gear = PRIORITY, bubble = ALT-OFFER)
  ✓ Agents Panel with live score, energy, wait, and favors
  ✓ Ghosts Status with color-coded alive/dead timers
  ✓ Conflict Status with heartbeat pulse indicator
  ✓ Live Negotiation Log with scrollable transcript view
  ✓ Color-coded corridor ownership glow
  ✓ Smooth animations and energy bars
  ✓ Auto-learning persistence across runs

-------------------------------------------------------------------------------
CONTROLS
-------------------------------------------------------------------------------
  SPACE      : Play / Pause simulation
  R          : Restart simulation
  P / O      : Switch negotiation protocol
  E          : Export logs (CSV + JSON + ZIP)
  F11        : Toggle fullscreen
  PageUp     : Scroll negotiation log (older)
  PageDown   : Scroll negotiation log (newer)
  ESC        : Exit simulation

-------------------------------------------------------------------------------
RESEARCH CONTEXT
-------------------------------------------------------------------------------
This framework demonstrates adaptive negotiation and cooperation in uncertain
multi-agent systems, relevant to:

  - Multi-robot coordination under limited resources
  - Autonomous vehicle lane and corridor allocation
  - Distributed decision-making and fairness
  - Reinforcement learning for cooperative negotiation
  - Spatial conflict resolution and multi-agent game theory

-------------------------------------------------------------------------------
FUTURE ENHANCEMENTS
-------------------------------------------------------------------------------
  - Reinforcement Learning-based negotiation agents
  - Multi-issue bargaining (space, time, and favors)
  - Negotiation replay and visualization dashboards
  - Advanced ghost AI with adaptive pursuit logic
  - Cooperative-competitive hybrid mode (alliances and betrayals)

-------------------------------------------------------------------------------
HOW TO USE
-------------------------------------------------------------------------------

1. Customize the UI
    - Before starting the simulation, press: ** R ** – Opens UI customization mode
    - Use this mode to adjust the layout, elements, or appearance according to your preference.

2. Start the Game
    - After customizing, press: ** Space Bar ** – Begins the game simulation

Notes:
1. You must press R first to customize the UI before starting.
2. The game will not begin until Space Bar is pressed.
3. Make sure you complete your UI adjustments before starting the simulation.

-------------------------------------------------------------------------------
DEVELOPER:  ASHLEY COMETA
-------------------------------------------------------------------------------
===============================================================================
