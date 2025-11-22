SPECIAL-TOPICS
MSCS – 1st Year, 1st Semester Projects

This repository contains two major simulation projects developed for the Special Topics in Computer Science course:

Multi-Agent Pac-Men: Negotiation, Learning, and Coordination

AI-Driven Pong Game (Reinforcement-Learning Inspired)

Both projects explore AI, multi-agent coordination, negotiation, optimization, and real-time simulation techniques using Python, Pygame, NumPy, and PyTorch.

📌 Project 1 — Multi-Agent Pac-Men
Negotiation, Learning, and Coordination in Shared Corridors
Overview

This project reimagines Pac-Man as a multi-agent coordination system, where multiple autonomous Pac-Men operate in a shared maze competing for space, resources, pellets, and survival.

Agents must negotiate corridor access, avoid ghosts, manage energy, and maintain fairness—modeling real-world multi-robot and distributed AI systems.

Environment

Maze Grid

2D grid maze with walls, pellets, power pellets, and critical corridors (C)

Movement

BFS pathfinding + random exploration to avoid deadlocks

Ghosts

Dynamic roaming hazards with respawn mechanics

Drain energy and cause delays on contact

Agents

Three autonomous agents with distinct roles:

Agent	Color	Role
A1	Yellow	Keyboard-controllable leader
A2	Blue	Coordination & timing specialist
A3	Green	Adaptive learner with probabilistic negotiation

Each agent maintains:

Score

Energy

Wait Counter

Negotiation Memory (favor balance + acceptance probability)

BFS-based pathfinding with stochastic adjustments

Objective:
Maximize team score, minimize conflicts and waiting time, maintain fairness through cooperation.

Conflict & Negotiation Model

A conflict occurs when agents attempt to enter the same Critical Corridor (C) tile.

Resolution Steps

Detect conflict

Execute negotiation protocol

Determine winner who gets access

Update fairness, energy, and logs

This models real-world distributed mutual exclusion and resource contention.

Negotiation Protocols

Two selectable systems:

1. PRIORITY Protocol

Deterministic

Score-based or predefined hierarchy

Fast but less fair

2. ALT-OFFER Protocol

Alternating-offer bargaining

PROPOSE / ACCEPT / REJECT message exchange

Utility-based reasoning (similar to Rubinstein Bargaining)

Promotes fairness

Switching Protocols:

P → PRIORITY

O → ALT-OFFER

Learning & Adaptation

Agents adapt through:

Evolving acceptance probability

Favor-based trust metrics

Persistent learning stored in agent_state.json

Synchronization & Coordination

Deadlock-preventing path rerouting

Corridor lock/unlock with ownership indicators

Intelligent ghost movement and respawn

Dynamic rerouting and fairness balancing

Negotiation Logging

Exportable structured logs:

negotiation_log.csv
metrics_log.csv
negotiation_transcripts.json

Press E to export all logs into simulation_logs.zip.

Simulation Metrics

Conflicts detected

Successful negotiations

Average negotiation rounds

Average waiting time

Fairness index

Utility scores

User Interface Features

Fullscreen dynamic scaling

Animated negotiation protocol badge

Agent status panel (score, energy, wait, favors)

Live conflict indicator

Scrollable negotiation transcript

Color-coded corridor ownership glow

Smooth UI animations

Auto-learning persistence

Controls
Key	Function
SPACE	Play / Pause simulation
R	Restart simulation
P / O	Switch negotiation protocol
E	Export logs
F11	Toggle fullscreen
PageUp/Down	Scroll negotiation log
ESC	Exit
Research Applications

Multi-robot coordination

Autonomous vehicle lane negotiation

Distributed AI resource allocation

Cooperative bargaining algorithms

Multi-agent game theory

How to Use
1. Customize UI

Press R to enter UI customization mode
Adjust layout and elements as needed.

2. Start Simulation

Press Space Bar after customizing the interface.

Note:

Press R first, then Space.

Simulation will not start until Space is pressed.

📌 Project 2 — PONG GAME (AI-Based Paddle Simulation)
Overview

A simple Pong-like environment where paddles use PyTorch-powered logic for movement prediction and learning-inspired behavior.

Installation

Ensure Python 3.8+ is installed, then run:

pip install torch pygame numpy

How to Run

Open the PongGame script

Run the program

The game window will automatically launch

Important Note

The paddles may be slow on the first launch due to model loading.

If this occurs:

Close the window

Restart the program

Performance will normalize

Developer : ASHLEY COMETA
