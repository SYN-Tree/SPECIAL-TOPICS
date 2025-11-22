# **SPECIAL-TOPICS**
### **MSCS – 1st Year, 1st Semester Projects**

This repository contains two major simulation projects developed for the **Special Topics in Computer Science** course:

- **Multi-Agent Pac-Men: Negotiation, Learning, and Coordination**
- **AI-Driven Pong Game (Reinforcement-Learning Inspired)**

Both projects explore AI, multi-agent coordination, negotiation, optimization, and real-time simulation techniques using **Python, Pygame, NumPy, and PyTorch**.

---

# 📌 **Project 1 — Multi-Agent Pac-Men**  
### *Negotiation, Learning, and Coordination in Shared Corridors*

---

## **Overview**
This project reimagines Pac-Man as a **multi-agent coordination system**, where multiple autonomous Pac-Men operate in a shared maze competing for space, resources, pellets, and survival.

Agents must negotiate corridor access, avoid ghosts, manage energy, and maintain fairness—modeling real-world multi-robot and distributed AI systems.

---

## **Environment**

### **Maze Grid**
- 2D grid maze with walls, pellets, power pellets, and critical corridors (C)

### **Movement**
- BFS pathfinding  
- Random exploration to avoid deadlocks

### **Ghosts**
- Dynamic roaming hazards with respawn  
- Drain energy and cause delays on contact  

---

## **Agents**

| Agent | Color  | Role |
|-------|---------|------------------------------|
| **A1** | Yellow | Keyboard-controllable leader |
| **A2** | Blue   | Coordination & timing specialist |
| **A3** | Green  | Adaptive learner with probabilistic negotiation |

Each agent maintains:
- Score  
- Energy  
- Wait Counter  
- Negotiation Memory (favor balance + acceptance probability)  
- BFS pathfinding with stochastic adjustments  

### **Objective**
Maximize team score, minimize conflicts, reduce waiting time, and ensure fairness through cooperation.

---

## **Conflict & Negotiation Model**

A conflict occurs when agents attempt to enter the same **Critical Corridor (C)** tile.

### **Resolution Steps**
1. Detect conflict  
2. Execute negotiation protocol  
3. Determine winner who gains access  
4. Update fairness, energy, and logs  

This models real-world **distributed mutual exclusion** and **resource contention**.

---

## **Negotiation Protocols**

### **1. PRIORITY Protocol**
- Deterministic  
- Score-based or predefined hierarchy  
- Fast but may lack fairness  

### **2. ALT-OFFER Protocol**
- Alternating-offer bargaining  
- PROPOSE / ACCEPT / REJECT system  
- Utility-based reasoning (similar to Rubinstein Bargaining)  
- Promotes fairness and cooperation  

**Switch Protocols:**
- **P** → PRIORITY  
- **O** → ALT-OFFER  

---

## **Learning & Adaptation**
Agents adapt through:
- Evolving acceptance probability  
- Favor-based trust metrics  
- Persistent learning saved in **agent_state.json**

---

## **Synchronization & Coordination**
- Deadlock-preventing rerouting  
- Corridor lock/unlock with ownership indicators  
- Intelligent ghost movement  
- Dynamic route adjustments for fairness & efficiency  

---

## **Negotiation Logging**

Exportable structured logs:
- `negotiation_log.csv`  
- `metrics_log.csv`  
- `negotiation_transcripts.json`  

Press **E** to export all logs into **simulation_logs.zip**.

---

## **Simulation Metrics**
- Conflicts detected  
- Successful negotiations  
- Average negotiation rounds  
- Average waiting time  
- Fairness index  
- Utility scores  

---

## **User Interface Features**
- Fullscreen dynamic scaling  
- Animated protocol badge  
- Agent status panel (score, energy, wait, favors)  
- Conflict indicator  
- Scrollable negotiation transcript  
- Corridor ownership glow  
- Smooth UI animations  
- Auto-learning persistence  

---

## **Controls**

| Key | Function |
|------|------------------------------|
| **SPACE** | Play / Pause simulation |
| **R** | Restart simulation |
| **P / O** | Switch negotiation protocol |
| **E** | Export logs |
| **F11** | Toggle fullscreen |
| **PageUp / PageDown** | Scroll negotiation log |
| **ESC** | Exit program |

---

## **Research Applications**
- Multi-robot coordination  
- Autonomous vehicle lane negotiation  
- Distributed AI resource allocation  
- Cooperative bargaining algorithms  
- Multi-agent game theory  

---

## **How to Use**

### **1. Customize UI**
Press **R** to enter UI customization mode.  
Adjust the layout and interface elements as needed.

### **2. Start Simulation**
Press **Space Bar** after customizing the layout.

**Note:**
- Press **R first**, then **Space**.  
- Simulation will not start until **Space** is pressed.  

---

# 📌 **Project 2 — PONG GAME (AI-Based Paddle Simulation)**

---

## **Overview**
A minimalist Pong-like environment where paddles use **PyTorch-powered logic** to simulate learning-based paddle movement.

---

## **Installation**
Ensure Python 3.8+ is installed, then run:

```bash
pip install torch pygame numpy
```
## **How to Run**

1. Open and execute the PongGame Python script.
2. The game window will open automatically.

## **Important Note**

The paddles may be slow on the first launch due to model initialization.
If this happens:

1. Close the window
2. Restart the program
3. Performance will normalize

## **Developer : ASHLEY COMETA**
