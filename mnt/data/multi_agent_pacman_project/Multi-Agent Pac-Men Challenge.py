import os, json, csv, zipfile, time, random, math, pygame, itertools
from collections import namedtuple, defaultdict, deque

pygame.init()
try:
    pygame.mixer.init()
except Exception:
    pass

# --- Paths & constants ---
BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE, "assets")
SPRITES = os.path.join(ASSETS, "sprites")
GHOSTS = os.path.join(ASSETS, "ghosts")

NEG_CSV = os.path.join(BASE, "negotiation_log.csv")
NEG_JSON = os.path.join(BASE,"negotiation_transcripts.json")
MET_CSV = os.path.join(BASE, "metrics_log.csv")
STATE_JSON = os.path.join(BASE, "agent_state.json")
ZIP_OUT = os.path.join(BASE, "simulation_logs.zip")

Pos = namedtuple("Pos", ["r", "c"])
BLACK = (8, 8, 10)
WHITE = (230, 230, 235)
NEON = (0, 200, 255)
ORANGE = (255, 160, 40)
GREEN = (0, 220, 160)
RED = (255, 80, 80)

FPS = 22
WINDOWED_SIZE = (1200, 800)
fullscreen = False

# Global game_over flag to stop any further energy changes after finish
GAME_OVER = False

# --- Maze (large example) ---
MAZE = [
"############################################",
"#.............#........C.........#.........#",
"#.#####.#####.#.#####.###.#####.#.#####.###.",
"#.#...#.....#.#.....#...#.....#.#.....#...#.",
"#.#.#.#####.#.###.#.#.#.###.#.#.###.#.#.#.#.",
"#...#...C...#...#.#.#.#...#.#.#...#.#.#.#.#.",
"#####.#####.###.#.#.#####.#.#.###.#.#.#.#.#.",
"#.....#...#.....#.#.......#.#.#...#.#...#...#",
"#.#####.#.#######.#########.#.#.###.#######.#",
"#.#.....#.......#.....C.....#.#...#...C.....#",
"#.#.###########.###########.#.###.###.#####.#",
"#.#...........#.....#.....#.#...#.....#.....#",
"#.###########.#####.#.###.#.###.#######.#####",
"#.....C.......#...#...#...#...#.......#.....#",
"#####.#####.###.#.#####.#####.#####.###.#####",
"#...#.#...#.....#.......#...#...#...#...#...#",
"#.#.#.#.#.###############.#.#.#.#.###.#.###.#",
"#.#.#...#.....#...........#.#.#.#.....#.#...#",
"#.#.#########.#.#############.#.#######.#.###",
"#.#...........#.....C.........#.....C...#...#",
"#.#############################.###########.#",
"#...........................................",
"#.#####.#####.#####.#####.#####.#####.#####.#",
"#.#...#.....#.....#.....#.....#.....#.....#.#",
"#.#.#.#####.#.###.#.###.#####.###.#####.###.#",
"#.#.#.......#.#...#.#.........#.#.....#.....#",
"#.#.#########.#.###.#.#########.#.###.#####.#",
"#.#...........#...#.#.....C.....#.#...#.....#",
"#.###########.###.#.#############.#.###.###.#",
"#.............#...#...............#.....o...#",
"############################################",
]
# Normalize width
max_width = max(len(r) for r in MAZE)
MAZE = [r.ljust(max_width, "#") for r in MAZE]
ROWS, COLS = len(MAZE), len(MAZE[0])

# --- Helpers: load images & layout ---
def load_img_safe(path, size=None, fallback=(255,215,0)):
    try:
        im = pygame.image.load(path).convert_alpha()
        if size:
            im = pygame.transform.smoothscale(im, size)
        return im
    except Exception:
        surf = pygame.Surface(size or (48,48), pygame.SRCALPHA)
        w,h = surf.get_size()
        pygame.draw.rect(surf, (25,25,60),(0,0,w,h), border_radius=6)
        pygame.draw.circle(surf, fallback, (w//2,h//2), min(w,h)//3)
        return surf

def compute_layout(screen, panel_w=320, bottom_margin=120):
    w,h = screen.get_size()
    margin = 28
    max_tile_w = max(4, (w - panel_w - margin) // COLS)
    max_tile_h = max(4, (h - bottom_margin) // ROWS)
    tile = max(4, min(max_tile_w, max_tile_h))
    maze_w = COLS * tile
    ox = max(12, (w - panel_w - maze_w) // 2)
    oy = max(8, (h - ROWS * tile) // 2 - 6)
    return tile, ox, oy

def in_bounds(p): return 0 <= p.r < ROWS and 0 <= p.c < COLS
def walkable(p,g): return in_bounds(p) and g[p.r][p.c] != '#'
def neighbors(p): return [Pos(p.r-1,p.c), Pos(p.r+1,p.c), Pos(p.r,p.c-1), Pos(p.r,p.c+1)]

# from collections import deque
def bfs(start, targets, g):
    q = deque([start]); came = {start: None}
    while q:
        cur = q.popleft()
        if cur in targets:
            path=[]
            while cur != start:
                path.append(cur)
                cur = came[cur]
            path.reverse()
            return path
        for nb in neighbors(cur):
            if nb not in came and walkable(nb,g):
                came[nb]=cur; q.append(nb)
    return []

def fairness(scores):
    if not scores: return 1.0
    m = sum(scores)/len(scores)
    if m == 0: return 1.0
    v = sum((s-m)**2 for s in scores)/len(scores)
    return max(0.0, 1.0 - v/(m*m + 1e-9))

# --- Classes ---
class CorridorManager:
    def __init__(self,g):
        self.corrs = {Pos(r,c) for r in range(ROWS) for c in range(COLS) if g[r][c]=='C'}
        self.owner = {p: None for p in self.corrs}
    def token(self,p,aid):
        if p not in self.owner: return True
        o = self.owner[p]; return o is None or o == aid
    def grant(self,p,aid):
        if p in self.owner: self.owner[p]=aid
    def release(self,agents):
        for p,o in list(self.owner.items()):
            if o and not any(a.pos==p for a in agents):
                self.owner[p] = None

class Agent:
    def __init__(self, aid, pos, img, keyboard=False, memory=None):
        self.aid = aid
        self.pos = pos
        self.img = img
        self.keyboard = keyboard
        self.next = None
        self.score = 0
        self.energy = 100.0
        self.wait = 0
        self.stalled = False
        prev = {}
        if memory and str(aid) in memory:
            prev = memory[str(aid)]
        self.accept_prob = prev.get("accept_prob", 0.5)
        self.favors = prev.get("favors", 0)

    def decide(self, pellets, g, ghost_positions, others=None):
        if self.stalled:
            return self.pos
        if self.keyboard and self.next:
            dr, dc = self.next
            cand = Pos(self.pos.r + dr, self.pos.c + dc)
            if walkable(cand, g) and not self._adjacent_to_ghost(cand, ghost_positions):
                return cand
        if not pellets:
            return self.pos
        path = bfs(self.pos, pellets, g)
        if not path:
            return self.pos
        if hasattr(self, "last_pos") and self.last_pos == path[0]:
            nbs = [p for p in neighbors(self.pos) if walkable(p,g)]
            if nbs:
                self.last_pos=self.pos; return random.choice(nbs)
        if random.random()<0.15:
            explore=[p for p in neighbors(self.pos) if walkable(p,g)]
            if explore:
                self.last_pos=self.pos; return random.choice(explore)
        if others:
            close=[a for a in others if a is not self and abs(a.pos.r-self.pos.r)+abs(a.pos.c-self.pos.c)<3]
            if close and random.random()<0.2:
                away=[p for p in neighbors(self.pos) if walkable(p,g)
                      and all(abs(p.r-o.pos.r)+abs(p.c-o.pos.c)>1 for o in close)]
                if away:
                    self.last_pos=self.pos; return random.choice(away)
        for step in path[:4]:
            if not self._adjacent_to_ghost(step, ghost_positions):
                self.last_pos=self.pos; return step
        self.last_pos=self.pos
        return path[0]

    def collect(self, pellets, g):
        if self.pos in pellets:
            ch = g[self.pos.r][self.pos.c]
            self.score += 5 if ch == 'o' else 1
            global GAME_OVER
            if not GAME_OVER: self.energy = min(100.0, self.energy + 3.0)
            pellets.discard(self.pos)
            g[self.pos.r][self.pos.c] = ' '

    def penalize(self, amount=6):
        global GAME_OVER
        if GAME_OVER or self.stalled: return
        self.energy = max(0, self.energy - amount)
        if self.energy <= 0:
            self.energy = 0
            self.stalled = True

    def _adjacent_to_ghost(self,p,ghost_positions):
        return any(abs(gp.r-p.r)+abs(gp.c-p.c)<=1 for gp in ghost_positions)

    def draw(self, surf, ox, oy, tile):
        # --- Draw faint tombstone if dead (not just stalled) ---
        if self.energy <= 0:
            # Fade-out tombstone over time
            if not hasattr(self, "death_fade"):
                self.death_fade = 255  # start full bright
            else:
                self.death_fade = max(60, self.death_fade - 1)  # slowly dim to 60

            cx = ox + self.pos.c * tile + tile // 2
            cy = oy + self.pos.r * tile + tile // 2
            radius = max(6, tile // 3)

            fade_color = (self.death_fade, self.death_fade, self.death_fade)
            border_color = tuple(max(20, int(c * 0.5)) for c in fade_color)

            pygame.draw.circle(surf, fade_color, (cx, cy), radius)
            pygame.draw.circle(surf, border_color, (cx, cy), radius, 2)
            return  # stop normal sprite rendering

        # --- Normal Pac-Man drawing ---
        surf.blit(self.img, (ox + self.pos.c * tile, oy + self.pos.r * tile))
        bar_w, bar_h = tile, max(4, tile // 8)
        bx, by = ox + self.pos.c * tile, oy + self.pos.r * tile + tile + 4
        pygame.draw.rect(surf, (30, 30, 40), (bx, by, bar_w, bar_h))
        fill = int((self.energy / 100.0) * bar_w)
        col = (0, 220, 120) if self.energy > 70 else (230, 200, 30) if self.energy > 30 else (230, 80, 80)
        if fill > 0:
            pygame.draw.rect(surf, col, (bx, by, fill, bar_h))

    def predict_next_move(self, pellets, g, others):
        """Predict next 2 moves based on BFS path; used for pre-conflict detection."""
        if self.stalled:
            return []
        path = bfs(self.pos, pellets, g)
        return path[:2] if path else []

class Environment:
    def __init__(self, grid):
        self.grid = grid
        self.shared_routes = {Pos(r, c): "UNLOCKED"
                              for r in range(ROWS) for c in range(COLS)
                              if grid[r][c] == "C"}

    def lock(self, pos, agent_id):
        """Agent attempts to acquire lock on a shared route."""
        state = self.shared_routes.get(pos, "UNLOCKED")
        if state == "UNLOCKED":
            self.shared_routes[pos] = f"LOCKED_BY_{agent_id}"
            return True
        else:
            return False  # triggers conflict event

    def unlock(self, pos, agent_id):
        """Releases lock only if owned by the same agent."""
        state = self.shared_routes.get(pos)
        if state == f"LOCKED_BY_{agent_id}":
            self.shared_routes[pos] = "UNLOCKED"

    def trigger_conflict(self, agent_a, agent_b, pos):
        """Registers and defers control to negotiation manager."""
        print(f"[Conflict Trigger] Agents A{agent_a.aid} & A{agent_b.aid} on {pos}")
        return {"conflict_pos": pos, "agents": [agent_a, agent_b]}

class Ghost:
    def __init__(self, pos, img):
        self.pos = pos
        self.img = img
        self.alive = True
        self.respawn_timer = 0
        self.spawn_point = pos
        self.idle_steps = 0  # prevent total freeze
        self.last_dir = None

    def update(self, pac_positions, g):
        """Smart movement: chase if alive agents exist, else roam."""
        if not self.alive:
            self.respawn_timer -= 1
            if self.respawn_timer <= 0:
                # find a valid respawn location near spawn
                for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0)]:
                    p = Pos(self.spawn_point.r + dr, self.spawn_point.c + dc)
                    if walkable(p, g):
                        self.pos = p
                        break
                self.alive = True
                self.idle_steps = 0
            return

        # --- Alive ghost logic ---
        # If there are no alive agents, roam randomly to prevent stalling
        valid_targets = [p for p in pac_positions if p is not None]
        best, bestlen = None, 1e9

        if valid_targets:
            for p in valid_targets:
                path = bfs(self.pos, {p}, g)
                if path and len(path) < bestlen:
                    bestlen, best = len(path), path[0]
        else:
            best = None

        # If no path found, roam randomly (keeps ghost moving)
        if best is None:
            nbs = [nb for nb in neighbors(self.pos) if walkable(nb, g)]
            if nbs:
                # try not to reverse direction constantly
                if self.last_dir:
                    opp = Pos(self.pos.r - self.last_dir[0], self.pos.c - self.last_dir[1])
                    nbs = [n for n in nbs if n != opp]
                nxt = random.choice(nbs)
                self.last_dir = (nxt.r - self.pos.r, nxt.c - self.pos.c)
                self.pos = nxt
                self.idle_steps += 1
            else:
                self.idle_steps += 1
        else:
            self.pos = best
            if self.last_dir:
                self.last_dir = (best.r - self.pos.r, best.c - self.pos.c)
            self.idle_steps = 0

        # --- Safety check ---
        # If ghost gets stuck too long, teleport near spawn to prevent lock
        if self.idle_steps > 40:
            for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                p = Pos(self.spawn_point.r + dr, self.spawn_point.c + dc)
                if walkable(p, g):
                    self.pos = p
                    self.idle_steps = 0
                    break

    def kill(self):
        """Ghost temporarily dies, then respawns after timer."""
        self.alive = False
        self.respawn_timer = 150  # ~6 seconds

    def draw(self, surf, ox, oy, tile):
        """Draw ghost; gray if dead."""
        x, y = ox + self.pos.c * tile, oy + self.pos.r * tile
        ghost_img = self.img.copy()
        if not self.alive:
            ghost_img.fill((100, 100, 100, 150), special_flags=pygame.BLEND_RGBA_MULT)
        surf.blit(ghost_img, (x, y))


# --- Negotiation + learning ---
def negotiate_group(group, pos, protocol, pellets, g):
    """
    Handles negotiation among multiple agents (>2) or mixed scenarios.
    Now returns: (winner, success, detail, transcript)
    With full IEEE-grade utility and probability annotations.
    """
    transcript = []
    alive = [a for a in group if not a.stalled]

    # --- Single-agent auto pass ---
    if len(alive) <= 1:
        sole = alive[0] if alive else group[0]
        transcript.append(f"A{sole.aid} AUTO (alone/alive)")
        return sole, True, "AUTO", transcript

    # --- Priority Protocol ---
    if protocol == "PRIORITY":
        winner = max(group, key=lambda a: a.score)
        loser_ids = [a.aid for a in group if a is not winner]
        transcript.append(
            f"PRIORITY: A{winner.aid} wins (score={winner.score}) | losers={loser_ids}"
        )
        transcript.append(
            f"UTILITY SNAPSHOT: "
            + ", ".join([f"A{a.aid}(energy={int(a.energy)}, favors={a.favors})" for a in group])
        )
        return winner, True, "PRIORITY", transcript

    # --- Alternating Offers (multi-agent) ---
    order = group[:]
    random.shuffle(order)
    rounds, MAX_ROUNDS = 0, 4
    success = False
    winner = None

    while rounds < MAX_ROUNDS and not success:
        rounds += 1
        for proposer in order:
            targets = [a for a in order if a is not proposer and not a.stalled]
            if not targets:
                transcript.append(f"R{rounds}:A{proposer.aid} AUTO (no targets)")
                return proposer, True, "ALT-OFFER-none", transcript

            recipient = min(targets, key=lambda x: x.score)
            decision_time = round(random.uniform(0.01, 0.06), 2)
            util_proposer = round(random.uniform(0.4, 1.0), 2)
            offer = 1 if proposer.favors >= 1 and random.random() < 0.45 else 0

            # PROPOSE
            transcript.append(
                f"R{rounds}:t={decision_time:.2f} | A{proposer.aid}→A{recipient.aid}: "
                f"PROPOSE (offer={offer}, util={util_proposer:.2f}, favors={proposer.favors}, energy={int(proposer.energy)})"
            )

            utility_responder = random.random()
            accept_threshold = 0.5 + (recipient.accept_prob - 0.5) * 0.6
            accept_prob = round(accept_threshold, 2)

            if utility_responder > accept_threshold or recipient.energy < 25:
                # ACCEPT
                if offer > 0:
                    proposer.favors -= offer
                    recipient.favors += offer
                recipient.accept_prob = min(0.95, recipient.accept_prob + 0.03)
                proposer.accept_prob = min(0.95, proposer.accept_prob + 0.01)

                transcript.append(
                    f"R{rounds}:t={decision_time + 0.02:.2f} | A{recipient.aid}→A{proposer.aid}: "
                    f"ACCEPT (prob={accept_prob:.2f}, util={utility_responder:.2f}, energy={int(recipient.energy)})"
                )

                winner = proposer
                success = True
                detail = (
                    f"ALT-ACCEPT proposer={proposer.aid} offer={offer} to={recipient.aid} rounds={rounds}"
                )
                break
            else:
                # REJECT
                recipient.accept_prob = max(0.05, recipient.accept_prob - 0.02)
                proposer.accept_prob = max(0.05, proposer.accept_prob - 0.01)
                transcript.append(
                    f"R{rounds}:t={decision_time + 0.03:.2f} | A{recipient.aid}→A{proposer.aid}: "
                    f"REJECT (prob={accept_prob:.2f}, util={utility_responder:.2f}, energy={int(recipient.energy)})"
                )

        if success:
            break

    if not success:
        winner = max(group, key=lambda a: a.score)
        for a in group:
            if a is not winner:
                a.penalize(amount=3)
        transcript.append(
            f"TIMEOUT→FALLBACK | WIN=A{winner.aid} after {MAX_ROUNDS} rounds "
            f"(fallback util={random.uniform(0.3, 0.8):.2f})"
        )
        detail = f"ALT-FALLBACK-PRIORITY rounds={MAX_ROUNDS}"

    return winner, success, detail, transcript

def formal_negotiation(a, b, corridor_pos, protocol="ALT-OFFER"):
    """
    Implements Alternating Offers protocol between two agents.
    Now with IEEE-grade utility-augmented transcript logging.
    Returns: (winner, success, detail, transcript)
    """
    rounds = 0
    MAX_ROUNDS = 3
    transcript = []  # log of all message exchanges

    proposer, responder = (a, b) if a.score <= b.score else (b, a)

    while rounds < MAX_ROUNDS:
        rounds += 1

        # Simulate decision latency and random utility
        decision_time = round(random.uniform(0.01, 0.06), 2)
        util_proposer = round(random.uniform(0.4, 1.0), 2)

        # Log proposal with utility & favors snapshot
        transcript.append(
            f"R{rounds}:t={decision_time:.2f} | A{proposer.aid}→A{responder.aid}: "
            f"PROPOSE (util={util_proposer:.2f}, favors={proposer.favors}, energy={int(proposer.energy)})"
        )

        # Acceptance decision based on learned probability
        utility_responder = random.random()
        accept_threshold = 0.5 + (responder.accept_prob - 0.5) * 0.6
        accept_prob = round(accept_threshold, 2)

        if utility_responder > accept_threshold or responder.energy < 25:
            # ACCEPT
            transcript.append(
                f"R{rounds}:t={decision_time + 0.02:.2f} | A{responder.aid}→A{proposer.aid}: "
                f"ACCEPT (prob={accept_prob:.2f}, util={utility_responder:.2f}, energy={int(responder.energy)})"
            )
            responder.accept_prob = min(0.95, responder.accept_prob + 0.03)
            proposer.accept_prob = min(0.95, proposer.accept_prob + 0.01)
            detail = f"ACCEPT rounds={rounds} util_responder={utility_responder:.2f}"
            return proposer, True, detail, transcript
        else:
            # REJECT and alternate proposer
            transcript.append(
                f"R{rounds}:t={decision_time + 0.03:.2f} | A{responder.aid}→A{proposer.aid}: "
                f"REJECT (prob={accept_prob:.2f}, util={utility_responder:.2f}, energy={int(responder.energy)})"
            )
            responder.accept_prob = max(0.05, responder.accept_prob - 0.02)
            proposer.accept_prob = max(0.05, proposer.accept_prob - 0.01)
            proposer, responder = responder, proposer  # alternate turn

    # Fallback
    fallback = max([a, b], key=lambda x: x.score)
    for loser in [a, b]:
        if loser is not fallback:
            loser.penalize(amount=15)

    transcript.append(
        f"TIMEOUT→FALLBACK | WIN=A{fallback.aid} after {MAX_ROUNDS} rounds "
        f"(fallback util={random.uniform(0.3, 0.8):.2f})"
    )
    detail = f"TIMEOUT-FALLBACK rounds={MAX_ROUNDS}"
    return fallback, False, detail, transcript


# --- Persistence & CSV helpers ---
def load_memory():
    if os.path.exists(STATE_JSON):
        try:
            with open(STATE_JSON,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_memory(pacs):
    data = {str(a.aid): {"accept_prob": a.accept_prob, "favors": a.favors} for a in pacs}
    try:
        with open(STATE_JSON,"w",encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def init_csv():
    try:
        # negotiation_log now includes 'location' field
        with open(NEG_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","step","protocol","agents","winner","success","detail","location"
            ])
    except Exception:
        pass
    try:
        # metrics_log includes neg_rounds
        with open(MET_CSV,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([
                "timestamp","step","pellets","conflicts","neg_success","neg_rounds","avg_wait","fairness","scores"
            ])
    except Exception:
        pass

def export_logs_zip():
    try:
        with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
            if os.path.exists(NEG_CSV):
                z.write(NEG_CSV, arcname=os.path.basename(NEG_CSV))
            if os.path.exists(MET_CSV):
                z.write(MET_CSV, arcname=os.path.basename(MET_CSV))
            if os.path.exists(NEG_JSON):
                z.write(NEG_JSON, arcname=os.path.basename(NEG_JSON))
        print("Exported logs to", ZIP_OUT)
    except Exception as e:
        print("Export error:", e)

# --- Game loop ---
def game_loop(screen, memory):
    global GAME_OVER
    # ...
    blink_timer = 0
    last_locked_count = 0
    last_locks_by_agent = {1: 0, 2: 0, 3: 0}
    agent_blink_timers = {1: 0, 2: 0, 3: 0}  # individual flash timers when dead
    g=[list(r) for r in MAZE]
    pellets={Pos(r,c) for r in range(ROWS) for c in range(COLS) if g[r][c] in ('.','o','C')}
    # environment / referee for locks on shared corridors
    env = Environment(g)
    TILE,OX,OY=compute_layout(screen)
    load=lambda f,size:(load_img_safe(os.path.join(SPRITES,f), size))
    pacs=[
        Agent(1,Pos(1,1),load("pacman_yellow.png",(TILE,TILE)),keyboard=True,memory=memory),
        Agent(2,Pos(1,10),load("pacman_blue.png",(TILE,TILE)),memory=memory),
        Agent(3,Pos(11,9),load("pacman_green.png",(TILE,TILE)),memory=memory),
    ]
    #  3 Ghosts safely positioned on walkable space
    ghosts = [
        Ghost(Pos(5, 8), load_img_safe(os.path.join(GHOSTS, "ghost_red.png"), (TILE, TILE))),
        Ghost(Pos(9, 20), load_img_safe(os.path.join(GHOSTS, "ghost_teal.png"), (TILE, TILE))),
        Ghost(Pos(23, 35), load_img_safe(os.path.join(GHOSTS, "ghost_green.png"), (TILE, TILE))),
    ]

    print("Ghost spawn positions:", [gh.pos for gh in ghosts])
    PEL = load_img_safe(os.path.join(SPRITES,"pellet.png"), (max(3,TILE//8), max(3,TILE//8)))
    POW = load_img_safe(os.path.join(SPRITES,"power.png"), (max(5,TILE//5), max(5,TILE//5)))
    WALL = load_img_safe(os.path.join(SPRITES,"wall.png"), (TILE, TILE))
    corr = CorridorManager(g)

    font = pygame.font.SysFont("Consolas", 16)
    header_font = pygame.font.SysFont("Consolas", 20)
    clock = pygame.time.Clock()

    # Live log data structure: newest-first
    live_log = deque(maxlen=1000)
    live_log_offset = 0   # 0 => show newest entries at top
    auto_scroll = True    # if True, new entries snap to top (offset=0)

    last_neg = ""                  # one-line debug
    step, playing, protocol = 0, False, "ALT-OFFER"
    metrics = {"conflicts":0,"neg_success":0,"total_wait":0}

    init_csv()

    running = True
    GAME_OVER = False
    while running:
        dt = clock.tick(FPS)
        # EVENTS
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                if ev.key == pygame.K_F11:
                    global fullscreen
                    fullscreen = not fullscreen
                    info = pygame.display.Info()
                    if fullscreen:
                        screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode(WINDOWED_SIZE, pygame.RESIZABLE)
                    # recalc tile & reload scaled assets
                    TILE, OX, OY = compute_layout(screen)
                    # rescale pac images & other assets
                    for a in pacs:
                        a.img = load_img_safe(os.path.join(SPRITES, f"pacman_{['yellow','blue','green'][a.aid-1]}.png"), (TILE,TILE))
                    PEL = load_img_safe(os.path.join(SPRITES,"pellet.png"), (max(3,TILE//8), max(3,TILE//8)))
                    POW = load_img_safe(os.path.join(SPRITES,"power.png"), (max(5,TILE//5), max(5,TILE//5)))
                    WALL = load_img_safe(os.path.join(SPRITES,"wall.png"), (TILE,TILE))
                if ev.key == pygame.K_SPACE:
                    # toggle pause only if not game over; if game over allow replay via R
                    if not GAME_OVER:
                        playing = not playing
                if ev.key == pygame.K_r:
                    # reset global game over on restart
                    GAME_OVER = False
                    return "RESTART"
                if ev.key == pygame.K_p:
                    protocol = "PRIORITY"
                if ev.key == pygame.K_o:
                    protocol = "ALT-OFFER"
                if ev.key == pygame.K_e:
                    export_logs_zip()
                # keyboard scrolling for log
                if ev.key == pygame.K_PAGEUP:
                    # scroll older entries up (toward older)
                    max_lines = ( (max(260, min(ROWS*TILE - 20, screen.get_height() - OY - 40)) - (36 + len(pacs)*100 + 36)) // 18 )
                    if max_lines < 1: max_lines = 6
                    live_log_offset = min(max(0, len(live_log)-max_lines), live_log_offset + max(1, max_lines//2))
                    auto_scroll = False
                if ev.key == pygame.K_PAGEDOWN:
                    live_log_offset = max(0, live_log_offset - max(1, ( ( (max(260, min(ROWS*TILE - 20, screen.get_height() - OY - 40)) - (36 + len(pacs)*100 + 36)) // 18 ) )//2))
                    if live_log_offset == 0:
                        auto_scroll = True
                if ev.key == pygame.K_HOME:
                    live_log_offset = 0
                    auto_scroll = True
                if ev.key == pygame.K_END:
                    live_log_offset = max(0, len(live_log) - 1)
                    auto_scroll = False

                keydirs = {
                    pygame.K_UP:(-1,0), pygame.K_w:(-1,0),
                    pygame.K_DOWN:(1,0), pygame.K_s:(1,0),
                    pygame.K_LEFT:(0,-1), pygame.K_a:(0,-1),
                    pygame.K_RIGHT:(0,1), pygame.K_d:(0,1)
                }
                if ev.key in keydirs:
                    pacs[0].next = keydirs[ev.key]
            if ev.type == pygame.KEYUP:
                if ev.key in (pygame.K_UP,pygame.K_DOWN,pygame.K_LEFT,pygame.K_RIGHT,pygame.K_w,pygame.K_a,pygame.K_s,pygame.K_d):
                    pacs[0].next = None

            if ev.type == pygame.MOUSEWHEEL:
                # scroll log with mouse wheel
                # positive y -> scroll up (show newer) => decrease offset
                # negative y -> scroll down (show older) => increase offset
                # compute max_lines to clamp offset
                panel_w = 320
                panel_margin = 20
                panel_x = OX + COLS*TILE + panel_margin
                panel_y = OY
                panel_h = max(260, min(ROWS*TILE - 20, screen.get_height() - panel_y - 40))
                sep_top = panel_y + 36 + len(pacs)*100
                log_area_y = sep_top + 8 + 28
                log_area_h = panel_y + panel_h - log_area_y - 16
                max_lines = max(1, log_area_h // 18)
                if ev.y > 0:
                    # scroll toward newest
                    live_log_offset = max(0, live_log_offset - abs(ev.y))
                else:
                    max_off = max(0, max(0, len(live_log) - max_lines))
                    live_log_offset = min(max_off, live_log_offset + abs(ev.y))
                auto_scroll = (live_log_offset == 0)

        # UPDATE (only while playing and not game-over)
        if playing and not GAME_OVER:
            step += 1
            ghost_positions = [gh.pos for gh in ghosts]

            # DECIDE
            desired = {}
            for a in pacs:
                desired[a] = a.decide(pellets, g, ghost_positions, others=pacs)

            # --- Conflict Pre-Detection ---
            import itertools  # make sure this is imported at the top of your file

            predicted_paths = {a: a.predict_next_move(pellets, g, pacs) for a in pacs}

            for a1, a2 in itertools.combinations(pacs, 2):
                if not a1.stalled and not a2.stalled:
                    # Check for overlapping predicted moves (next 1–2 steps)
                    common = set(predicted_paths[a1]) & set(predicted_paths[a2])
                    for pos in common:
                        if g[pos.r][pos.c] == 'C':  # shared corridor
                            env.trigger_conflict(a1, a2, pos)

            # CONFLICTS & NEGOTIATIONS
            posmap = defaultdict(list)
            for a,d in desired.items():
                posmap[d].append(a)
            # --- Conflict Processing (consolidated, extended logging) ---
            for pos, group in posmap.items():
                if len(group) > 1:
                    # Increment conflict count and blink effect
                    metrics["conflicts"] += 1
                    blink_timer = max(blink_timer, FPS // 2)

                    conflict_id = time.time_ns()
                    transcript = []  # record of offers/messages

                    # --- Negotiation Resolution ---
                    if len(group) == 2:
                        a, b = group
                        winner, success, detail, transcript = formal_negotiation(a, b, pos, protocol)
                    else:
                        winner, success, detail, transcript = negotiate_group(group, pos, protocol, pellets, g)

                    try:
                        # --- CSV logging ---
                        with open(NEG_CSV, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                step,
                                protocol,
                                ",".join([str(a.aid) for a in group]),
                                winner.aid if winner else -1,
                                int(success),
                                detail,
                                f"Conflict@{pos.r},{pos.c}"
                            ])

                        # --- JSON Transcript Logging ---
                        json_entry = {
                            "conflict_id": metrics.get("conflicts", 0),
                            "step": step,
                            "protocol": protocol,
                            "corridor": [pos.r, pos.c],
                            "agents": [a.aid for a in group],
                            "winner": winner.aid if winner else None,
                            "success": bool(success),
                            "detail": detail,
                            "rounds": sum("R" in line for line in transcript),
                            "transcript": transcript,
                            "winner_energy": int(winner.energy if winner else 0),
                            "fairness_delta": round(random.uniform(0.0, 0.2), 3),
                            "timestamp": time.strftime("%H:%M:%S"),
                        }

                        # Append to JSON file
                        try:
                            with open(NEG_JSON, "r", encoding="utf-8") as jf:
                                data = json.load(jf)
                        except (FileNotFoundError, json.JSONDecodeError):
                            data = []

                        data.append(json_entry)
                        with open(NEG_JSON, "w", encoding="utf-8") as jf:
                            json.dump(data, jf, indent=2)

                    except Exception as e:
                        print(f"[WARN] Log write failed: {e}")

                    # --- Compute negotiation rounds if found in detail ---
                    rounds_found = 0
                    if "rounds=" in detail:
                        try:
                            rounds_found = int(detail.split("rounds=")[1].split()[0])
                        except Exception:
                            rounds_found = 0
                    metrics["neg_rounds"] = metrics.get("neg_rounds", 0) + rounds_found
                    if success:
                        metrics["neg_success"] += 1

                    # --- Per-agent action and penalties ---
                    for a in group:
                        if a is winner:
                            # Agent wins and acquires corridor
                            if g[pos.r][pos.c] == 'C':
                                env.lock(pos, a.aid)
                                visual_owner[pos] = (a.aid, FPS // 3)  # linger 0.3s
                            a.wait = max(0, a.wait - 1)
                        else:
                            a.wait += 1
                            a.penalize(amount=6)
                            metrics["total_wait"] += 1
                            agent_blink_timers[a.aid] = FPS // 2  # individual blink

                    # --- Build detailed log line ---
                    timestamp = time.strftime("%H:%M:%S")
                    transcript_str = ";".join(transcript) if transcript else "-"
                    log_entry = (
                        f"[{timestamp}] Step {step}: Conflict@({pos.r},{pos.c}) | "
                        f"Protocol={protocol} | Winner=A{winner.aid} | "
                        f"{'SUCCESS' if success else 'FAIL'} ({detail})"
                    )

                    # Console + in-game live log
                    print(log_entry)
                    live_log.appendleft(log_entry)
                    if transcript:
                        for msg in reversed(transcript[-3:]):  # show last 3 exchanges for brevity
                            live_log.appendleft(f"    ↳ {msg}")
                        if auto_scroll:
                            live_log_offset = 0
                    if auto_scroll:
                        live_log_offset = 0
                    last_neg = log_entry

                    # --- CSV: single, consolidated write ---
                    try:
                        with open(NEG_CSV, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
                                step,
                                conflict_id,
                                protocol,
                                ",".join(str(a.aid) for a in group),
                                winner.aid,
                                int(success),
                                detail,
                                f"{pos.r},{pos.c}",
                                len(group),
                                rounds_found,
                                transcript_str
                            ])
                    except Exception as e:
                        print("CSV write error:", e)

                    # --- Penalize losers ---
                    for a in group:
                        if a is not winner:
                            a.wait += 1
                            a.penalize()
                            metrics["total_wait"] += 1

                    # --- Generate readable log entry ---
                    timestamp = time.strftime("%H:%M:%S")
                    last_neg = (
                        f"[{timestamp}] Step {step}: Conflict@({pos.r},{pos.c}) | "
                        f"Protocol={protocol} | Winner=A{winner.aid} | "
                        f"{'SUCCESS' if success else 'FAIL'} ({detail})"
                    )
                    print(last_neg)
                    live_log.appendleft(last_neg)
                    if auto_scroll:
                        live_log_offset = 0

                    # console debug
                    print(last_neg)
                    # add to live log (newest-first)
                    live_log.appendleft(last_neg)
                    # auto-scroll behavior: if user hasn't scrolled, keep auto-scroll at top
                    if auto_scroll:
                        live_log_offset = 0

                    # CSV write
                    try:
                        with open(NEG_CSV, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"), step, protocol,
                                ",".join(str(x.aid) for x in group), winner.aid, int(success), detail
                            ])
                    except Exception:
                        pass

                    # negotiation penalties
                    for a in group:
                        if a is not winner:
                            a.wait += 1
                            a.penalize()
                    if success:
                        metrics["neg_success"] += 1

            # --- Optional periodic test log every ~200 steps (for visibility) ---
            if step % 200 == 0 and playing:
                test_entry = f"[{time.strftime('%H:%M:%S')}] Step {step}: TEST-EVENT -> A{random.randint(1, 3)} (SUCCESS)"
                if step % 100 == 0:
                    live_log.appendleft(
                        f"[{time.strftime('%H:%M:%S')}] Conflicts={metrics['conflicts']} | Success={metrics['neg_success']}"
                    )
                    if auto_scroll:
                        live_log_offset = 0
                print(test_entry)
                live_log.appendleft(test_entry)
                if auto_scroll:
                    live_log_offset = 0

            # APPLY MOVES
            for a in pacs:
                if a.stalled:
                    continue
                nxt = desired.get(a, a.pos)

                # corridor token system
                if g[nxt.r][nxt.c] == 'C':
                    # Attempt to acquire lock in environment first
                    state = env.shared_routes.get(nxt, "UNLOCKED")
                    if state == "UNLOCKED":
                        env.lock(nxt, a.aid)
                    elif state != f"LOCKED_BY_{a.aid}":
                        # Locked by someone else — conflict wait
                        a.wait += 1
                        a.penalize()
                        metrics["total_wait"] += 1
                        continue

                # avoid agents
                if any(other.pos == nxt and other is not a for other in pacs):
                    a.wait += 1;
                    metrics["total_wait"] += 1;
                    continue

                # avoid ghost position
                if any(gp == nxt for gp in [gh.pos for gh in ghosts]):
                    a.wait += 1;
                    a.penalize();
                    metrics["total_wait"] += 1;
                    continue

                # move and grant corridor ownership
                corr.grant(nxt, a.aid)
                a.pos = nxt
                a.collect(pellets, g)

            # release old corridor locks when agents move away
            corr.release(pacs)
            for pos, owner in list(env.shared_routes.items()):
                if "LOCKED_BY_" in owner:
                    aid = int(owner.split("_")[-1])
                    if not any(a.pos == pos for a in pacs if a.aid == aid):
                        env.unlock(pos, aid)

            # ghosts update
            pac_positions = [p.pos for p in pacs if p.energy > 0]
            for gh in ghosts:
                gh.update(pac_positions, g)

            # --- Ghost-Agent collision handling (safe + cooldown) ---
            COLLISION_COOLDOWN = 40  # frames (~2 seconds)
            if not hasattr(game_loop, "last_hits"):
                game_loop.last_hits = {a.aid: -COLLISION_COOLDOWN for a in pacs}

            for gh in ghosts:
                if not gh.alive:
                    continue
                for a in pacs:
                    if a.stalled or not hasattr(a, "pos"):
                        continue
                    # Detect exact overlap (same tile)
                    if a.pos == gh.pos:
                        # Prevent multiple hits within cooldown window
                        if step - game_loop.last_hits[a.aid] < COLLISION_COOLDOWN:
                            continue

                        a.penalize(amount=15)
                        a.energy = max(0, a.energy - 30)

                        # Record hit frame
                        game_loop.last_hits[a.aid] = step

                        # Kill ghost only if it attacked successfully (keeps balance)
                        gh.kill()

                        # If energy drops to zero -> remove from game
                        if a.energy <= 0:
                            a.energy = 0
                            a.stalled = True
                            live_log.appendleft(
                                f"[{time.strftime('%H:%M:%S')}] A{a.aid} has died and is removed from the maze!"
                            )
                            if auto_scroll:
                                live_log_offset = 0

                        # Safely remove dead agents (outside this loop, after processing all collisions)

            # --- Remove permanently dead agents ---
            alive_before = len(pacs)
            #pacs = [a for a in pacs if not a.stalled or a.energy > 0]
            if len(pacs) < alive_before:
                print(f"[INFO] Removed dead agents. Remaining: {len(pacs)}")

            # --- Game Over Condition: all agents dead ---
            if all(a.energy <= 0 for a in pacs):
                playing = False
                GAME_OVER = True
                # clear any motion or decisions
                for a in pacs:
                    a.next = None
                # add log entry
                timestamp = time.strftime("%H:%M:%S")
                live_log.appendleft(f"[{timestamp}] All agents are dead! Simulation ended. (Game over)")
                if auto_scroll:
                    live_log_offset = 0
                print("[END] All agents have died. Game Over.")

            # --- Metrics Tracking ---
            avg_wait = sum(a.wait for a in pacs) / max(1, len(pacs))
            fair_val = fairness([a.score for a in pacs])

            # Write step-level metrics
            try:
                with open(MET_CSV, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        step,
                        len(pellets),
                        metrics["conflicts"],
                        metrics["neg_success"],
                        f"{avg_wait:.3f}",
                        f"{fair_val:.4f}",
                        "/".join(str(a.score) for a in pacs),
                    ])
            except Exception as e:
                print("Metrics write error:", e)

            # --- Update visible metrics immediately ---
            display_conflicts = metrics["conflicts"]
            display_success = metrics["neg_success"]
            display_wait = avg_wait
            display_fair = fair_val

            if not pellets or step > 8000:
                # --- Proper Game Over Freeze ---
                playing = False
                GAME_OVER = True
                for a in pacs:
                    a.next = None
                    a.stalled = False  # stop movement, no more energy drain

                timestamp = time.strftime("%H:%M:%S")

                # --- End conditions ---
                if not pellets:
                    # GOOD GAME ENDING
                    live_log.appendleft(f"[{timestamp}] All pellets cleared! Good Game! (SIMULATION COMPLETE)")
                    live_log.appendleft(f"[{time.strftime('%H:%M:%S')}] Simulation ended under {protocol} protocol.")
                    if auto_scroll:
                        live_log_offset = 0
                    print("[END] All pellets cleared — GOOD GAME!")
                    game_end_status = "GOOD GAME!"
                else:
                    # Time-out fallback
                    live_log.appendleft(f"[{timestamp}] Step limit reached — simulation stopped. (Game over)")
                    game_end_status = "GAME OVER"

                if auto_scroll:
                    live_log_offset = 0

                print(f"[END] Game finished at step {step}. Agents frozen.")

                # Store for header display
                global END_MESSAGE
                END_MESSAGE = game_end_status

        # DRAW
        screen.fill(BLACK)
        TILE, OX, OY = compute_layout(screen)
        header_font = pygame.font.SysFont("Consolas", 20)
        font_small = pygame.font.SysFont("Consolas", 16)
        # --- Header with Protocol Badge (aligned, no emoji artifacts) ---
        prot_color = (255, 150, 50) if protocol == "PRIORITY" else (80, 200, 255)
        mode_text = "GAME OVER" if GAME_OVER else ("Playing" if playing else "Paused")

        # Render 'Protocol:' text
        prefix_text = "Protocol:"
        prefix_surface = header_font.render(prefix_text, True, NEON)
        prefix_x, prefix_y = 20, 8
        screen.blit(prefix_surface, (prefix_x, prefix_y))

        # Position references
        prefix_width = prefix_surface.get_width()
        badge_x = prefix_x + prefix_width + 18  # nice spacing
        badge_y = prefix_y + prefix_surface.get_height() // 2 - 2  # better vertical alignment

        # --- Glow Pulse (dim when paused or game over) ---
        pulse_strength = 0.3 if not playing or GAME_OVER else 0.6
        pulse = int(128 + 127 * math.sin(step / 8))
        glow_color = tuple(min(255, int(c + pulse * pulse_strength)) for c in prot_color)

        pygame.draw.circle(screen, glow_color, (badge_x, badge_y + 6), 6)
        pygame.draw.circle(screen, prot_color, (badge_x, badge_y + 6), 6, 2)

        # --- Replace emoji with vector icons ---
        if protocol == "PRIORITY":
            # stylized gear icon (drawn manually)
            center = (badge_x, badge_y + 6)
            for i in range(6):
                angle = math.radians(i * 60)
                dx = int(9 * math.cos(angle))
                dy = int(9 * math.sin(angle))
                pygame.draw.line(screen, prot_color, center, (center[0] + dx, center[1] + dy), 2)
            pygame.draw.circle(screen, prot_color, center, 3)
        else:
            # stylized chat bubble icon (drawn manually)
            bubble_rect = pygame.Rect(badge_x - 5, badge_y + 1, 12, 9)
            pygame.draw.rect(screen, prot_color, bubble_rect, 2, border_radius=3)
            pygame.draw.polygon(screen, prot_color,
                                [(badge_x + 3, badge_y + 10), (badge_x + 7, badge_y + 10), (badge_x + 5, badge_y + 13)],
                                2)

        # --- Protocol Name ---
        protocol_surface = header_font.render(protocol, True, prot_color)
        screen.blit(protocol_surface, (badge_x + 20, prefix_y))

        # --- Mode Text ---
        mode_surface = header_font.render(f"| {mode_text}", True, NEON)
        screen.blit(mode_surface, (badge_x + 30 + protocol_surface.get_width(), prefix_y))

        visual_owner = {}  # position -> (agent_id, remaining_frames)

        if blink_timer > 0:
            blink_timer -= 1

        for aid in agent_blink_timers:
            if agent_blink_timers[aid] > 0:
                agent_blink_timers[aid] -= 1

        # maze drawing (tiles)
        for r in range(ROWS):
            for c in range(COLS):
                ch = g[r][c]; x = OX + c*TILE; y = OY + r*TILE
                if ch == '#':
                    # draw wall tile
                    pygame.draw.rect(screen,(20,24,36),(x,y,TILE,TILE), border_radius=max(2, TILE//8))
                elif ch == '.':
                    pygame.draw.circle(screen, (255,170,40), (x+TILE//2, y+TILE//2), max(2, TILE//10))
                elif ch == 'o':
                    pygame.draw.circle(screen, (255,80,80), (x+TILE//2, y+TILE//2), max(3, TILE//6))
                elif ch == 'C':
                    # simple glow border
                    pygame.draw.rect(screen, (12,40,60), (x+1,y+1,TILE-2,TILE-2), border_radius=max(2, TILE//8))
                    pygame.draw.rect(screen, NEON, (x+3,y+3,TILE-6,TILE-6), 2, border_radius=max(2, TILE//10))

        # --- Visual linger for corridor ownership (prevents flicker) ---
        for pos, (aid, timer) in list(visual_owner.items()):
            if timer <= 0:
                visual_owner.pop(pos)
            else:
                visual_owner[pos] = (aid, timer - 1)
                color = {
                    1: (255, 220, 40),
                    2: (80, 180, 255),
                    3: (80, 255, 160)
                }.get(aid, (180, 180, 180))
                pygame.draw.rect(
                    screen, color,
                    (OX + pos.c * TILE + 2, OY + pos.r * TILE + 2, TILE - 4, TILE - 4),
                    2, border_radius=4
                )

        # draw ghosts & pacs
        for gh in ghosts: gh.draw(screen, OX, OY, TILE)
        for a in pacs: a.draw(screen, OX, OY, TILE)

        # --- Right-side single panel (Agents + Live Log) ---
        panel_w = 320
        panel_margin = 20
        panel_x = OX + COLS*TILE + panel_margin
        panel_y = OY
        panel_h = max(260, min(ROWS*TILE - 20, screen.get_height() - panel_y - 40))

        # background
        pygame.draw.rect(screen, (18,18,28), (panel_x - 8, panel_y - 8, panel_w + 16, panel_h + 16), border_radius=10)
        pygame.draw.rect(screen, (40,90,160), (panel_x - 8, panel_y - 8, panel_w + 16, panel_h + 16), 2, border_radius=10)

        # Agents header
        screen.blit(header_font.render("Agents", True, ORANGE), (panel_x, panel_y))
        agent_block_h = 100  # slightly taller to fit color line
        # render agents block
        color_labels = {1: ("Yellow", (255, 220, 40)),
                        2: ("Blue", (90, 150, 255)),
                        3: ("Green", (90, 255, 150))}
        for i, a in enumerate(pacs):
            ay = panel_y + 36 + i * agent_block_h
            col = RED if a.stalled else WHITE
            label, rgb = color_labels.get(a.aid, ("", WHITE))
            status = "DEAD" if a.energy <= 0 else ("Stalled" if a.stalled else "Active")
            status_color = RED if a.energy <= 0 else (ORANGE if a.stalled else WHITE)
            screen.blit(font_small.render(f"A{i + 1} — {status}", True, status_color),
                        (panel_x + 6, ay))

            # show color label and small dot
            screen.blit(font_small.render(label, True, rgb), (panel_x + 180, ay))
            pygame.draw.circle(screen, rgb, (panel_x + 170, ay + 8), 6)
            screen.blit(font_small.render(f"Score:  {a.score:3d}", True, col), (panel_x + 10, ay + 20))
            screen.blit(font_small.render(f"Wait:   {a.wait:2d}", True, col), (panel_x + 10, ay + 40))
            screen.blit(font_small.render(f"Energy: {int(a.energy):3d}", True, col), (panel_x + 10, ay + 60))
            screen.blit(font_small.render(f"Favors: {a.favors}", True, col), (panel_x + 170, ay + 40))
            # small vertical energy bar at side
            bar_x = panel_x + panel_w - 28
            bar_y = ay + 22
            pygame.draw.rect(screen, (28, 28, 36), (bar_x, bar_y, 12, 52))
            fill_h = int((a.energy / 100.0) * 50)
            if a.energy > 70:
                bcol = (0, 220, 120)
            elif a.energy > 30:
                bcol = (230, 200, 30)
            else:
                bcol = (230, 80, 80)
            if fill_h > 0:
                pygame.draw.rect(screen, bcol, (bar_x + 2, bar_y + (50 - fill_h) + 2, 8, fill_h))

        # --- Separator below Agents ---
        sep_y = panel_y + 36 + len(pacs) * agent_block_h
        pygame.draw.line(screen, (40, 60, 80), (panel_x, sep_y), (panel_x + panel_w - 4, sep_y), 2)

        # --- Ghosts Status Section ---
        ghost_status_y = sep_y + 12
        screen.blit(header_font.render("Ghosts Status", True, ORANGE), (panel_x, ghost_status_y))

        # --- Ghosts Status Section (colored) ---
        ghost_colors = {
            0: ("Red", (255, 80, 80)),
            1: ("Blue", (80, 180, 255)),
            2: ("Green", (80, 255, 160)),
        }
        for i, gh in enumerate(ghosts):
            label, gcol = ghost_colors.get(i, ("?", WHITE))
            status = "Alive" if gh.alive else f"Dead ({gh.respawn_timer // FPS}s)"
            text_color = GREEN if gh.alive else RED
            line_y = ghost_status_y + 26 + i * 22

            pygame.draw.circle(screen, gcol, (panel_x + 10, line_y + 6), 6)
            screen.blit(font_small.render(f"{label}", True, gcol), (panel_x + 24, line_y))
            screen.blit(font_small.render(f"— {status}", True, text_color), (panel_x + 90, line_y))

        # --- Separator below Ghosts ---
        sep2_y = ghost_status_y + 26 + len(ghosts) * 20 + 8

        # --- Conflict Status Section ---
        conflict_y = sep2_y + 20  # more vertical spacing below ghosts
        # --- Conflict Status Header with Blink ---
        header_text = "Conflict Status"
        # --- Enhanced Conflict Status with pulsing indicator ---
        if protocol == "PRIORITY":
            prot_color = (255, 170, 60)  # amber/orange glow
        else:
            prot_color = (0, 220, 255)  # cyan/blue glow

        # --- Pulse intensity ---
        pulse_phase = abs(math.sin(step / 6))
        pulse_radius = int(5 + 3 * pulse_phase)
        pulse_alpha = int(120 + 100 * pulse_phase)

        # Render text slightly shifted for alignment
        title_x = panel_x + 24
        title_y = conflict_y

        # --- Draw pulsing glow indicator (like a heartbeat) ---
        if blink_timer > 0:
            pulse_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(pulse_surf, (*prot_color, pulse_alpha), (10, 10), pulse_radius)
            screen.blit(pulse_surf, (panel_x - 2, conflict_y + 4))

        # --- Text glow effect ---
        text_color = prot_color if blink_timer > 0 else ORANGE
        screen.blit(header_font.render("Conflict Status", True, text_color), (title_x, title_y))

        # --- Corridor lock stats ---
        locked_corrs = [pos for pos, state in env.shared_routes.items() if state != "UNLOCKED"]
        locked_count = len(locked_corrs)

        # Count ownership per agent
        locks_by_agent = {1: 0, 2: 0, 3: 0}
        for state in env.shared_routes.values():
            if "LOCKED_BY_" in state:
                try:
                    aid = int(state.split("_")[-1])
                    locks_by_agent[aid] += 1
                except:
                    pass
        # --- Detect changes for blink animation ---
        if locked_count != last_locked_count:
            blink_timer = FPS // 2  # global header flash

        # check each agent’s ownership count
        for aid in (1, 2, 3):
            if locks_by_agent[aid] != last_locks_by_agent.get(aid, 0):
                agent_blink_timers[aid] = FPS // 2  # individual agent flash

        # update previous state trackers
        last_locked_count = locked_count
        last_locks_by_agent = locks_by_agent.copy()

        # --- Detect changes for blink animation ---
        if locked_count != last_locked_count or locks_by_agent != last_locks_by_agent:
            blink_timer = FPS // 2  # ~0.5 sec flash when change detected

        last_locked_count = locked_count
        last_locks_by_agent = locks_by_agent.copy()

        # Line 1: total + locked count
        # --- Blinking text when locked_count changes ---
        lock_color = (0, 255, 255) if blink_timer > 0 and (step % 4 < 2) else NEON
        screen.blit(
            font_small.render(
                f"Shared Corridors: {len(env.shared_routes)} (Locked: {locked_count})",
                True, lock_color
            ),
            (panel_x + 12, conflict_y + 26)
        )

        # Line 2: ownership breakdown per agent (with colored segments)
        agent_colors = {
            1: (255, 220, 40),  # yellow
            2: (80, 180, 255),  # blue
            3: (80, 255, 160),  # green
        }
        line2_y = conflict_y + 46
        x_cursor = panel_x + 12

        for i in range(1, 4):
            label = f"A{i} owns: {locks_by_agent[i]}"
            color = agent_colors[i]

            # individual blink effect (white flash)
            if agent_blink_timers[i] > 0:
                pulse = int(128 + 127 * math.sin(step / 3))
                color = (pulse, pulse, 255)

            screen.blit(font_small.render(label, True, color), (x_cursor, line2_y))
            x_cursor += font_small.size(label)[0] + 20

        # --- Separator below Conflict Status ---
        sep3_y = conflict_y + 70  # more vertical padding below
        pygame.draw.line(screen, (40, 60, 80), (panel_x, sep3_y), (panel_x + panel_w - 4, sep3_y), 2)

        # --- Live Negotiation Log header ---
        log_title_y = sep3_y + 8
        screen.blit(header_font.render("Live Negotiation Log", True, NEON), (panel_x + 6, log_title_y))

        # wrapped log area
        log_area_x = panel_x + 6
        log_area_w = panel_w - 12
        log_area_y = log_title_y + 28
        log_area_h = panel_y + panel_h - log_area_y - 16

        # Draw a subtle inner box for log
        pygame.draw.rect(screen, (8,12,18), (log_area_x-4, log_area_y-6, log_area_w+8, log_area_h+12), border_radius=6)

        # --- Scrollable Negotiation Log Rendering (fixed wrapping + no overlap) ---
        line_height = 18
        max_lines = max(1, log_area_h // line_height)

        # Pre-wrap entries into visual lines
        entries = list(live_log)
        wrapped_lines = []
        for entry in entries:
            color = GREEN if "(SUCCESS)" in entry or "SUCCESS" in entry else RED if "(FAIL)" in entry else NEON
            words = entry.split(" ")
            cur_line = ""
            for w in words:
                test = (cur_line + " " + w).strip()
                if font_small.size(test)[0] <= log_area_w:
                    cur_line = test
                else:
                    wrapped_lines.append((cur_line, color))
                    cur_line = w
            if cur_line:
                wrapped_lines.append((cur_line, color))

        # Clamp scroll offset (now in line units)
        max_offset = max(0, len(wrapped_lines) - max_lines)
        if live_log_offset > max_offset:
            live_log_offset = max_offset
        if live_log_offset < 0:
            live_log_offset = 0

        # Select visible portion
        visible_lines = wrapped_lines[live_log_offset: live_log_offset + max_lines]

        # Draw visible lines cleanly
        y_cursor = log_area_y
        for text, color in visible_lines:
            screen.blit(font_small.render(text, True, color), (log_area_x, y_cursor))
            y_cursor += line_height

        # --- Scrollbar ---
        if len(wrapped_lines) > max_lines:
            sb_x = panel_x + panel_w - 12
            sb_y = log_area_y - 4
            sb_h = log_area_h + 8
            pygame.draw.rect(screen, (20, 20, 26), (sb_x, sb_y, 8, sb_h), border_radius=4)
            thumb_h = max(12, int(sb_h * (max_lines / len(wrapped_lines))))
            if max_offset > 0:
                thumb_y = int(sb_y + (sb_h - thumb_h) * (live_log_offset / max_offset))
            else:
                thumb_y = sb_y
            pygame.draw.rect(screen, (90, 140, 220), (sb_x + 1, thumb_y, 6, thumb_h), border_radius=3)

        # --- Footer Metrics Summary (left side) ---
        left_y = OY + ROWS * TILE + 8

        # Calculate success rate safely
        if metrics["conflicts"] > 0:
            success_rate = (metrics["neg_success"] / metrics["conflicts"]) * 100
        else:
            success_rate = 0.0

        # Step + Pellets
        screen.blit(font_small.render(f"Step: {step}   Pellets: {len(pellets)}", True, NEON), (20, left_y))

        # Conflicts, Success, and Success Rate
        conflicts = metrics.get("conflicts", 0)
        successes = metrics.get("neg_success", 0)
        success_rate = (successes / conflicts * 100) if conflicts > 0 else 0.0

        screen.blit(
            font_small.render(
                f"Conflicts: {conflicts:<3d}   Success: {successes:<3d}   Rate: {success_rate:5.1f}%",
                True, WHITE
            ),
            (20, left_y + 18)
        )

        # Average Wait and Fairness
        avg_wait = sum(a.wait for a in pacs) / max(1, len(pacs))
        fair_val = fairness([a.score for a in pacs])
        screen.blit(
            font_small.render(f"Avg Wait: {avg_wait:.2f}   Fairness: {fair_val:.3f}", True, NEON),
            (20, left_y + 36)
        )

        # last negotiation one-line debug (center / above footer)
        dbg = last_neg or "Last negotiation: --"
        dbg_w = font_small.size(dbg)[0]
        screen.blit(font_small.render(dbg, True, ORANGE), ( (screen.get_width()-dbg_w)//2, screen.get_height()-60 ))

        # footer controls
        footer = "[Space]Play/Pause   [R]Restart   [P/O]Protocol   [E]Export Logs   [F11]Fullscreen   [ESC]Quit"
        fw = font_small.size(footer)[0]
        screen.blit(font_small.render(footer, True, WHITE), ((screen.get_width()-fw)//2, screen.get_height()-36))

        pygame.display.flip()

    # end loop
    save_memory(pacs)
    return False

# --- main launcher ---
def init_and_run():
    memory = load_memory()
    screen = pygame.display.set_mode(WINDOWED_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("Multi-Agent Pac-Men — Full UI")
    # warm assets to avoid first-draw hiccups
    TILE, OX, OY = compute_layout(screen)
    # run
    while True:
        res = game_loop(screen, memory)
        if res == "RESTART":
            memory = load_memory()
            continue
        else:
            break
    pygame.quit()

if __name__ == "__main__":
    init_and_run()
