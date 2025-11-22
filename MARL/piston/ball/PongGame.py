import argparse
import math
import os
import random
import time
from collections import deque, namedtuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
WIDTH, HEIGHT = 800, 600
FPS = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RL hyperparams
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
VF_COEF = 0.5
LR_ACTOR = 2.5e-4
LR_CRITIC = 2.5e-4
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
MINI_BATCH = 256

BALL_SPEED_GROW = 0.6
MAX_BALL_SPEED = 20.0
BASE_BALL_SPEED = 9.0
SPIN_FACTOR = 0.18

# Collection / training sizes (reduce for faster/smaller runs)
FRAME_STACK = 4
OBS_SINGLE = 7   # ball + paddle positions + prediction (as before)
OBS_DIM = OBS_SINGLE * FRAME_STACK
NUM_ACTIONS = 3

STEPS_PER_UPDATE = 2048         # environment steps collected per update
TOTAL_UPDATES = 300             # number of updates (reduce for faster experiments)
BC_TRANSITIONS = 16000          # BC pretrain transitions (increased)
BC_EPOCHS = 4
AUTOSAVE_EVERY_FRAMES = 60000
CURRICULUM_STEP_FRAMES = 40000

# Ball speed tuning (faster gameplay)
BALL_SPEED_GROW = 0.6        # rate of speed increase during curriculum updates
MAX_BALL_SPEED = 20.0        # cap for extreme speed
BASE_BALL_SPEED = 9.0        # starting ball speed (default was 6.0)
SPIN_FACTOR = 0.18           # spin strength for curve and bounce control

# Self-play pool
SELFPLAY_SAVE_EVERY_UPDATES = 5  # add checkpoint to pool every N updates
SELFPLAY_POOL_MAX = 6            # keep most recent/strongest N checkpoints

Transition = namedtuple("Transition", ["obs", "action", "logp", "reward", "done", "value"])

# Environment constants
PADDLE_W, PADDLE_H = 10, 100
BALL_SIZE = 12
BASE_BALL_SPEED = 6.0
SPIN_FACTOR = 0.15
PADDLE_ACCEL = 0.7
PADDLE_MAX_SPEED = 12.0

# ENVIRONMENT (with frame stacking compatible)
class PongEnv:
    def __init__(self, render=False, show_info=False, fullscreen=False):
        self.render = render
        self.show_info = show_info
        self.fullscreen = fullscreen
        self.width, self.height = WIDTH, HEIGHT
        self.ball_speed = BASE_BALL_SPEED
        self.pw, self.ph = PADDLE_W, PADDLE_H
        self.ball_size = BALL_SIZE
        self.spin_factor = SPIN_FACTOR
        self.player_left_name = "Player X"
        self.player_right_name = "Player Y"
        self.round_count = 1  # Track how many full matches (rounds) have been played
        # Learning improvement tracking
        self.learning_progress = {
            "Player X": {"history": [], "improvement": 0.0},
            "Player Y": {"history": [], "improvement": 0.0}
        }
        self.max_points = 5  # Points required to win one round
        # history of winners per round (most recent last)
        self.round_history = []  # list of tuples (round_number, "Left"/"Right")
        self.show_round_winner = None  # active winner text to display
        self.winner_timer = 0.0  # how long to show winner banner (seconds)
        self.freeze = False  # temporarily pause movement
        self.reset()
        if render:
            pygame.init()
            self._create_window()
            pygame.display.set_caption("MARL Pong (CTDE + Self-Play)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 16)
            self.bigfont = pygame.font.SysFont("Consolas", 28)

    def _create_window(self):
        """Create or resize the window. In fullscreen, only the ball and paddles scale."""
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
            self.width, self.height = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
            self.width, self.height = WIDTH, HEIGHT

        # Scale only ball + paddles (not text/UI)
        scale_x = self.width / WIDTH
        scale_y = self.height / HEIGHT
        scale_avg = (scale_x + scale_y) / 2

        # Only enlarge gameplay objects
        self.pw = int(PADDLE_W * scale_avg)
        self.ph = int(PADDLE_H * scale_avg)
        self.ball_size = int(BALL_SIZE * scale_avg)
        self.ball_speed = BASE_BALL_SPEED * scale_avg
        self.spin_factor = SPIN_FACTOR * scale_avg

        # Keep scores and text consistent size
        self.font = pygame.font.SysFont("Consolas", 16)
        self.bigfont = pygame.font.SysFont("Consolas", 28)

        # Reset positions proportionally
        self.left_y = self.height / 2
        self.right_y = self.height / 2
        self.ball_x = self.width / 2
        self.ball_y = self.height / 2

        print(f"Window resized to {self.width}x{self.height} — paddles & ball scaled x{scale_avg:.2f}")

    def reset(self):
        self.left_y = self.height / 2
        self.right_y = self.height / 2
        self.left_v = 0.0
        self.right_v = 0.0
        self.ball_x, self.ball_y = self.width / 2, self.height / 2
        angle = random.uniform(-0.35, 0.35)
        self.ball_vx = random.choice([-1, 1]) * self.ball_speed * math.cos(angle)
        self.ball_vy = self.ball_speed * math.sin(angle)
        self.left_score = 0
        self.right_score = 0
        return self._get_obs()

    def _get_obs(self):
        pred_y = self._predict_ball_y_for_side(40)  # predict at left side X
        return np.array([
            (self.ball_x - self.width / 2) / (self.width / 2),
            (self.ball_y - self.height / 2) / (self.height / 2),
            (self.ball_vx / max(1e-6, self.ball_speed)),
            (self.ball_vy / max(1e-6, self.ball_speed)),
            (self.left_y - self.height / 2) / (self.height / 2),
            (self.right_y - self.height / 2) / (self.height / 2),
            (pred_y - self.height / 2) / (self.height / 2)
        ], dtype=np.float32)

    def _predict_ball_y_for_side(self, side_x):
        bx, by, vx, vy = self.ball_x, self.ball_y, self.ball_vx, self.ball_vy
        if vx == 0:
            return by
        t = (side_x - bx) / vx
        if t <= 0:
            return by
        pred = by + vy * t
        # reflect against horizontal walls
        while pred < 0 or pred > self.height:
            if pred < 0:
                pred = -pred
            else:
                pred = 2 * self.height - pred
        return pred

    def step(self, a_l, a_r):
        # discrete: 0 stay, 1 up, 2 down
        self.left_v = self._update_velocity(self.left_v, a_l)
        self.right_v = self._update_velocity(self.right_v, a_r)
        self.left_y += self.left_v
        self.right_y += self.right_v
        self.left_y = np.clip(self.left_y, self.ph / 2, self.height - self.ph / 2)
        self.right_y = np.clip(self.right_y, self.ph / 2, self.height - self.ph / 2)

        # advance ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # small time penalty
        r_l, r_r = -0.001, -0.001

        # bounce y walls
        if self.ball_y <= 0 or self.ball_y >= self.height:
            self.ball_vy *= -1

        # left hit?
        if self.ball_x <= 40 and abs(self.ball_y - self.left_y) <= self.ph / 2:
            offset = (self.ball_y - self.left_y) / (self.ph / 2)
            self.ball_vx = abs(self.ball_vx) + 0.8  # stronger reflection boost
            self.ball_vy += offset * self.ball_speed * self.spin_factor

            # Turbo bounce multiplier (faster after every hit)
            speed = math.hypot(self.ball_vx, self.ball_vy)
            speed = min(speed * 1.08, MAX_BALL_SPEED)  # was 1.02
            ang = math.atan2(self.ball_vy, self.ball_vx)
            self.ball_vx = speed * math.cos(ang)
            self.ball_vy = speed * math.sin(ang)
            r_l += 0.14

            ang = math.atan2(self.ball_vy, self.ball_vx)
            self.ball_vx = speed * math.cos(ang)
            self.ball_vy = speed * math.sin(ang)
            r_l += 0.14
        elif self.ball_x < 0:
            # point to right
            r_l -= 1.0; r_r += 1.0
            self.right_score += 1
            self._respawn(ball_to_left=False)

        # right hit?
        if self.ball_x >= self.width - 40 and abs(self.ball_y - self.right_y) <= self.ph / 2:
            offset = (self.ball_y - self.right_y) / (self.ph / 2)
            self.ball_vx = -abs(self.ball_vx) - 0.8  # stronger reflection boost
            self.ball_vy += offset * self.ball_speed * self.spin_factor

            # ⚡ Turbo bounce multiplier (faster after every hit)
            speed = math.hypot(self.ball_vx, self.ball_vy)
            speed = min(speed * 1.08, MAX_BALL_SPEED)  # was 1.02
            ang = math.atan2(self.ball_vy, self.ball_vx)
            self.ball_vx = speed * math.cos(ang)
            self.ball_vy = speed * math.sin(ang)
            r_r += 0.14

            ang = math.atan2(self.ball_vy, self.ball_vx)
            self.ball_vx = speed * math.cos(ang)
            self.ball_vy = speed * math.sin(ang)
            r_r += 0.14
            # Check if someone wins the round (this was originally under right hit in an earlier edit — keep consistent checks)
            # NOTE: The round end check is centralized below after score updates to ensure correctness.

        # If ball passes right boundary -> left scores
        if self.ball_x > self.width:
            r_r -= 1.0
            r_l += 1.0
            self.left_score += 1
            self._respawn(ball_to_left=True)

        # ----- Round Win Check -----
        if self.left_score >= self.max_points or self.right_score >= self.max_points:
            winner = self.player_left_name if self.left_score > self.right_score else self.player_right_name
            print(f"Round {self.round_count} finished! Winner: {winner}")
            # log round result to file
            self._log_round_result(winner)
            # record in history (keep last 10)
            self.round_history.append((self.round_count, winner))
            if len(self.round_history) > 10:
                self.round_history.pop(0)
            # 🆕 show winner banner for a few seconds
            self.show_round_winner = f"🏆 {winner} Wins Round {self.round_count}!"
            self.winner_timer = 2.5  # seconds to display
            self.round_count += 1
            self.left_score = 0
            self.right_score = 0
            self.ball_speed = BASE_BALL_SPEED  # reset ball speed
            self.spin_factor = SPIN_FACTOR  # reset spin
            self._respawn(ball_to_left=random.choice([True, False]))

        # Encourage longer rallies & penalize idling
        r_l += 0.001 * abs(self.ball_vx)
        r_r += 0.001 * abs(self.ball_vx)
        r_l -= 0.0005 * abs(self.left_v)
        r_r -= 0.0005 * abs(self.right_v)

        obs = self._get_obs()
        info = {"score": (self.left_score, self.right_score)}
        return obs, (r_l, r_r), False, info

    def _update_velocity(self, vel, action):
        if action == 1:
            vel -= PADDLE_ACCEL
        elif action == 2:
            vel += PADDLE_ACCEL
        else:
            vel *= 0.85
        return float(np.clip(vel, -PADDLE_MAX_SPEED, PADDLE_MAX_SPEED))

    def _respawn(self, ball_to_left=True):
        # Respawn the ball in the center after a point or round.
        self.ball_x, self.ball_y = self.width / 2, self.height / 2
        angle = random.uniform(-0.35, 0.35)
        sign = -1 if ball_to_left else 1
        self.ball_vx = sign * self.ball_speed * math.cos(angle)
        self.ball_vy = self.ball_speed * math.sin(angle)

    def _log_round_result(self, winner):
        """Append round result to a log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Round {self.round_count} Winner: {winner} ({self.left_score}-{self.right_score})\n"
        with open("round_log.txt", "a", encoding="utf-8") as f:
            f.write(log_entry)
        print("Logged:", log_entry.strip())

    def update_learning_progress(self, avg_reward_left, avg_reward_right):
        """Compute learning improvement percentage for each player."""
        for name, avg in zip(["Player X", "Player Y"], [avg_reward_left, avg_reward_right]):
            data = self.learning_progress[name]["history"]
            data.append(avg)
            if len(data) > 1:
                prev_avg = data[-2]
                if abs(prev_avg) < 1e-6:
                    improvement = 0.0
                else:
                    improvement = ((avg - prev_avg) / abs(prev_avg)) * 100.0
                self.learning_progress[name]["improvement"] = improvement
            # keep last 20 samples only
            if len(data) > 20:
                data.pop(0)

    # input/render helpers
    def handle_events(self, human_control=False, human_action_ref=None):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); raise SystemExit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); raise SystemExit()
                elif ev.key == pygame.K_f:
                    self.fullscreen = not self.fullscreen; self._create_window()
                elif ev.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif ev.key == pygame.K_p:
                    return "toggle_pause"
                elif ev.key == pygame.K_o:
                    return "single_step"
        if human_control and human_action_ref is not None:
            k = pygame.key.get_pressed()
            if k[pygame.K_w]:
                human_action_ref[0] = 1
            elif k[pygame.K_s]:
                human_action_ref[0] = 2
            else:
                human_action_ref[0] = 0
        return None

    def render_frame(self, frame, rewards, actions, paused=False):
        if not self.render:
            return
        pygame.event.pump()
        # decrease winner timer each frame
        if self.winner_timer > 0:
            self.winner_timer -= 1.0 / FPS
            if self.winner_timer <= 0:
                self.show_round_winner = None
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), (30, int(self.left_y - self.ph / 2), self.pw, self.ph))
        pygame.draw.rect(self.screen, (255, 255, 255), (self.width - 30 - self.pw, int(self.right_y - self.ph / 2), self.pw, self.ph))
        pygame.draw.ellipse(self.screen, (255, 255, 255), (int(self.ball_x - self.ball_size / 2), int(self.ball_y - self.ball_size / 2), self.ball_size, self.ball_size))
        pygame.draw.aaline(self.screen, (80, 80, 80), (self.width / 2, 0), (self.width / 2, self.height))
        # Draw player names on top of the score
        left_name_surf = self.font.render(self.player_left_name, True, (120, 220, 255))
        right_name_surf = self.font.render(self.player_right_name, True, (255, 150, 150))
        vs_text = self.font.render("VS", True, (200, 200, 200))

        # Positioning for centered layout
        total_width = left_name_surf.get_width() + vs_text.get_width() + right_name_surf.get_width() + 20
        names_x = (self.width - total_width) // 2
        names_y = 8  # top padding

        self.screen.blit(left_name_surf, (names_x, names_y))
        self.screen.blit(vs_text, (names_x + left_name_surf.get_width() + 10, names_y))
        self.screen.blit(right_name_surf, (names_x + left_name_surf.get_width() + vs_text.get_width() + 20, names_y))

        # Display learning improvement percentages beside names
        imp_x = self.learning_progress["Player X"]["improvement"]
        imp_y = self.learning_progress["Player Y"]["improvement"]
        imp_color_x = (0, 255, 0) if imp_x > 0 else (255, 100, 100)
        imp_color_y = (0, 255, 0) if imp_y > 0 else (255, 100, 100)
        imp_text_x = self.font.render(f"{imp_x:+.1f}%", True, imp_color_x)
        imp_text_y = self.font.render(f"{imp_y:+.1f}%", True, imp_color_y)

        # Position improvement beside player names
        self.screen.blit(imp_text_x, (names_x - imp_text_x.get_width() - 10, names_y))
        self.screen.blit(imp_text_y, (names_x + left_name_surf.get_width() + vs_text.get_width() + right_name_surf.get_width() + 30, names_y))

        # Draw score and round just below names
        score_text = self.bigfont.render(f"{self.left_score} : {self.right_score}", True, (255, 255, 255))
        round_text = self.font.render(f"Round: {self.round_count}", True, (180, 220, 255))
        score_x = self.width // 2 - score_text.get_width() // 2
        round_x = self.width // 2 - round_text.get_width() // 2
        self.screen.blit(score_text, (score_x, names_y + 26))  # move score below names
        self.screen.blit(round_text, (round_x, names_y + 60))

        # Optional detailed info
        info_lines_count = 0
        if self.show_info:
            lines = [
                f"Frame: {frame}",
                f"Actions L:{actions[0]} R:{actions[1]}",
                f"Rewards L:{rewards[0]:+.3f} R:{rewards[1]:+.3f}",
                f"Ball: ({int(self.ball_x)},{int(self.ball_y)})",
                f"Ball vel: ({self.ball_vx:.2f},{self.ball_vy:.2f})",
                f"Paddles L:{int(self.left_y)} R:{int(self.right_y)}",
                f"Ball speed param: {self.ball_speed:.2f}",
                f"Paused: {paused}"
            ]
            for i, line in enumerate(lines):
                surf = self.font.render(line, True, (200, 200, 200))
                self.screen.blit(surf, (12, 70 + i * 18))
            info_lines_count = len(lines)

        # Round Winners panel (left side). Shows up to last 10 rounds.
        if self.round_history:
            base_y = 70 + (info_lines_count * 18) + 10
            title = self.font.render("Round Winners:", True, (180, 220, 255))
            self.screen.blit(title, (12, base_y))
            for j, (num, winner) in enumerate(self.round_history):
                txt = self.font.render(f"  Round {num:>2}: {winner}", True, (220, 220, 220))
                # Space each line a bit tighter so panel doesn't become too tall
                self.screen.blit(txt, (12, base_y + 18 + j * 16))

        # HUD Controls
        hud = ["[F] Fullscreen", "[I] Info", "[P] Pause", "[O] Step", "[ESC] Exit"]
        for i, t in enumerate(hud):
            surf = self.font.render(t, True, (120, 200, 255))
            self.screen.blit(surf, (self.width - 260, self.height - 120 + i * 18))

        pygame.display.flip()
        # draw large centered winner banner if active
        if self.show_round_winner:
            banner = self.bigfont.render(self.show_round_winner, True, (255, 215, 0))
            shadow = self.bigfont.render(self.show_round_winner, True, (0, 0, 0))
            bx = self.width // 2 - banner.get_width() // 2
            by = self.height // 2 - banner.get_height() // 2
            self.screen.blit(shadow, (bx + 2, by + 2))
            self.screen.blit(banner, (bx, by))
        self.clock.tick(FPS)


# Scripted expert (used for BC)
def scripted_expert_action(obs_flat, env: PongEnv, side: str):
    # side: 'left' or 'right' -> move paddle toward predicted intercept
    if side == 'left':
        pred = env._predict_ball_y_for_side(40)
        py = env.left_y
    else:
        pred = env._predict_ball_y_for_side(env.width - 40)
        py = env.right_y
    if abs(pred - py) < 8:
        return 0
    return 1 if pred < py else 2



# Networks (actor, centralized critic)
class ActorNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU()
        )
        self.actor = nn.Linear(128, n_actions)

    def forward(self, x):
        h = self.model(x)
        return self.actor(h)

class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim_total):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim_total, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)



# PPO Actor with local buffer (actor-only) + central critic used for values
class PPOActor:
    def __init__(self, obs_dim, n_actions, lr=LR_ACTOR):
        self.model = ActorNet(obs_dim, n_actions).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # local rollout buffer
        self.obs = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.dones = []
        self.values = []  # will store critic-provided value at that timestep

    def select_action(self, obs_np, critic: CentralizedCritic, other_stack):
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = self.model(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action).item()

        central_input = np.concatenate([obs_np, other_stack], axis=0)[None, :]
        with torch.no_grad():
            value = critic(torch.tensor(central_input, dtype=torch.float32, device=DEVICE)).cpu().item()
        return int(action.item()), logp, value

    def store(self, obs, action, logp, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.logps.append(logp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.obs = []; self.actions = []; self.logps = []; self.rewards = []; self.dones = []; self.values = []

    def compute_gae(self, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            lastgaelam = deltas[t] + gamma * lam * (1 - dones[t]) * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values[:-1]
        return adv, returns

    def update_policy(self, critic: CentralizedCritic):
        if len(self.obs) == 0:
            return
        # bootstrap last value 0 (we'll require external last_value) to compute GAE; but our training loop provides last_value instead
        # Outside caller will compute advs and returns then call this with prepared tensors; to keep things simple here we'll leave this method unused:
        pass



# Training helpers: updates (actors + centralized critic)
def ppo_update_actors_and_critic(left_actor: PPOActor, right_actor: PPOActor,
                                 central_critic: CentralizedCritic,
                                 actor_opts, critic_opt,
                                 clip=CLIP):
    """
    left_actor/right_actor: contain local rollout buffers
    central_critic: network to update (takes concat of stacked left+right obs)
    Compute advantages per-sample using the stored values coming from the critic at runtime.
    """
    # build dataset: for each stored timestep we have two samples (left, right) with same central_obs
    n_left = len(left_actor.obs)
    n_right = len(right_actor.obs)
    assert n_left == n_right, "Both agents should have same number of stored steps (we store one each per env step)."
    N = n_left  # number of time steps recorded

    # prepare arrays
    # create central_obs for each timestep by concatenating left_stack and right_stack (both were stored as obs)
    central_obs = np.array([np.concatenate([left_actor.obs[i], right_actor.obs[i]], axis=0) for i in range(N)], dtype=np.float32)

    # left arrays
    left_obs = np.array(left_actor.obs, dtype=np.float32)
    left_actions = np.array(left_actor.actions, dtype=np.int64)
    left_oldlogp = np.array(left_actor.logps, dtype=np.float32)
    left_rewards = np.array(left_actor.rewards, dtype=np.float32)
    left_dones = np.array(left_actor.dones, dtype=np.float32)
    left_values = np.array(left_actor.values, dtype=np.float32)

    # right arrays
    right_obs = np.array(right_actor.obs, dtype=np.float32)
    right_actions = np.array(right_actor.actions, dtype=np.int64)
    right_oldlogp = np.array(right_actor.logps, dtype=np.float32)
    right_rewards = np.array(right_actor.rewards, dtype=np.float32)
    right_dones = np.array(right_actor.dones, dtype=np.float32)
    right_values = np.array(right_actor.values, dtype=np.float32)

    # compute GAE for left & right using their stored values and last bootstrap values supplied earlier (compute last values before calling this)
    # The caller provides last_values via appending to actor.values before calling ppo_update_actors_and_critic.
    # For simplicity, we compute advantages using stored values + last value appended already.
    # Compute advs and returns for left:
    # Note: In train(), append last_value to actor.values before calling this function.
    def compute_advs(vals, rewards, dones):
        vals = np.array(vals, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        # values already includes last bootstrap appended (size = T+1)
        deltas = rewards + GAMMA * vals[1:] * (1 - dones) - vals[:-1]
        advs = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(rewards))):
            lastgaelam = deltas[t] + GAMMA * GAE_LAMBDA * (1 - dones[t]) * lastgaelam
            advs[t] = lastgaelam
        returns = advs + vals[:-1]
        return advs, returns

    # append bootstrap using central critic for the last central_obs (have appended last bootstrap earlier at caller)
    # But to keep code simple and robust: compute last bootstrap values now:
    with torch.no_grad():
        last_central_vals = central_critic(torch.tensor(central_obs[-1:], dtype=torch.float32, device=DEVICE)).cpu().numpy()
    # Build values arrays with appended last value
    left_vals_with_boot = np.concatenate([left_values, last_central_vals], axis=0)
    right_vals_with_boot = np.concatenate([right_values, last_central_vals], axis=0)

    left_advs, left_returns = compute_advs(left_vals_with_boot, left_rewards, left_dones)
    right_advs, right_returns = compute_advs(right_vals_with_boot, right_rewards, right_dones)

    # standardize advantages
    left_advs = (left_advs - left_advs.mean()) / (left_advs.std() + 1e-8)
    right_advs = (right_advs - right_advs.mean()) / (right_advs.std() + 1e-8)

    # Prepare tensors
    left_obs_t = torch.tensor(left_obs, dtype=torch.float32, device=DEVICE)
    left_act_t = torch.tensor(left_actions, device=DEVICE)
    left_oldlogp_t = torch.tensor(left_oldlogp, dtype=torch.float32, device=DEVICE)
    left_ret_t = torch.tensor(left_returns, dtype=torch.float32, device=DEVICE)
    left_adv_t = torch.tensor(left_advs, dtype=torch.float32, device=DEVICE)

    right_obs_t = torch.tensor(right_obs, dtype=torch.float32, device=DEVICE)
    right_act_t = torch.tensor(right_actions, device=DEVICE)
    right_oldlogp_t = torch.tensor(right_oldlogp, dtype=torch.float32, device=DEVICE)
    right_ret_t = torch.tensor(right_returns, dtype=torch.float32, device=DEVICE)
    right_adv_t = torch.tensor(right_advs, dtype=torch.float32, device=DEVICE)

    central_obs_t = torch.tensor(central_obs, dtype=torch.float32, device=DEVICE)

    # Update actors (independently) and critic jointly
    N_total = left_obs_t.size(0)
    batch_size = max(64, MINI_BATCH)

    for epoch in range(PPO_EPOCHS):
        idxs = np.random.permutation(N_total)
        for start in range(0, N_total, batch_size):
            mb = idxs[start:start + batch_size]

            # ----- LEFT actor update -----
            logits_l = left_actor_model_forward(left_actor, left_obs_t[mb])
            dist_l = Categorical(logits=logits_l)
            newlogp_l = dist_l.log_prob(left_act_t[mb])
            entropy_l = dist_l.entropy().mean()
            ratio_l = torch.exp(newlogp_l - left_oldlogp_t[mb])
            s1_l = ratio_l * left_adv_t[mb]
            s2_l = torch.clamp(ratio_l, 1 - CLIP, 1 + CLIP) * left_adv_t[mb]
            policy_loss_l = -torch.min(s1_l, s2_l).mean()

            # ----- RIGHT actor update -----
            logits_r = right_actor_model_forward(right_actor, right_obs_t[mb])
            dist_r = Categorical(logits=logits_r)
            newlogp_r = dist_r.log_prob(right_act_t[mb])
            entropy_r = dist_r.entropy().mean()
            ratio_r = torch.exp(newlogp_r - right_oldlogp_t[mb])
            s1_r = ratio_r * right_adv_t[mb]
            s2_r = torch.clamp(ratio_r, 1 - CLIP, 1 + CLIP) * right_adv_t[mb]
            policy_loss_r = -torch.min(s1_r, s2_r).mean()

            # ----- Centralized Critic update -----
            values_pred = central_critic(central_obs_t[mb])
            # Central critic predicts shared/team return (average of both)
            ret_mb = (left_ret_t[mb] + right_ret_t[mb]) / 2.0
            value_loss = F.mse_loss(values_pred, ret_mb)

            # ----- Left actor optimization -----
            actor_opts[0].zero_grad()
            (policy_loss_l - entropy_coef * entropy_l).backward()
            nn.utils.clip_grad_norm_(left_actor.model.parameters(), MAX_GRAD_NORM)
            actor_opts[0].step()

            # ----- Right actor optimization -----
            actor_opts[1].zero_grad()
            (policy_loss_r - entropy_coef * entropy_r).backward()
            nn.utils.clip_grad_norm_(right_actor.model.parameters(), MAX_GRAD_NORM)
            actor_opts[1].step()

            # ----- Critic optimization -----
            critic_opt.zero_grad()
            (VF_COEF * value_loss).backward()
            nn.utils.clip_grad_norm_(central_critic.parameters(), MAX_GRAD_NORM)
            critic_opt.step()

            # Decay entropy coefficient once per PPO update (after all mini-batches)
            entropy_coef = max(0.002, entropy_coef * entropy_decay)


# helper wrappers to directly call actor model forward (avoid attribute lookup in inner loop)
def left_actor_model_forward(left_actor: PPOActor, x):
    return left_actor.model(x)  # ActorNet returns logits via .model then actor head; in our ActorNet .forward returns logits

def right_actor_model_forward(right_actor: PPOActor, x):
    return right_actor.model(x)


# HIGH-LEVEL TRAIN LOOP with CTDE + Self-play + Frame stacking
def train(render=True, info=True, human=False, load_models=False, pretrain=True, watch=False):
    env = PongEnv(render=render, show_info=info)
    # Actors
    left_actor = PPOActor(OBS_DIM, NUM_ACTIONS)
    right_actor = PPOActor(OBS_DIM, NUM_ACTIONS)
    # Centralized critic
    central_critic = CentralizedCritic(OBS_DIM * 2).to(DEVICE)
    actor_opts = [
        torch.optim.Adam(left_actor.model.parameters(), lr=LR_ACTOR),
        torch.optim.Adam(right_actor.model.parameters(), lr=LR_ACTOR),
    ]
    critic_opt = torch.optim.Adam(central_critic.parameters(), lr=LR_CRITIC)

    # Self-play pool (list of checkpoint paths)
    selfplay_pool = []

    # Load if requested
    if load_models:
        if os.path.exists("ppo_left_final.pth"):
            left_actor.model.load_state_dict(torch.load("ppo_left_final.pth", map_location=DEVICE))
            print("Loaded ppo_left_final.pth")
        if os.path.exists("ppo_right_final.pth"):
            right_actor.model.load_state_dict(torch.load("ppo_right_final.pth", map_location=DEVICE))
            print("Loaded ppo_right_final.pth")

    # Behavior cloning pretrain (use scripted expert to collect many transitions)
    if pretrain:
        print("Running Behavior Cloning pretrain (this may take a while)...")
        bc_env = PongEnv(render=False)
        bc_actor = PPOActor(OBS_DIM, NUM_ACTIONS)
        # collect transitions quickly with scripted expert
        obs = bc_env.reset()
        # build frame stacks
        frames_l = deque([obs.copy() for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
        frames_r = deque([obs.copy() for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
        bc_obs = []
        bc_acts = []
        for i in range(BC_TRANSITIONS // 2):
            stacked_l = np.concatenate(list(frames_l), axis=0)
            stacked_r = np.concatenate(list(frames_r), axis=0)
            a_l = scripted_expert_action(stacked_l, bc_env, 'left')
            a_r = scripted_expert_action(stacked_r, bc_env, 'right')
            bc_obs.append(stacked_l)
            bc_acts.append(a_l)
            bc_obs.append(stacked_r)
            bc_acts.append(a_r)
            obs, _, _, _ = bc_env.step(a_l, a_r)
            frames_l.append(obs.copy()); frames_r.append(obs.copy())
        # train both actor networks supervised
        def bc_train_model(actor_model, obs_data, act_data, epochs=BC_EPOCHS, batch=256):
            actor_model.model.train()
            opt = torch.optim.Adam(actor_model.model.parameters(), lr=LR_ACTOR * 2)
            dataset = list(zip(obs_data, act_data))
            for e in range(epochs):
                random.shuffle(dataset)
                for j in range(0, len(dataset), batch):
                    batch_ = dataset[j:j + batch]
                    obs_b = torch.tensor(np.array([b[0] for b in batch_], dtype=np.float32), device=DEVICE)
                    acts_b = torch.tensor([b[1] for b in batch_], dtype=torch.long, device=DEVICE)
                    logits = actor_model.model(obs_b)

                    probs = F.log_softmax(logits, dim=-1)
                    loss = F.nll_loss(probs, acts_b)
                    loss += 0.001 * (logits ** 2).mean()  # mild regularization

                    opt.zero_grad(); loss.backward(); opt.step()
            actor_model.model.eval()
        # split dataset and train left & right actors
        split = len(bc_obs) // 2
        bc_train_model(left_actor, bc_obs[:split], bc_acts[:split])
        bc_train_model(right_actor, bc_obs[split:], bc_acts[split:])
        print("BC pretrain done.")

    # frame stacks for training env
    obs = env.reset()
    frames_l = deque([obs.copy() for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
    frames_r = deque([obs.copy() for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)

    # training loop variables
    frame = 0
    total_frames = 0
    updates = 0
    paused = False
    single_step = False
    human_action = [0]
    t0 = time.time()
    entropy_coef = 0.02     # initial entropy for exploration
    entropy_decay = 0.995   # anneal rate per PPO update

    print("Starting CTDE MARL training on", DEVICE)
    # main updates loop
    while updates < TOTAL_UPDATES:
        # clear actor buffers
        left_actor.clear()
        right_actor.clear()

        # collect STEPS_PER_UPDATE environment steps
        for step in range(STEPS_PER_UPDATE):
            evt = env.handle_events(human_control=human, human_action_ref=human_action)
            if evt == "toggle_pause":
                paused = not paused
            if evt == "single_step":
                single_step = True
            if paused and not single_step:
                if render:
                    env.render_frame(total_frames, (0, 0), (0, 0), paused=True)
                time.sleep(0.02)
                continue

            # build stacked obs for each agent
            stacked_l = np.concatenate(list(frames_l), axis=0)
            stacked_r = np.concatenate(list(frames_r), axis=0)

            # agents choose actions (decentralized)
            if human:
                a_l = human_action[0]
                lp_l = 0.0; v_l = 0.0
            else:
                a_l, lp_l, v_l = left_actor.select_action(stacked_l, central_critic, stacked_r)
            a_r, lp_r, v_r = right_actor.select_action(stacked_r, central_critic, stacked_l)

            # step environment
            next_obs, (r_l, r_r), done, info = env.step(a_l, a_r)

            # append next obs into frame stacks
            frames_l.append(next_obs.copy()); frames_r.append(next_obs.copy())

            # store transitions locally (value used was provided at selection time via critic)
            left_actor.store(stacked_l, a_l, lp_l, r_l, float(done), v_l)
            right_actor.store(stacked_r, a_r, lp_r, r_r, float(done), v_r)

            obs = next_obs
            frame += 1
            total_frames += 1

            if render:
                env.render_frame(total_frames, (r_l, r_r), (a_l, a_r), paused=paused)

            if total_frames % CURRICULUM_STEP_FRAMES == 0 and total_frames > 0:
                env.ball_speed = min(env.ball_speed + BALL_SPEED_GROW, MAX_BALL_SPEED)
                print(f"Curriculum bump -> ball_speed {env.ball_speed:.2f}")

            if total_frames % AUTOSAVE_EVERY_FRAMES == 0 and total_frames > 0:
                torch.save(left_actor.model.state_dict(), f"ppo_left_autosave_{total_frames}.pth")
                torch.save(right_actor.model.state_dict(), f"ppo_right_autosave_{total_frames}.pth")
                print("Autosaved models at frame", total_frames)

            if single_step:
                single_step = False
                paused = True

        # After collecting rollouts, compute last bootstrap values using central_critic for final central_obs
        # build central_obs array for last timestep
        final_left_stack = np.concatenate(list(frames_l), axis=0)
        final_right_stack = np.concatenate(list(frames_r), axis=0)
        central_final = np.concatenate([final_left_stack, final_right_stack], axis=0)[None, :]
        with torch.no_grad():
            last_value = central_critic(torch.tensor(central_final, dtype=torch.float32, device=DEVICE)).cpu().item()

        # Prepare data and compute advantages inside update function
        # Prepare arrays (convert lists to numpy arrays once)
        N = len(left_actor.obs)
        if N == 0:
            print("No samples collected — skipping update")
            continue

        # central_obs for all time steps
        central_obs = np.concatenate([np.concatenate([left_actor.obs[i], right_actor.obs[i]], axis=0)[None, :] for i in range(N)], axis=0).astype(np.float32)
        left_obs = np.array(left_actor.obs, dtype=np.float32)
        right_obs = np.array(right_actor.obs, dtype=np.float32)
        left_actions = np.array(left_actor.actions, dtype=np.int64)
        right_actions = np.array(right_actor.actions, dtype=np.int64)
        left_oldlogp = np.array(left_actor.logps, dtype=np.float32)
        right_oldlogp = np.array(right_actor.logps, dtype=np.float32)
        left_rewards = np.array(left_actor.rewards, dtype=np.float32)
        right_rewards = np.array(right_actor.rewards, dtype=np.float32)
        left_dones = np.array(left_actor.dones, dtype=np.float32)
        right_dones = np.array(right_actor.dones, dtype=np.float32)
        left_values = np.array(left_actor.values, dtype=np.float32)
        right_values = np.array(right_actor.values, dtype=np.float32)

        # compute bootstrapped values: last central value used for both agents
        left_vals_with_boot = np.concatenate([left_values, np.array([last_value], dtype=np.float32)], axis=0)
        right_vals_with_boot = np.concatenate([right_values, np.array([last_value], dtype=np.float32)], axis=0)

        # compute advantages and returns (vectorized)
        def compute_adv_ret(vals_with_boot, rewards, dones):
            T = len(rewards)
            vals = vals_with_boot
            deltas = rewards + GAMMA * vals[1:] * (1 - dones) - vals[:-1]
            adv = np.zeros_like(rewards, dtype=np.float32)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                lastgaelam = deltas[t] + GAMMA * GAE_LAMBDA * (1 - dones[t]) * lastgaelam
                adv[t] = lastgaelam
            returns = adv + vals[:-1]
            return adv, returns

        left_advs, left_returns = compute_adv_ret(left_vals_with_boot, left_rewards, left_dones)
        right_advs, right_returns = compute_adv_ret(right_vals_with_boot, right_rewards, right_dones)

        left_advs = (left_advs - left_advs.mean()) / (left_advs.std() + 1e-8)
        right_advs = (right_advs - right_advs.mean()) / (right_advs.std() + 1e-8)

        # convert to tensors
        left_obs_t = torch.tensor(left_obs, dtype=torch.float32, device=DEVICE)
        right_obs_t = torch.tensor(right_obs, dtype=torch.float32, device=DEVICE)
        left_act_t = torch.tensor(left_actions, device=DEVICE)
        right_act_t = torch.tensor(right_actions, device=DEVICE)
        left_oldlogp_t = torch.tensor(left_oldlogp, dtype=torch.float32, device=DEVICE)
        right_oldlogp_t = torch.tensor(right_oldlogp, dtype=torch.float32, device=DEVICE)
        left_ret_t = torch.tensor(left_returns, dtype=torch.float32, device=DEVICE)
        right_ret_t = torch.tensor(right_returns, dtype=torch.float32, device=DEVICE)
        left_adv_t = torch.tensor(left_advs, dtype=torch.float32, device=DEVICE)
        right_adv_t = torch.tensor(right_advs, dtype=torch.float32, device=DEVICE)
        central_obs_t = torch.tensor(central_obs, dtype=torch.float32, device=DEVICE)

        N_total = left_obs_t.size(0)
        batch_size = max(64, MINI_BATCH)

        # update loop
        for epoch in range(PPO_EPOCHS):
            perm = np.random.permutation(N_total)
            for start_ in range(0, N_total, batch_size):
                mb = perm[start_:start_ + batch_size]
                mb_left_obs = left_obs_t[mb]
                mb_right_obs = right_obs_t[mb]
                mb_left_act = left_act_t[mb]
                mb_right_act = right_act_t[mb]
                mb_left_oldlogp = left_oldlogp_t[mb]
                mb_right_oldlogp = right_oldlogp_t[mb]
                mb_left_adv = left_adv_t[mb]
                mb_right_adv = right_adv_t[mb]
                mb_left_ret = left_ret_t[mb]
                mb_right_ret = right_ret_t[mb]
                mb_central = central_obs_t[mb]

                # actor forward
                logits_l = left_actor.model(mb_left_obs)
                dist_l = Categorical(logits=logits_l)
                newlogp_l = dist_l.log_prob(mb_left_act)
                entropy_l = dist_l.entropy().mean()
                ratio_l = torch.exp(newlogp_l - mb_left_oldlogp)
                s1_l = ratio_l * mb_left_adv
                s2_l = torch.clamp(ratio_l, 1 - CLIP, 1 + CLIP) * mb_left_adv
                policy_loss_l = -torch.min(s1_l, s2_l).mean()

                logits_r = right_actor.model(mb_right_obs)
                dist_r = Categorical(logits=logits_r)
                newlogp_r = dist_r.log_prob(mb_right_act)
                entropy_r = dist_r.entropy().mean()
                ratio_r = torch.exp(newlogp_r - mb_right_oldlogp)
                s1_r = ratio_r * mb_right_adv
                s2_r = torch.clamp(ratio_r, 1 - CLIP, 1 + CLIP) * mb_right_adv
                policy_loss_r = -torch.min(s1_r, s2_r).mean()

                # critic forward
                values_pred = central_critic(mb_central)
                # target is average of both returns
                ret_target = (mb_left_ret + mb_right_ret) * 0.5
                value_loss = F.mse_loss(values_pred, ret_target)

                # actor updates
                actor_opts[0].zero_grad()
                (policy_loss_l - entropy_coef * entropy_l).backward()
                nn.utils.clip_grad_norm_(left_actor.model.parameters(), MAX_GRAD_NORM)
                actor_opts[0].step()

                actor_opts[1].zero_grad()
                entropy_coef = max(0.002, entropy_coef * entropy_decay)
                nn.utils.clip_grad_norm_(right_actor.model.parameters(), MAX_GRAD_NORM)
                actor_opts[1].step()

                # critic update
                critic_opt.zero_grad()
                (VF_COEF * value_loss).backward()
                nn.utils.clip_grad_norm_(central_critic.parameters(), MAX_GRAD_NORM)
                critic_opt.step()

        updates += 1
        entropy_coef = max(0.002, entropy_coef * entropy_decay)

        if env.left_score >= 3 or env.right_score >= 3:
            env.ball_speed = min(env.ball_speed + 0.4, MAX_BALL_SPEED)
            env.spin_factor = min(env.spin_factor + 0.02, 0.35)
            env.left_score = env.right_score = 0
            print(f"🏓 Curriculum boost → speed {env.ball_speed:.2f}, spin {env.spin_factor:.2f}")

        # self-play: snapshot current left policy into pool every few updates
        if updates % SELFPLAY_SAVE_EVERY_UPDATES == 0:
            ckpt_path = f"selfplay_left_upd{updates}.pth"
            torch.save(left_actor.model.state_dict(), ckpt_path)
            selfplay_pool.append(ckpt_path)
            # keep pool bounded
            if len(selfplay_pool) > SELFPLAY_POOL_MAX:
                old = selfplay_pool.pop(0)
                try:
                    os.remove(old)
                except Exception:
                    pass
            print("Added self-play checkpoint:", ckpt_path)

        # occasional opponent swap: load a random checkpoint from pool into right actor at start of next collection
        if selfplay_pool and random.random() < 0.15:
            ckpt = random.choice(selfplay_pool)
            try:
                sd = torch.load(ckpt, map_location=DEVICE)
                right_actor.model.load_state_dict(sd)
                print("Self-play: loaded opponent from", ckpt)
            except Exception as e:
                print("Failed to load self-play ckpt", ckpt, e)

        # logging
        elapsed = time.time() - t0
        fps = int(total_frames / elapsed) if elapsed > 0 else 0
        # 🧮 Update learning progress based on mean rewards per step
        avg_r_l = np.mean(left_actor.rewards) if left_actor.rewards else 0.0
        avg_r_r = np.mean(right_actor.rewards) if right_actor.rewards else 0.0
        env.update_learning_progress(avg_r_l, avg_r_r)
        print(f"Round {env.round_count} | Update {updates}/{TOTAL_UPDATES} | frames {total_frames} | FPS {fps} | scores L:{env.left_score} R:{env.right_score}")

    # final save
    torch.save(left_actor.model.state_dict(), "ppo_left_final.pth")
    torch.save(right_actor.model.state_dict(), "ppo_right_final.pth")
    torch.save(central_critic.state_dict(), "ppo_central_critic_final.pth")
    print("Training finished; models saved.")



# WATCH mode
def watch(render=True, episodes=5, load_models=True, human=False):
    env = PongEnv(render=render, show_info=True)
    left = PPOActor(OBS_DIM, NUM_ACTIONS)
    right = PPOActor(OBS_DIM, NUM_ACTIONS)
    critic = CentralizedCritic(OBS_DIM * 2).to(DEVICE)
    if load_models:
        if os.path.exists("ppo_left_final.pth"):
            left.model.load_state_dict(torch.load("ppo_left_final.pth", map_location=DEVICE))
            print("Loaded left model.")
        if os.path.exists("ppo_right_final.pth"):
            right.model.load_state_dict(torch.load("ppo_right_final.pth", map_location=DEVICE))
            print("Loaded right model.")
        if os.path.exists("ppo_central_critic_final.pth"):
            critic.load_state_dict(torch.load("ppo_central_critic_final.pth", map_location=DEVICE))
            print("Loaded central critic.")
    human_action = [0]
    for ep in range(episodes):
        obs = env.reset()
        frames_l = deque([obs.copy() for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
        frames_r = deque([obs.copy() for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
        frame = 0
        while True:
            evt = env.handle_events(human_control=human, human_action_ref=human_action)
            if human:
                a_l = human_action[0]
            else:
                s_l = np.concatenate(list(frames_l), axis=0)
                s_r = np.concatenate(list(frames_r), axis=0)
                a_l, _, _ = left.select_action(s_l, critic, s_r)
            s_r = np.concatenate(list(frames_r), axis=0)
            a_r, _, _ = right.select_action(s_r, critic, np.concatenate(list(frames_l), axis=0))
            obs, (r_l, r_r), _, info = env.step(a_l, a_r)
            frames_l.append(obs.copy()); frames_r.append(obs.copy())
            frame += 1
            env.render_frame(frame, (r_l, r_r), (a_l, a_r))
            # reset on score to keep watching
            if info["score"][0] + info["score"][1] > 0:
                pygame.time.wait(300)
                obs = env.reset()
                break



# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--info", type=int, default=1)
    parser.add_argument("--human", type=int, default=0)
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--watch", type=int, default=0)
    args = parser.parse_args()

    if args.watch:
        watch(render=bool(args.render), load_models=bool(args.load), human=bool(args.human))
    else:
        train(render=bool(args.render), info=bool(args.info),
              human=bool(args.human), load_models=bool(args.load), pretrain=bool(args.pretrain))
