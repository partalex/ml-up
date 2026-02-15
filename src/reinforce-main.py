"""
REINFORCE algorithm for grid world environment.

REINFORCE is a policy gradient method that directly learns policy π_θ(a|s).
Updates parameters θ using Monte Carlo gradient estimation.

Implements:
- Stochastic policy with softmax distribution
- Monte Carlo estimation of returns
- Policy gradient update
- Tracking learning progress through 10 test episodes
- Visualization of learned policy and progress
"""

import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from simulator import Simulator, Action


class ReinforceAgent:
    """REINFORCE agent with stochastic policy."""

    def __init__(
            self,
            gamma: float = 0.9,
            learning_rate_type: str = "variable",  # "variable" or "constant"
            constant_alpha: float = 0.01,
    ) -> None:
        """
        Initializes REINFORCE agent.

        Args:
            gamma: Discount factor for future rewards
            learning_rate_type: Learning rate type ("variable" or "constant")
            constant_alpha: Constant learning rate
        """
        self.gamma = gamma
        self.learning_rate_type = learning_rate_type
        self.constant_alpha = constant_alpha

        # Policy parameters: θ(s, a) for each state-action combination
        # These are action preferences - higher values = higher probability
        self.theta: defaultdict[tuple[int, Action], float] = defaultdict(float)

        # History for tracking
        self.policy_params_history: list[dict[tuple[int, Action], float]] = []
        self.test_rewards_history: list[float] = []

    def get_learning_rate(self, episode: int) -> float:
        """
        Returns learning rate for given episode.

        Args:
            episode: Episode number (starts from 0)

        Returns:
            Learning rate
        """
        if self.learning_rate_type == "variable":
            # α_e = ln(e+1)/(e+1)
            e = episode + 1
            return np.log(e + 1) / (e + 1)

        return self.constant_alpha

    def get_action_probabilities(self, state: int) -> dict[Action, float]:
        """
        Calculates action probabilities using softmax over θ(s, a).

        Args:
            state: Current state

        Returns:
            Dictionary with probabilities for each action
        """
        actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        # Calculate preferences (θ values) for all actions
        preferences = np.array([self.theta[(state, a)] for a in actions])

        # Softmax for numerical stability
        # exp(θ - max(θ)) / sum(exp(θ - max(θ)))
        max_pref = np.max(preferences)
        exp_prefs = np.exp(preferences - max_pref)
        probs = exp_prefs / np.sum(exp_prefs)

        return {a: p for a, p in zip(actions, probs)}

    def select_action(self, state: int, explore: bool = True) -> Action:
        """
        Chooses action according to learned policy.
        Args:
            state: Current state
            explore: If True, samples by probabilities; if False, chooses best
        Returns:
            Chosen action
        """
        prob_dict = self.get_action_probabilities(state)
        actions = list(prob_dict.keys())
        probs = list(prob_dict.values())

        if explore:
            # Sample action according to probabilities
            # np.random.choice returns numpy type, convert back to Action
            idx = np.random.choice(len(actions), p=probs)
            return actions[idx]
        else:
            # Choose action with highest probability (greedy)
            return max(prob_dict.items(), key=lambda x: x[1])[0]

    def update_policy(
            self,
            episode_trajectory: list[tuple[int, Action, float]],
            alpha: float
    ) -> None:
        """
        Updates policy parameters using REINFORCE algorithm.
        REINFORCE update:
        θ(s,a) ← θ(s,a) + α·G_t·∇log(π(a|s))
        Where:
        - G_t = sum of discounted rewards from step t
        - ∇log(π(a|s)) = I(a=a_t) - π(a|s) (for softmax policy)
        Args:
            episode_trajectory: List of (state, action, reward) for entire episode
            alpha: Learning rate
        """
        trajectory_length = len(episode_trajectory)

        # Calculate returns (G_t) for each step
        returns: list[float] = []
        accumulative_return = 0.0
        for t in range(trajectory_length - 1, -1, -1):
            _, _, reward = episode_trajectory[t]
            accumulative_return = reward + self.gamma * accumulative_return
            returns.insert(0, accumulative_return)

        # Update parameters for each step
        for t in range(trajectory_length):
            state, action_taken, _ = episode_trajectory[t]
            G_t = returns[t]

            # Get current probabilities
            action_probs = self.get_action_probabilities(state)

            # Update θ for all actions
            for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
                # Log probability gradient: I(a=a_t) - π(a|s)
                if action == action_taken:
                    grad_log_pi = 1.0 - action_probs[action]
                else:
                    grad_log_pi = -action_probs[action]

                # REINFORCE update
                self.theta[(state, action)] += alpha * G_t * grad_log_pi

    def record_policy_params(self, states: list[int]) -> None:
        """
        Records current policy parameters.
        Args:
            states: List of all states
        """
        params = {}
        for state in states:
            for action in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
                params[(state, action)] = self.theta[(state, action)]
        self.policy_params_history.append(params.copy())


def run_test_episodes(
        agent: ReinforceAgent,
        simulator: Simulator,
        num_episodes: int = 10
) -> tuple[float, list[float]]:
    """
    Runs test episodes with current policy (without learning).
    Args:
        agent: REINFORCE agent
        simulator: Environment simulator
        num_episodes: Number of test episodes
    Returns:
        Tuple (average reward, list of rewards)
    """
    rewards = []

    for _ in range(num_episodes):
        state = simulator.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(100):  # Max 100 steps
            action = agent.select_action(state, explore=True)  # Use stochastic policy
            reward, next_state, done = simulator.step(action)
            total_reward += reward

            # If we entered a terminal state (done=True, reward=0),
            # we must execute one more action to get the reward
            if done and reward == 0.0 and next_state in simulator.TERMINAL_STATES:
                terminal_action = agent.select_action(next_state, explore=True)
                terminal_reward, final_state, _ = simulator.step(terminal_action)
                total_reward += terminal_reward
                break

            state = next_state
            steps += 1

            if done:
                break

        rewards.append(total_reward)

    return float(np.mean(rewards)), rewards


def train_reinforce(
        agent: ReinforceAgent,
        simulator: Simulator,
        num_episodes: int = 2000,
        test_interval: int = 100,
        max_steps_per_episode: int = 100,
) -> dict[str, list]:
    """
    Trains REINFORCE agent.
    Args:
        agent: REINFORCE agent
        simulator: Environment simulator
        num_episodes: Number of episodes for training
        test_interval: How often to test (every N episodes)
        max_steps_per_episode: Maximum number of steps per episode
    Returns:
        Dictionary with training statistics (rewards per episode, test rewards, etc.)
    """
    all_states = list(simulator.STATE_TO_COORD.keys())
    # Filter only non-terminal states
    non_terminal_states = [s for s in all_states if s not in simulator.TERMINAL_STATES]

    episode_rewards: list[float] = []
    test_episodes_list: list[int] = []
    test_rewards_list: list[float] = []
    terminal_hits: int = 0  # Counter for how many times we reached terminal state

    print(f"\n{'=' * 70}")
    print(f"Training REINFORCE agent")
    print(f"{'=' * 70}")
    print(f"Parameters:")
    print(f"  γ (gamma): {agent.gamma}")
    print(f"  Learning rate type: {agent.learning_rate_type}")
    if agent.learning_rate_type == "constant":
        print(f"  Constant α: {agent.constant_alpha}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Test interval: every {test_interval} episodes")
    print(f"{'=' * 70}\n")

    for episode in range(num_episodes):
        state = simulator.reset()
        alpha = agent.get_learning_rate(episode)

        # Collect entire episode
        trajectory: list[tuple[int, Action, float]] = []
        reached_terminal = False

        # Debug for first 3 episodes
        debug = episode < 3

        if debug:
            print(f"\n[DEBUG Episode {episode + 1}] Initial state: {simulator.get_state_name(state)}")

        for step in range(max_steps_per_episode):
            # Choose action according to current policy
            action = agent.select_action(state, explore=True)
            reward, next_state, done = simulator.step(action)

            if debug:
                print(f"  Step {step + 1}: {simulator.get_state_name(state)} -[{action.name}]-> "
                      f"{simulator.get_state_name(next_state)}, r={reward:.1f}, done={done}")

            trajectory.append((state, action, reward))

            # If we entered a terminal state (done=True, reward=0),
            # we must execute one more action to get the reward
            if done and reward == 0.0 and next_state in simulator.TERMINAL_STATES:
                if debug:
                    print(
                        f"    → Entered terminal {simulator.get_state_name(next_state)}, executing additional action...")
                # Now we're in terminal state, execute one more action to get reward
                terminal_action = agent.select_action(next_state, explore=True)
                terminal_reward, final_state, _ = simulator.step(terminal_action)
                if debug:
                    print(f"    → {simulator.get_state_name(next_state)} -[{terminal_action.name}]-> "
                          f"{simulator.get_state_name(final_state)}, r={terminal_reward:.1f}")
                trajectory.append((next_state, terminal_action, terminal_reward))
                reached_terminal = True
                break

            state = next_state

            if done:
                reached_terminal = True
                break

        if reached_terminal:
            terminal_hits += 1

        # Update policy using entire episode
        agent.update_policy(trajectory, alpha)

        # Calculate total episode reward
        total_reward = sum(r for _, _, r in trajectory)
        episode_rewards.append(total_reward)

        if debug:
            print(f"  Total episode reward: {total_reward:.1f}")
            print(f"  Trajectory length: {len(trajectory)}")

        # Testing every test_interval episodes
        if (episode + 1) % test_interval == 0:
            avg_test_reward, test_rewards = run_test_episodes(agent, simulator, num_episodes=10)
            test_episodes_list.append(episode + 1)
            test_rewards_list.append(avg_test_reward)
            agent.test_rewards_history.append(avg_test_reward)

            # Record policy parameters
            agent.record_policy_params(non_terminal_states)

            # Debug info - show some θ values
            sample_state = 0  # A1
            probs = agent.get_action_probabilities(sample_state)

            terminal_hit_rate = terminal_hits / (episode + 1)

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Average reward in training (last 100): "
                  f"{np.mean(episode_rewards[-100:]):.3f}")
            print(f"  Average reward in test (10 episodes): {avg_test_reward:.3f}")
            print(f"  Test rewards: {[f'{r:.1f}' for r in test_rewards[:5]]}...")
            print(f"  Terminal states reached: {terminal_hits}/{episode + 1} ({terminal_hit_rate:.1%})")
            print(f"  π(a|A1): UP={probs[Action.UP]:.3f}, DOWN={probs[Action.DOWN]:.3f}, "
                  f"LEFT={probs[Action.LEFT]:.3f}, RIGHT={probs[Action.RIGHT]:.3f}")
            print(f"  α: {alpha:.4f}\n")

    return {
        "episode_rewards": episode_rewards,
        "test_episodes": test_episodes_list,
        "test_rewards": test_rewards_list,
    }


def plot_results(
        agent: ReinforceAgent,
        stats: dict,
        simulator: Simulator,
        title_suffix: str = "",
) -> None:
    """
    Displays training results.
    Args:
        agent: Trained agent
        stats: Training statistics
        simulator: Simulator
        title_suffix: Title suffix
    """
    non_terminal_states = [s for s in simulator.STATE_TO_COORD.keys()
                           if s not in simulator.TERMINAL_STATES]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    fig.suptitle(f"REINFORCE Results {title_suffix}", fontsize=16, fontweight='bold')

    # 1. Rewards during training
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(stats["episode_rewards"], alpha=0.3, label="Reward per episode", color='blue')

    window = 50
    if len(stats["episode_rewards"]) >= window:
        moving_avg = np.convolve(
            stats["episode_rewards"],
            np.ones(window) / window,
            mode='valid'
        )
        ax1.plot(range(window - 1, len(stats["episode_rewards"])),
                 moving_avg,
                 color='red',
                 linewidth=2,
                 label=f'Moving average ({window} episodes)')

    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Total reward", fontsize=11)
    ax1.set_title("Rewards during training", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Test rewards (average in 10 episodes)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(stats["test_episodes"], stats["test_rewards"],
             marker='o', linewidth=2, markersize=6, color='green')
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Average reward", fontsize=11)
    ax2.set_title("Average reward in 10 test episodes", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Policy parameters during learning (non-terminal states)
    ax3 = fig.add_subplot(gs[1, 1])

    # Show θ(s, a) for some key states
    key_states = non_terminal_states[:min(3, len(non_terminal_states))]  # First 3 non-terminal

    for state in key_states:
        state_name = simulator.get_state_name(state)
        # Average θ value across all actions for given state
        avg_theta_values = []
        for params in agent.policy_params_history:
            actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
            avg_theta = np.mean([params.get((state, a), 0.0) for a in actions])
            avg_theta_values.append(avg_theta)

        ax3.plot(range(len(avg_theta_values)), avg_theta_values,
                 marker='o', markersize=4, label=state_name, linewidth=2)

    ax3.set_xlabel("Test iteration", fontsize=11)
    ax3.set_ylabel("Average θ value", fontsize=11)
    ax3.set_title("Policy parameters (average θ per state)", fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Learned policy (grid visualization)
    ax4 = fig.add_subplot(gs[2, :])

    arrow_map = {
        Action.UP: '↑',
        Action.DOWN: '↓',
        Action.LEFT: '←',
        Action.RIGHT: '→'
    }

    for state, (row, col) in simulator.STATE_TO_COORD.items():
        color = 'lightcoral' if state in simulator.TERMINAL_STATES else 'lightblue'

        rect = plt.Rectangle((col - 0.4, 1 - row - 0.4), 0.8, 0.8,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax4.add_patch(rect)

        state_name = simulator.get_state_name(state)
        ax4.text(col, 1 - row + 0.25, state_name, ha='center', va='center', fontsize=11, fontweight='bold')

        if state not in simulator.TERMINAL_STATES:
            # Show action with highest probability
            probs = agent.get_action_probabilities(state)
            best_action = max(probs.items(), key=lambda x: x[1])[0]
            best_prob = probs[best_action]

            arrow = arrow_map[best_action]
            ax4.text(col, 1 - row - 0.05, arrow,
                     ha='center', va='center', fontsize=20, color='darkred', fontweight='bold')

            # Show probability
            ax4.text(col, 1 - row - 0.28, f"p={best_prob:.2f}",
                     ha='center', va='center', fontsize=8, color='black')
        else:
            # Terminal state - show reward
            reward = simulator.TERMINAL_STATES[state]
            ax4.text(col, 1 - row - 0.1, f"R={reward:+.0f}",
                     ha='center', va='center', fontsize=10, color='darkgreen', fontweight='bold')

    ax4.set_xlim(-0.5, simulator.COLS - 0.5)
    ax4.set_ylim(-0.5, simulator.ROWS - 0.5)
    ax4.set_aspect('equal')
    ax4.set_title("Learned policy (best actions and probabilities)",
                  fontsize=12, fontweight='bold')
    ax4.axis('off')

    plt.tight_layout()
    filename = f"../out/02-reinforce/{title_suffix.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Graph saved as: {filename}")
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("REINFORCE ALGORITHM FOR GRID WORLD")
    print("=" * 70)

    # ========================================================================
    # EXPERIMENT 1: Variable learning rate, γ = 0.9
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Variable learning rate, γ = 0.9")
    print("=" * 70)

    simulator1 = Simulator()
    agent1 = ReinforceAgent(
        gamma=0.9,
        learning_rate_type="variable"
    )

    stats1 = train_reinforce(agent1, simulator1, num_episodes=2000, test_interval=100)

    print("\n" + "=" * 70)
    print("FINAL TESTING (10 episodes)")
    print("=" * 70)
    avg_reward1, rewards1 = run_test_episodes(agent1, simulator1, num_episodes=10)
    print(f"Average reward: {avg_reward1:.3f}")
    print(f"All rewards: {rewards1}")

    plot_results(agent1, stats1, simulator1, title_suffix="(γ=0.9, variable α)")

    # ========================================================================
    # EXPERIMENT 2: Constant learning rate, γ = 0.9
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Constant learning rate, γ = 0.9")
    print("=" * 70)

    simulator2 = Simulator()
    agent2 = ReinforceAgent(
        gamma=0.9,
        learning_rate_type="constant",
        constant_alpha=0.01  # Smaller α for REINFORCE (policy gradient is more sensitive)
    )

    stats2 = train_reinforce(agent2, simulator2, num_episodes=2000, test_interval=100)

    print("\n" + "=" * 70)
    print("FINAL TESTING (10 episodes)")
    print("=" * 70)
    avg_reward2, rewards2 = run_test_episodes(agent2, simulator2, num_episodes=10)
    print(f"Average reward: {avg_reward2:.3f}")
    print(f"All rewards: {rewards2}")

    plot_results(agent2, stats2, simulator2, title_suffix="(γ=0.9, constant α=0.01-q-learning)")

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n1. Variable α, γ=0.9:")
    print(f"   Average reward: {avg_reward1:.3f}")

    print(f"\n2. Constant α=0.01, γ=0.9:")
    print(f"   Average reward: {avg_reward2:.3f}")

    print(f"\n{'=' * 70}")
    print("ANALYSIS:")
    print("=" * 70)

    print("\n📊 REINFORCE algorithm:")
    print("   • Learns stochastic policy directly")
    print("   • Uses Monte Carlo estimate of returns")
    print("   • Updates parameters θ(s,a) using policy gradient")
    print("   • Softmax policy: π(a|s) = exp(θ(s,a)) / Σ exp(θ(s,a'))")

    print("\n📊 Impact of learning rate:")
    if abs(avg_reward1 - avg_reward2) < 0.1:
        print("   • Both strategies give similar results")
    elif avg_reward1 > avg_reward2:
        print("   • Variable α gives better results")
    else:
        print("   • Constant α gives better results")

    print("\n" + "=" * 70)
    print("Experiments finished!")
    print("=" * 70 + "\n")
