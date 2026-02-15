"""
Q-learning implementation for grid world environment.

Implements:
- Q-learning with ϵ-greedy exploration
- Variable learning rate: α_e = ln(e+1)/(e+1)
- Constant learning rate (for comparison)
- Tracking V-values during learning
- Testing learned policy
- Experiments with different γ values (0.9 and 0.999)
"""

import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from simulator import Simulator, Action, LearningRateType, arrow_map


class QLearningAgent:
    """Q-learning agent with ϵ-greedy exploration."""

    def __init__(
            self,
            gamma: float = 0.9,
            epsilon: float = 0.1,
            learning_rate_type: LearningRateType = LearningRateType.VARIABLE,
            constant_alpha: float = 0.1,
    ) -> None:
        """
        Initializes Q-learning agent.
        Args:
            gamma: Discount factor for future rewards
            epsilon: Probability for exploration (ϵ-greedy)
            learning_rate_type: Type of learning rate ("variable" or "constant")
            constant_alpha: Constant learning rate (if learning_rate_type="constant")
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate_type = learning_rate_type
        self.constant_alpha = constant_alpha

        # Q-table: (state, action) -> Q-value
        self.q_table: defaultdict[tuple[int, Action], float] = defaultdict(float)

        # History of V-values for each state during learning
        self.v_history: list[dict[int, float]] = []

    def get_learning_rate(self, episode: int) -> float:
        """
        Returns learning rate for given episode.
        Args:
            episode: Episode number (starts from 0)
        Returns:
            Learning rate
        """
        if self.learning_rate_type == LearningRateType.VARIABLE:
            # α_e = ln(e+1)/(e+1), where e is episode number (we start from 1)
            e = episode + 1
            return np.log(e + 1) / (e + 1)
        return self.constant_alpha

    def get_action(self, state: int, explore: bool = True) -> Action:
        """
        Chooses action using ϵ-greedy exploration.
        Args:
            state: Current state
            explore: Whether to use exploration (False for testing)
        Returns:
            Chosen action
        """
        if explore and random.random() < self.epsilon:
            # Exploration: choose random action
            return random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])
        # Exploitation: choose best action
        return self.get_best_action(state)

    def get_best_action(self, state: int) -> Action:
        """
        Returns best action for given state according to Q-table.
        Args:
            state: State
        Returns:
            Best action
        """
        actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        q_values = [self.q_table[(state, action)] for action in actions]
        max_q = max(q_values)

        # If there are multiple actions with same max value, choose randomly
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def get_v_value(self, state: int) -> float:
        """
        Returns V-value of state: V(s) = max_a Q(s, a).
        Args:
            state: State
        Returns:
            V-value of the state
        """
        actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        q_values = [self.q_table[(state, action)] for action in actions]
        return max(q_values) if q_values else 0.0

    def update_q_value(
            self,
            state: int,
            action: Action,
            reward: float,
            next_state: int,
            done: bool,
            alpha: float,
    ) -> None:
        """
        Updates Q-value using Q-learning algorithm.
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Whether episode is finished
            alpha: Learning rate
        """
        # Current Q-value
        current_q = self.q_table[(state, action)]

        # Maximum Q-value of next state
        if done:
            max_next_q = 0.0  # Terminal state
        else:
            actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
            max_next_q = max(self.q_table[(next_state, a)] for a in actions)

        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
        new_q = current_q + alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def record_v_values(self, states: list[int]) -> None:
        """
        Records V-values for all states.
        Args:
            states: List of all states
        """
        v_values = {state: self.get_v_value(state) for state in states}
        self.v_history.append(v_values)


def train_q_learning(
        agent: QLearningAgent,
        simulator: Simulator,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 100,
) -> dict[str, list[float]]:
    """
    Trains Q-learning agent.
    Args:
        agent: Q-learning agent
        simulator: Environment simulator
        num_episodes: Number of episodes for training
        max_steps_per_episode: Maximum number of steps per episode
    Returns:
        Dictionary with training statistics (e.g., rewards per episode)
    """
    all_states = list(simulator.STATE_TO_COORD.keys())
    episode_rewards: list[float] = []

    print(f"\n{'=' * 70}")
    print(f"Training Q-learning agent")
    print(f"{'=' * 70}")
    print(f"Parameters:")
    print(f"  γ (gamma): {agent.gamma}")
    print(f"  ϵ (epsilon): {agent.epsilon}")
    print(f"  Learning rate type: {agent.learning_rate_type}")
    if agent.learning_rate_type == "constant":
        print(f"  Constant α: {agent.constant_alpha}")
    print(f"  Number of episodes: {num_episodes}")
    print(f"{'=' * 70}\n")

    for episode in range(num_episodes):
        state = simulator.reset()
        total_reward = 0.0
        alpha = agent.get_learning_rate(episode)

        for step in range(max_steps_per_episode):
            # Choose and execute action
            action = agent.get_action(state, explore=True)
            reward, next_state, done = simulator.step(action)

            # Update Q-table
            agent.update_q_value(state, action, reward, next_state, done, alpha)

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        # Record V-values every 10 episodes
        if episode % 10 == 0:
            agent.record_v_values(all_states)

        # Show progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Average reward (last 100): {avg_reward:.3f}, "
                  f"α: {alpha:.4f}")

    # Record final V-values
    agent.record_v_values(all_states)

    return {"episode_rewards": episode_rewards}


def test_policy(
        agent: QLearningAgent,
        simulator: Simulator,
        num_test_episodes: int = 10,
        max_steps: int = 100,
) -> tuple[float, list[float]]:
    """
    Tests learned policy.
    Args:
        agent: Trained Q-learning agent
        simulator: Environment simulator
        num_test_episodes: Number of test episodes
        max_steps: Maximum number of steps per episode
    Returns:
        Tuple with average reward and list of all rewards
    """
    print(f"\n{'=' * 70}")
    print(f"Testing learned policy")
    print(f"{'=' * 70}\n")

    episode_rewards: list[float] = []

    for episode in range(num_test_episodes):
        state = simulator.reset()
        total_reward = 0.0
        steps = 0

        trajectory: list[tuple[str, str, float]] = []

        for step in range(max_steps):
            # Choose best action (without exploration)
            action = agent.get_action(state, explore=False)
            reward, next_state, done = simulator.step(action)

            trajectory.append((
                simulator.get_state_name(state),
                action.name,
                reward
            ))

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        print(f"Test episode {episode + 1}:")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        if len(trajectory) <= 20:
            print(f"  Trajectory: ", end="")
            for i, (s, a, r) in enumerate(trajectory):
                if i < len(trajectory) - 1:
                    print(f"{s}-[{a}]->", end="")
                else:
                    print(f"{s} (reward: {r:.1f})")
        print()

    avg_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))

    print(f"{'=' * 70}")
    print(f"Average total reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"{'=' * 70}\n")

    return avg_reward, episode_rewards


def plot_results(
        agent: QLearningAgent,
        episode_rewards: list[float],
        simulator: Simulator,
        title_suffix: str = "",
) -> None:
    """
    Displays training results.
    Args:
        agent: Trained agent
        episode_rewards: Rewards per episode
        simulator: Simulator
        title_suffix: Suffix for graph title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Q-learning results {title_suffix}", fontsize=16)

    # 1. Rewards per episode
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, label="Reward per episode")

    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        ax1.plot(range(window - 1, len(episode_rewards)),
                 moving_avg,
                 color='red',
                 linewidth=2,
                 label=f'Moving average ({window} episodes)')

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total reward")
    ax1.set_title("Rewards during training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. V-values during learning
    ax2 = axes[0, 1]
    all_states = list(simulator.STATE_TO_COORD.keys())

    for state in all_states:
        v_values = [v_dict[state] for v_dict in agent.v_history]
        state_name = simulator.get_state_name(state)
        ax2.plot(v_values, label=state_name, marker='o', markersize=3)

    ax2.set_xlabel("Iteration (× 10 episodes)")
    ax2.set_ylabel("V(s) = max_a Q(s,a)")
    ax2.set_title("Evolution of V-values")
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Final Q-values (heat map)
    ax3 = axes[1, 0]

    # Create matrix of best actions
    best_actions_grid = np.zeros((simulator.ROWS, simulator.COLS), dtype=int)
    best_actions_grid.fill(-1)  # -1 for holes

    for state, (row, col) in simulator.STATE_TO_COORD.items():
        best_action = agent.get_best_action(state)
        best_actions_grid[row, col] = best_action

    # Draw grid
    for state, (row, col) in simulator.STATE_TO_COORD.items():
        v_value = agent.get_v_value(state)
        color = 'lightgreen' if state in simulator.TERMINAL_STATES else 'lightblue'

        rect = plt.Rectangle((col - 0.4, 1 - row - 0.4), 0.8, 0.8,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax3.add_patch(rect)

        # State name
        state_name = simulator.get_state_name(state)
        ax3.text(col, 1 - row + 0.2, state_name,
                 ha='center', va='center', fontsize=10, fontweight='bold')

        # V-value
        ax3.text(col, 1 - row, f"V={v_value:.2f}",
                 ha='center', va='center', fontsize=8)

        # Best action
        best_action = agent.get_best_action(state)
        arrow = arrow_map[best_action]
        ax3.text(col, 1 - row - 0.2, arrow,
                 ha='center', va='center', fontsize=16, color='red')

    ax3.set_xlim(-0.5, simulator.COLS - 0.5)
    ax3.set_ylim(-0.5, simulator.ROWS - 0.5)
    ax3.set_aspect('equal')
    ax3.set_title("Learned policy and V-values")
    ax3.axis('off')

    # 4. Learning rate over time
    ax4 = axes[1, 1]

    if agent.learning_rate_type == LearningRateType.VARIABLE:
        episodes = list(range(1, len(episode_rewards) + 1))
        alphas = [agent.get_learning_rate(e) for e in range(len(episode_rewards))]
        ax4.plot(episodes, alphas, linewidth=2, color='purple')
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("α (learning rate)")
        ax4.set_title("Variable learning rate: α = ln(e+1)/(e+1)")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, f"Constant learning rate\nα = {agent.constant_alpha}",
                 ha='center', va='center', fontsize=14,
                 transform=ax4.transAxes)
        ax4.set_title("Constant learning rate")
        ax4.axis('off')

    plt.tight_layout()
    filename = f"../out/01-q-learning/{title_suffix.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150)
    print(f"Graph saved as: {filename}")
    plt.show()


if __name__ == "__main__":

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("Q-LEARNING EXPERIMENTS FOR GRID WORLD")
    print("=" * 70)

    # ========================================================================
    # EXPERIMENT 1: Variable learning rate, γ = 0.9
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Variable learning rate, γ = 0.9")
    print("=" * 70)

    simulator1 = Simulator()
    agent1 = QLearningAgent(
        gamma=0.9,
        epsilon=0.1,
        learning_rate_type=LearningRateType.VARIABLE,
    )

    stats1 = train_q_learning(agent1, simulator1, num_episodes=1000)
    avg_reward1, rewards1 = test_policy(agent1, simulator1, num_test_episodes=10)
    plot_results(agent1, stats1["episode_rewards"], simulator1,
                 title_suffix="(γ=0.9, variable α)")

    # ========================================================================
    # EXPERIMENT 2: Constant learning rate, γ = 0.9
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Constant learning rate, γ = 0.9")
    print("=" * 70)

    simulator2 = Simulator()
    agent2 = QLearningAgent(
        gamma=0.9,
        epsilon=0.1,
        learning_rate_type=LearningRateType.CONSTANT,
        constant_alpha=0.1
    )

    stats2 = train_q_learning(agent2, simulator2, num_episodes=1000)
    avg_reward2, rewards2 = test_policy(agent2, simulator2, num_test_episodes=10)
    plot_results(agent2, stats2["episode_rewards"], simulator2,
                 title_suffix="(γ=0.9, constant α=0.1)")

    # ========================================================================
    # EXPERIMENT 3: Variable learning rate, γ = 0.999
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Variable learning rate, γ = 0.999")
    print("=" * 70)

    simulator3 = Simulator()
    agent3 = QLearningAgent(
        gamma=0.999,
        epsilon=0.1,
        learning_rate_type=LearningRateType.VARIABLE,
    )

    stats3 = train_q_learning(agent3, simulator3, num_episodes=1000)
    avg_reward3, rewards3 = test_policy(agent3, simulator3, num_test_episodes=10)
    plot_results(agent3, stats3["episode_rewards"], simulator3,
                 title_suffix="(γ=0.999, variable α)")

    # ========================================================================
    # COMPARISON OF RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON OF RESULTS")
    print("=" * 70)

    print(f"\n1. Variable α, γ=0.9:")
    print(f"   Average reward: {avg_reward1:.3f}")

    print(f"\n2. Constant α=0.1, γ=0.9:")
    print(f"   Average reward: {avg_reward2:.3f}")

    print(f"\n3. Variable α, γ=0.999:")
    print(f"   Average reward: {avg_reward3:.3f}")

    print(f"\n{'=' * 70}")
    print("ANALYSIS:")
    print("=" * 70)

    print("\n Impact of learning rate type (α):")
    if abs(avg_reward1 - avg_reward2) < 0.1:
        print("   • Both strategies (variable and constant) give similar results.")
    elif avg_reward1 > avg_reward2:
        print("   • Variable learning rate gives better results.")
    else:
        print("   • Constant learning rate gives better results.")

    print("\n Impact of discount factor (γ):")
    diff_gamma = avg_reward3 - avg_reward1
    if abs(diff_gamma) < 0.1:
        print(f"   • Small difference between γ=0.9 and γ=0.999 ({diff_gamma:.3f})")
    elif diff_gamma > 0:
        print(f"   • γ=0.999 gives BETTER results (+{diff_gamma:.3f})")
        print("   • Higher γ values future rewards more → agent is more 'far-sighted'")
    else:
        print(f"   • γ=0.9 gives better results ({abs(diff_gamma):.3f})")

    print("\n" + "=" * 70)
    print("Experiments finished!")
    print("=" * 70 + "\n")
