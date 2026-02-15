from simulator import Simulator, Action

if __name__ == "__main__":
    sim = Simulator()

    print("=== Grid World Simulator ===\n")
    print("Grid layout (2×5 with holes at B2 and B4):")
    print("  A1(S)  A2    A3    A4    A5")
    print("  B1(T)  --    B3(T) --    B5(T)")
    print()
    print("Terminal states:")
    print("  B1 (ID=5): reward = -1")
    print("  B3 (ID=6): reward = -1")
    print("  B5 (ID=7): reward = +3")
    print()
    print("Stochasticity:")
    print("  - 0.7 probability: goes in chosen direction")
    print("  - 0.1 probability: goes in each of the other 3 directions")
    print()
    print("⚠️  IMPORTANT: Reward is obtained only when agent TAKES ACTION")
    print("    in terminal state, not upon entering!")
    print("\n" + "=" * 60 + "\n")

    # Run episode
    actions = [Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.DOWN, Action.RIGHT]

    for step_num in range(25):
        # Agent chooses action
        action = actions[step_num % len(actions)]

        current_state = sim.state
        current_state_name = sim.get_state_name(current_state)

        # Simulator executes action and returns results
        reward, new_state, done = sim.step(action)

        new_state_name = sim.get_state_name(new_state)

        print(f"Step {step_num + 1}:")
        print(f"  State: {current_state_name} (ID={current_state})")
        print(f"  Action: {action.name}")
        print(f"  New state: {new_state_name} (ID={new_state})")

        # Special marker if we entered terminal state
        if new_state in sim.TERMINAL_STATES and reward == 0:
            print(f"  ⚠️  Entered terminal state! Reward will be obtained in next step.")

        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        print()

        if done:
            print(">>> Episode finished! Agent returns to initial state A1.\n")
            if step_num < 24:  # If not at the end, continue with new episode
                print("--- New episode ---\n")

    # Manual reset demonstration
    print("\n" + "=" * 60)
    print("Manual simulator reset...")
    initial = sim.reset()
    print(f"Simulator reset to state: {sim.get_state_name(initial)} (ID={initial})")
