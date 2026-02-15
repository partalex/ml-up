from simulator import Simulator, Action

if __name__ == "__main__":
    # Inicijalizacija simulatora
    sim = Simulator()

    print("=== Grid World Simulator ===\n")
    print("Grid layout (2×5 sa rupama na B2 i B4):")
    print("  A1(S)  A2    A3    A4    A5")
    print("  B1(T)  --    B3(T) --    B5(T)")
    print()
    print("Terminalna stanja:")
    print("  B1 (ID=5): nagrada = -1")
    print("  B3 (ID=6): nagrada = -1")
    print("  B5 (ID=7): nagrada = +3")
    print()
    print("Stohastičnost:")
    print("  - 0.7 verovatnoća: ide u izabranom smeru")
    print("  - 0.1 verovatnoća: ide u svakom od ostalih smerova")
    print()
    print("⚠️  VAŽNO: Nagrada se dobija tek kada agent PREDUZME AKCIJU")
    print("    u terminalnom stanju, ne pri samom ulasku!")
    print("\n" + "=" * 60 + "\n")

    # Pokretanje epizode
    actions = [Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.DOWN, Action.RIGHT]

    for step_num in range(25):
        # Agent bira akciju
        action = actions[step_num % len(actions)]

        current_state = sim.state
        current_state_name = sim.get_state_name(current_state)

        # Simulator izvršava akciju i vraća rezultate
        reward, new_state, done = sim.step(action)

        new_state_name = sim.get_state_name(new_state)

        print(f"Korak {step_num + 1}:")
        print(f"  Stanje: {current_state_name} (ID={current_state})")
        print(f"  Akcija: {action.name}")
        print(f"  Novo stanje: {new_state_name} (ID={new_state})")

        # Posebna oznaka ako smo ušli u terminalno stanje
        if new_state in sim.TERMINAL_STATES and reward == 0:
            print(f"  ⚠️  Ušao u terminalno stanje! Nagrada će se dobiti u sledećem koraku.")

        print(f"  Nagrada: {reward:.2f}")
        print(f"  Završeno: {done}")
        print()

        if done:
            print(">>> Epizoda završena! Agent se vraća u početno stanje A1.\n")
            if step_num < 24:  # Ako nismo na kraju, nastavi sa novom epizodom
                print("--- Nova epizoda ---\n")

    # Demonstracija ručnog resetovanja
    print("\n" + "=" * 60)
    print("Ručno resetovanje simulatora...")
    initial = sim.reset()
    print(f"Simulator resetovan na stanje: {sim.get_state_name(initial)} (ID={initial})")
