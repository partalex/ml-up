# Grid World Simulator

## Struktura Okruženja

Grid world okruženje je organizovano kao 2×5 grid sa rupama na pozicijama B2 i B4:

```
  A1(S)  A2    A3    A4    A5
  B1(T)  --    B3(T) --    B5(T)
```

### Stanja

| ID | Naziv | Koordinate | Tip            | Nagrada |
|----|-------|------------|----------------|---------|
| 0  | A1    | (0, 0)     | Početno (S)    | 0       |
| 1  | A2    | (0, 1)     | Obično         | 0       |
| 2  | A3    | (0, 2)     | Obično         | 0       |
| 3  | A4    | (0, 3)     | Obično         | 0       |
| 4  | A5    | (0, 4)     | Obično         | 0       |
| 5  | B1    | (1, 0)     | Terminalno (T) | -1      |
| 6  | B3    | (1, 2)     | Terminalno (T) | -1      |
| 7  | B5    | (1, 4)     | Terminalno (T) | +3      |

**Napomena**: Pozicije B2 (1, 1) i B4 (1, 3) ne postoje - to su "rupe" u grid-u.

## Akcije

Agent ima na raspolaganju 4 akcije:

- `UP` (0): gore
- `DOWN` (1): dole
- `LEFT` (2): levo
- `RIGHT` (3): desno

## Dinamika Okruženja

### Stohastičnost

Kada agent odabere akciju, okruženje je stohastičko:

- **70% verovatnoća**: agent se pomera u izabranom smeru
- **10% verovatnoća**: agent se pomera u svakom od preostala 3 smera

### Zidovi i Rupe

Ako agent pokuša da:

- Izađe van granica grid-a (udari u zid) → ostaje u istom stanju
- Uđe u rupu (B2 ili B4) → ostaje u istom stanju

### Terminalna Stanja

Stanja B1, B3 i B5 su terminalna:

- **Kada agent uđe u terminalno stanje**: Dobija nagradu 0 i `done=True`
- **Kada agent preduzme akciju u terminalnom stanju**: Tada dobija odgovarajuću nagradu (-1, -1, ili +3)
- Epizoda se završava
- Agent se automatski vraća u početno stanje A1

**Važno**: Nagrada se dobija tek kada agent **preduzme akciju** u terminalnom stanju, ne pri samom ulasku!

**Primer**:
**Primer**:

```
Korak 1: Agent u A4
         ↓ (izvršava DOWN)
         Rezultat: stanje=B5, reward=0, done=True
         └→ Ušao u terminalno stanje, ali JOŠ NEMA NAGRADE

Korak 2: Agent u B5 (terminalno)
         ↓ (izvršava bilo koju akciju, npr. UP)
         Rezultat: stanje=A1, reward=+3, done=True
         └→ SADA dobija nagradu i automatski se resetuje
```

| Korak | Stanje PRE | Akcija | Stanje POSLE | Reward | Done | Objašnjenje          |
|-------|------------|--------|--------------|--------|------|----------------------|
| 1     | A4         | DOWN   | B5           | 0      | True | Ulazak u terminalno  |
| 2     | B5         | UP     | A1           | +3     | True | Akcija u terminalnom |

## Primer Korišćenja

```python
from simulator import Simulator, Action

# Inicijalizuj simulator
sim = Simulator()

# Izvrši akciju
reward, new_state, done = sim.step(Action.RIGHT)

# Proveri stanje
print(f"Novo stanje: {sim.get_state_name(new_state)}")
print(f"Nagrada: {reward}")
print(f"Završeno: {done}")

# Resetuj simulator
sim.reset()
```

## Pokretanje

```bash
# Pokreni primer
python src/main.py

# Proveri tipove
mypy src/simulator.py src/main.py
```

## Važne napomene

### Nagrađivanje u terminalnim stanjima

**Ključna osobina**: Nagrada se dobija tek kada agent **preduzme akciju** u terminalnom stanju, ne pri samom ulasku!

**Razlog**: Ovo je u skladu sa MDP (Markov Decision Process) semantikom gde nagrada dolazi kao posledica akcije, ne
stanja.

Za detaljno objašnjenje, pogledajte: [REWARD_TIMING.md](REWARD_TIMING.md)

## Struktura Projekta

```
ml-up/
├── src/
│   ├── simulator.py    # Grid world simulator
│   └── main.py         # Primer korišćenja
├── test_grid.py        # Test skripta
├── mypy.ini            # Konfiguracija mypy
├── requirements.txt    # Python dependencies
└── README.md           # Ova dokumentacija
```

