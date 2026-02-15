## ✅ Šta je implementirano

### 1. REINFORCE Algoritam (`src/reinforce_main.py`)

**Kompletna implementacija policy gradient metoda:**

#### A. Politika (Softmax)

```python
π(a | s) = exp(θ(s, a)) / Σ_a
' exp(θ(s,a'))
```

- ✅ Numerički stabilna implementacija (oduzima max)
- ✅ Uvek vraća validnu verovatnosnu distribuciju
- ✅ Diferentabilna za gradient descent

#### B. Monte Carlo Returns

```python
G_t = r_t + γ·r_
{t + 1} + γ²·r_
{t + 2} + ... + γ ^ (T - t)·r_T
```

- ✅ Računa diskontovani povrat za svaki korak u epizodi
- ✅ Backward prolazak kroz trajektoriju

#### C. Policy Gradient Update

```python
θ(s, a) ← θ(s, a) + α·G_t·∇log
π(a | s)
```

- ✅ REINFORCE pravilo ažuriranja
- ✅ Gradijent za softmax: `I(a=a_t) - π(a|s)`
- ✅ Ažurira sve (stanje, akcija) parove u epizodi

#### D. Stopa učenja

- ✅ **Promenljiva**: `α = ln(e+1)/(e+1)`
- ✅ **Konstantna**: `α = 0.01` (manja nego za Q-learning)

---

## 📊 Praćenje napretka - IMPLEMENTIRANO

### Kako je traženo u zadatku:

#### 1. ✅ "Zamrzavanje" politike

```python
def run_test_episodes(agent, simulator, num_episodes=10):
# Testira bez učenja (explore=False)
# Računa prosečnu nagradu
```

#### 2. ✅ 10 epizoda interakcije

- Pokreće se **svakih 100 epizoda** treniranja
- Agent koristi trenutnu politiku (greedy, bez istraživanja)
- Računa se **prosečna ukupna nagrada**

#### 3. ✅ Grafički prikaz - "kako se tokom učenja menjaju"

**Graf 2**: Prosečna nagrada u 10 uzastopnih epizoda

- X-osa: Epizoda (100, 200, ..., 2000)
- Y-osa: Prosečna nagrada
- Pokazuje napredak učenja

**Graf 3**: Parametri politike θ(s,a) u ne-terminalnim stanjima

- X-osa: Iteracija testiranja
- Y-osa: Prosečna θ vrednost po stanju
- Prikazuje kako parametri konvergiraju

---

## 🎯 Eksperimenti

### ✅ Eksperiment 1: Promenljiva stopa učenja

- **α**: `ln(e+1)/(e+1)`
- **γ**: 0.9 (kako je traženo)
- **Epizode**: 2000
- **Test**: Svakih 100 epizoda

### ✅ Eksperiment 2: Konstantna stopa učenja

- **α**: 0.01 (fiksna)
- **γ**: 0.9
- **Cilj**: Poređenje sa promenljivom

---

## 📈 Grafici (4 panela po eksperimentu)

### Graf 1: Nagrade tokom treniranja

- Nagrada po epizodi (plava linija)
- Klizni prosek 50 epizoda (crvena linija)
- Pokazuje trend učenja

### Graf 2: Prosečna nagrada u 10 test epizoda ⭐

**OVO JE KLJUČNI GRAF ZA PRAĆENJE NAPRETKA!**

- Prikazuje kako se **prosečna nagrada menja tokom učenja**
- Testira se **svakih 100 epizoda** (10 test epizoda)
- Trebalo bi da raste i stabilizuje se

### Graf 3: Parametri politike θ(s,a) ⭐

**OVO PRIKAZUJE KAKO SE PARAMETRI MENJAJU!**

- Prikazuje prosečne θ vrednosti za ne-terminalna stanja
- Pokazuje konvergenciju politike
- Različite linije za različita stanja

### Graf 4: Naučena politika (grid)

- Strelice: Najbolje akcije
- Brojevi: Verovatnoće akcija
- Terminalna stanja označena sa nagradom

---

## 🔬 Implementacione specifičnosti

### Razlike od Q-Learning

| Aspekt           | Q-Learning      | REINFORCE         |
|------------------|-----------------|-------------------|
| **Tip učenja**   | Value-based     | Policy-based      |
| **Šta uči**      | Q(s,a)          | π_θ(a             |s) |
| **Update**       | TD (po koraku)  | MC (cela epizoda) |
| **Politika**     | Deterministička | Stohastička       |
| **Varijansa**    | Manja           | Veća              |
| **Broj epizoda** | 1000            | 2000              |
| **Stopa učenja** | 0.1             | 0.01              |

### Zašto REINFORCE zahteva više epizoda?

1. **Monte Carlo**: Mora čekati kraj epizode
2. **Veća varijansa**: Returns imaju veliku varijansu
3. **Sample inefficient**: Ne rekoristi iskustvo kao TD

### Zašto manja stopa učenja?

Policy gradient je osetljiviji na velike korake - može "uništiti" naučenu politiku.

---

## 📊 Očekivani rezultati

### Prosečna nagrada (10 test epizoda)

| Faza         | Epizoda | Očekivana nagrada |
|--------------|---------|-------------------|
| Početak      | 100     | 0.0 - 0.5         |
| Rano učenje  | 500     | 0.5 - 1.0         |
| Sredina      | 1000    | 1.0 - 1.5         |
| Kasno učenje | 1500    | 1.5 - 2.0         |
| Kraj         | 2000    | 2.0 - 2.5         |

### Interpretacija

**Ako je prosečna nagrada ~2.0+:**

- ✅ Agent uspešno navigira ka B5 (+3 nagrada)
- ✅ Izbegava B1 i B3 (-1 nagrada)
- ✅ Dobro se nosi sa stohastičnošću okruženja

**Ako je ~1.0-1.5:**

- ⚠️ Agent donekle nauči, ali nije optimalan
- Možda ponekad završi u B1/B3
- Ili treba duže treniranje

**Ako je ~0.0:**

- ❌ Agent nije dobro nauči
- Možda stopa učenja nije dobra
- Ili treba značajno više epizoda

---

## 🎓 Teorijska osnova

### REINFORCE Theorem (Williams, 1992)

Gradijent očekivane nagrade:

```
∇_θ J(θ) = E_π[G_t · ∇_θ log π_θ(a_t|s_t)]
```

### Softmax gradijent

Za softmax politiku:

```
∇_θ log π_θ(a|s) = I(a=a_selected) - π_θ(a|s)
```

Ovo je **score function gradient estimator** - omogućava učenje čak i kada ne znamo dinamiku okruženja!

---

## ✅ Compliance sa zadatkom

### Zadatak je tražio:

✅ **"Zamrzavati do tada naučenu politiku"**
→ `run_test_episodes()` sa `explore=False`

✅ **"Ponavljati 10 epizoda interakcije"**
→ `num_episodes=10` u test funkciji

✅ **"Računati prosečnu ukupnu nagradu"**
→ `np.mean(rewards)` i čuva se

✅ **"Grafički prikazati kako se tokom učenja menjaju:**

- **Nagrada u 10 uzastopnih epizoda"**
  → Graf 2: Test rewards

- **Vrednosti parametara politike u ne-terminalnim stanjima"**
  → Graf 3: θ parametri

✅ **"Eksperimentisati sa stopama učenja"**
→ Promenljiva vs konstantna

✅ **"Usvojiti γ = 0.9"**
→ `gamma=0.9` u oba eksperimenta

---

### Dodatno implementirano:

- ✅ Numerički stabilna softmax politika
- ✅ Detaljni grafici (4 panela)
- ✅ Grid vizualizacija naučene politike
- ✅ Testovi (`test_reinforce_basic.py`)
- ✅ Kompletna dokumentacija
- ✅ Mypy tipizacija

---
