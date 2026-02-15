# REINFORCE Algoritam - Dokumentacija

## Šta je REINFORCE?

**REINFORCE** je policy gradient algoritam koji direktno uči stohastičku politiku π_θ(a|s) umesto učenja vrednosnih
funkcija (kao Q-learning).

### Ključne razlike od Q-learning:

| Aspekt       | Q-Learning               | REINFORCE                     |
|--------------|--------------------------|-------------------------------|
| Šta uči      | Q-vrednosti Q(s,a)       | Politiku π_θ(a,s)             |
| Tip politike | Deterministička (greedy) | Stohastička (probabilistička) |
| Update       | TD (Temporal Difference) | Monte Carlo (cela epizoda)    |
| Pristup      | Value-based              | Policy-based                  |
| Parametri    | Q-tabela                 | θ parametri politike          |

## Implementacija

### 1. Politika (Softmax)

Politika je definisana kao softmax distribucija preko θ parametara:

```
π(a|s) = exp(θ(s,a)) / Σ_a' exp(θ(s,a'))
```

**Svojstva:**

- Uvek daje validnu verovatnosnu distribuciju (suma = 1)
- Diferentabilna (potrebno za gradient descent)
- Veće θ(s,a) → veća verovatnoća za akciju a

### 2. REINFORCE Update

Nakon svake epizode, parametri se ažuriraju:

```
θ(s,a) ← θ(s,a) + α·G_t·∇log π(a|s)
```

Gde je:

- **G_t** = Σ_{k=t}^T γ^(k-t) · r_k (diskontovani povrat od koraka t)
- **∇log π(a|s)** = I(a_t=a) - π(a|s) (gradijent za softmax)
- **α** = stopa učenja

### 3. Monte Carlo Returns

Za svaki korak t u epizodi, računamo diskontovani povrat:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^(T-t)·r_T
```

Ovo zahteva **celu epizodu** pre ažuriranja (za razliku od Q-learning koji ažurira posle svakog koraka).

## Implementirane funkcionalnosti

### ✅ Glavne funkcije

1. **`get_action_probabilities(state)`**
    - Računa π(a|s) za sve akcije koristeći softmax
    - Numerički stabilan (oduzima max)

2. **`select_action(state, explore=True)`**
    - `explore=True`: Uzorkuje akciju po π(a|s)
    - `explore=False`: Bira akciju sa najvećom verovatnoćom (za test)

3. **`update_policy(trajectory, alpha)`**
    - Implementira REINFORCE algoritam
    - Ažurira θ za sve (stanje, akcija) parove u epizodi

4. **`run_test_episodes(agent, simulator, num_episodes=10)`**
    - "Zamrzava" politiku
    - Pokreće 10 test epizoda
    - Računa prosečnu nagradu

5. **`train_reinforce(...)`**
    - Trenira agenta
    - Testira svakih `test_interval` epizoda
    - Beleži parametre politike i test nagrade

### ✅ Praćenje napretka

Implementirano kako je traženo:

1. **Prosečna nagrada u 10 test epizoda**
    - Testira se svakih 100 epizoda treniranja
    - Grafički prikazano kako se menja tokom učenja

2. **Parametri politike θ(s,a)**
    - Beleže se za ne-terminalna stanja
    - Grafički prikazan prosek θ za ključna stanja
    - Prikazuje kako politika evoluira

### ✅ Vizualizacija

Generiše 4 grafika:

1. **Nagrade tokom treniranja**
    - Nagrada po epizodi
    - Klizni prosek (50 epizoda)

2. **Test nagrade**
    - Prosečna nagrada u 10 uzastopnih test epizoda
    - Prikazano kako se menja tokom učenja

3. **Parametri politike**
    - Prosečne θ vrednosti za ključna ne-terminalna stanja
    - Pokazuje kako se parametri menjaju

4. **Naučena politika**
    - Grid vizualizacija
    - Strelice pokazuju najbolju akciju
    - Verovatnoće najboljih akcija

## Parametri

| Parametar     | Vrednost           | Opis                               |
|---------------|--------------------|------------------------------------|
| γ (gamma)     | 0.9                | Faktor umanjenja (kako je traženo) |
| α (alpha)     | promenljiva / 0.01 | Stopa učenja                       |
| Broj epizoda  | 2000               | Broj epizoda treniranja            |
| Test interval | 100                | Testiranje svakih 100 epizoda      |

### Stopa učenja

#### Promenljiva (default)

```python
α_e = ln(e + 1) / (e + 1)
```

#### Konstantna

```python
α = 0.01  # Manja nego za Q-learning!
```

**Napomena**: REINFORCE je osetljiviji na stopu učenja od Q-learning, pa koristimo manje vrednosti.

## Pokretanje

### Instalacija

```bash
pip install numpy matplotlib
```

### Eksperimenti

```bash
python src/02-reinforce-reinforce-main.py
```

Program pokreće 2 eksperimenta:

1. **Promenljiva stopa učenja, γ=0.9**
2. **Konstantna stopa učenja (α=0.01), γ=0.9**

**Trajanje**: ~2-3 minuta (2000 epizoda × 2 eksperimenta)

## Output

### Grafici

Za svaki eksperiment:

- `reinforce_results_(γ=0.9,_promenljiva_α).png`
- `reinforce_results_(γ=0.9,_konstantna_α=0.01).png`

### Konzolni output

```
Epizoda 100/2000
  Prosečna nagrada u treniranju (poslednjih 100): 0.523
  Prosečna nagrada u testu (10 epizoda): 1.240
  α: 0.4620

Epizoda 200/2000
  Prosečna nagrada u treniranju (poslednjih 100): 0.891
  Prosečna nagrada u testu (10 epizoda): 1.780
  α: 0.3784

...

FINALNO TESTIRANJE (10 epizoda)
Prosečna nagrada: 2.450
```

## Prednosti REINFORCE-a

✅ **Direktno uči politiku**

- Ne treba eksplicitno izvlačiti politiku iz Q-vrednosti

✅ **Prirodno stohastička politika**

- Automatski balansira exploration i exploitation

✅ **Radi u velikim/kontinualnim prostorima akcija**

- Ne treba čuvati Q-vrednost za svaku akciju

✅ **Može učiti stohastičke optimalne politike**

- Korisno u nedeterminističkim okruženjima

## Nedostaci REINFORCE-a

❌ **Visoka varijansa**

- Monte Carlo estimacija ima veliku varijansu
- Sporija konvergencija od TD metoda

❌ **Potrebna cela epizoda**

- Ne može učiti online (korak po korak)

❌ **Sample inefficient**

- Treba više epizoda za dobro učenje od Q-learning

❌ **Osetljiv na stopu učenja**

- Potrebna manja α nego za value-based metode

## Očekivani rezultati

### Konvergencija

- **Promenljiva α**: Stabilnija, ali možda sporija
- **Konstantna α**: Brža na početku, ali može oscilirati

### Naučena politika

Agent bi trebalo da nauči:

- **Ići ka B5** (nagrada +3) iz većine stanja
- **Izbegavati B1 i B3** (nagrada -1)
- **Uzimati u obzir stohastičnost** okruženja

### Prosečna nagrada

U idealnom slučaju:

- Početak: ~0.0 (slučajne akcije)
- Sredina: ~1.0 - 2.0 (učenje u toku)
- Kraj: ~2.0 - 2.5 (dobra politika)

**Napomena**: Teško je uvek dostići B5 zbog stohastičnosti okruženja (0.7 verovatnoća).

## Tumačenje grafika

### Graf 1: Nagrade tokom treniranja

- Treba da raste tokom vremena
- Klizni prosek pokazuje trend

### Graf 2: Test nagrade

- **Ovo je ključni graf!**
- Pokazuje stvarni napredak učenja
- Treba da raste i stabilizuje se

### Graf 3: Parametri politike

- Pokazuje kako se θ vrednosti menjaju
- Treba da konvergiraju ka stabilnim vrednostima

### Graf 4: Naučena politika

- Strelice pokazuju preferiranu akciju
- Verovatnoće pokazuju "sigurnost" politike
- Trebalo bi da pokazuju put ka B5

## Poređenje sa Q-Learning

| Aspekt            | Q-Learning      | REINFORCE             |
|-------------------|-----------------|-----------------------|
| Broj epizoda      | 1000            | 2000                  |
| Tip učenja        | TD (brže)       | Monte Carlo (sporije) |
| Politika          | Deterministička | Stohastička           |
| Stabilnost        | Stabilniji      | Više varijanse        |
| Interpretabilnost | Q-vrednosti     | Verovatnoće akcija    |

---
