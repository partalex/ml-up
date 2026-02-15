# Q-Learning Implementacija

## Implementirane funkcionalnosti

### 1. Q-Learning Algoritam

- **ϵ-gramzivo istraživanje**: Agent bira nasumičnu akciju sa verovatnoćom ϵ, inače bira najbolju akciju
- **Q-tabela**: Čuva Q(s, a) vrednosti za svaku kombinaciju stanja i akcije
- **Ažuriranje**: Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]

### 2. Stopa učenja (α)

#### Promenljiva stopa

- Formula: **α_e = ln(e+1)/(e+1)**, gde je e redni broj epizode
- Karakteristike:
    - Počinje sa većom vrednošću (~0.69 u prvoj epizodi)
    - Postepeno se smanjuje
    - Ne smanjuje se prebrzo (logaritamski)

#### Konstantna stopa

- Fiksna vrednost (npr. α = 0.1) kroz sve epizode
- Koristi se za poređenje sa promenljivom stopom

### 3. Faktor umanjenja (γ)

Eksperimenti sa dve vrednosti:

- **γ = 0.9**: Agent manje vrednuje buduće nagrade
- **γ = 0.999**: Agent skoro jednako vrednuje buduće i trenutne nagrade

### 4. Praćenje V-vrednosti

- **V(s) = max_a Q(s, a)** za svako stanje
- Beleži se tokom treniranja (svakih 10 epizoda)
- Prikazuje konvergenciju algoritma

### 5. Testiranje naučene politike

- 10 test epizoda nakon treniranja
- Agent koristi isključivo best akciju (explore=False)
- Računa se prosečna ukupna nagrada

## Parametri

| Parametar    | Vrednost          | Opis                     |
|--------------|-------------------|--------------------------|
| γ (gamma)    | 0.9 / 0.999       | Faktor umanjenja         |
| ϵ (epsilon)  | 0.1               | Verovatnoća istraživanja |
| α (alpha)    | promenljiva / 0.1 | Stopa učenja             |
| Broj epizoda | 1000              | Broj epizoda treniranja  |
| Test epizode | 10                | Broj test epizoda        |

## Eksperimenti

Program izvršava 3 eksperimenta:

### Eksperiment 1: Promenljiva α, γ=0.9

- Stopa učenja: α_e = ln(e+1)/(e+1)
- Faktor umanjenja: 0.9

### Eksperiment 2: Konstantna α=0.1, γ=0.9

- Stopa učenja: 0.1 (konstantna)
- Faktor umanjenja: 0.9
- **Cilj**: Poređenje brzine konvergencije sa Eksperimentom 1

### Eksperiment 3: Promenljiva α, γ=0.999

- Stopa učenja: α_e = ln(e+1)/(e+1)
- Faktor umanjenja: 0.999
- **Cilj**: Analiza uticaja većeg γ

## Rezultati

Za svaki eksperiment generiše se:

1. **Grafik sa 4 panela**:
    - Nagrade po epizodama (sa kliznim prosekom)
    - Evolucija V-vrednosti tokom učenja
    - Naučena politika (strelice) i finalne V-vrednosti
    - Stopa učenja tokom vremena

2. **Test rezultati**:
    - Prosečna ukupna nagrada kroz 10 test epizoda
    - Standardna devijacija
    - Putanja agenta

3. **Poređenje**:
    - Analiza uticaja tipa stope učenja (promenljiva vs konstantna)
    - Analiza uticaja faktora umanjenja (γ=0.9 vs γ=0.999)

## Očekivani nalazi

### Uticaj stope učenja

**Promenljiva stopa** (α_e = ln(e+1)/(e+1)):

- ✅ **Prednosti**: Stabilnija konvergencija na kraju
- ❌ **Mane**: Može biti sporija na početku

**Konstantna stopa** (α = 0.1):

- ✅ **Prednosti**: Brže početno učenje
- ❌ **Mane**: Može biti nestabilna na kraju

### Uticaj faktora umanjenja

**γ = 0.9**:

- Agent manje vrednuje buduće nagrade
- Fokusiran na krakoročne ciljeve
- Može biti dovoljno za jednostavna okruženja

**γ = 0.999**:

- Agent skoro jednako vrednuje buduće i trenutne nagrade
- "Dalekozorniji" - planira dugoročno
- Može postići bolje rezultate jer vodi računa o celokupnoj putanji do cilja (B5 sa nagradom +3)

### Primer interpretacije

Ako agent sa γ=0.999 postiže prosečnu nagradu od ~2.5, a agent sa γ=0.9 postiže ~1.8:

- Agent sa većim γ je naučio da ciljano ide ka B5 (+3 nagrada)
- Agent sa manjim γ možda prihvata i kraće putanje koje vode u B1 ili B3 (-1 nagrada) jer ne vrednuje dovoljno buduće
  nagrade

## Struktura koda

```python
# class QLearningAgent:
# - __init__(): Inicijalizacija
# - get_learning_rate(): Vraćaαzadatuepizodu
# - get_action(): ϵ - gramzivoistraživanje
# - update_q_value(): Q - learningažuriranje
# - get_v_value(): RačunaV(s) = max_aQ(s, a)
# - record_v_values(): BeležiV - vrednosti

# train_q_learning():
# - Trenira agenta kroz zadati broj epizoda
# - Beleži V - vrednosti tokom učenja

# test_policy():
# - Testira naučenu politiku
# - Računa prosečnu nagradu

# plot_results():
# - Generiše grafike rezultata

# __main__
# - Pokreće sve eksperimente
# - Upoređuje rezultate
```

## Napomene

1. **Stohastičnost okruženja**: Simulator je stohastičan (0.7 verovatnoća za izabranu akciju), što znači da isti
   eksperiment može dati malo drugačije rezultate
2. **Seed**: Postavljen je `random.seed(42)` za reproduktivnost
3. **Rupe u grid-u**: B2 i B4 ne postoje - agent ostaje na mestu ako pokuša da uđe
4. **Terminalna stanja**: Nagrada se dobija tek kada agent preduzme akciju U terminalnom stanju

