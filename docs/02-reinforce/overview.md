## 📋 ŠTA JE REINFORCE?

**REINFORCE** je **policy gradient** algoritam koji **direktno uči stohastičku politiku** π_θ(a|s).

### Glavna ideja:

1. **Sakupi celu epizodu**: (s₀, a₀, r₀), (s₁, a₁, r₁), ..., (sₜ, aₜ, rₜ)
2. **Izračunaj returns**: Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...
3. **Ažuriraj politiku**: θ ← θ + α·Gₜ·∇log π(a|s)

### Softmax politika:

```
π(a|s) = exp(θ(s,a)) / Σ exp(θ(s,a'))
```

---

## ✅ IMPLEMENTIRANE FUNKCIONALNOSTI

### Algoritam

- ✅ Softmax politika (numerički stabilna)
- ✅ Monte Carlo returns
- ✅ Policy gradient update
- ✅ Promenljiva stopa učenja: α = ln(e+1)/(e+1)
- ✅ Konstantna stopa učenja: α = 0.01

### Praćenje napretka (kako je traženo)

- ✅ **Zamrzavanje politike** svakih 100 epizoda
- ✅ **10 test epizoda** za evaluaciju
- ✅ **Prosečna ukupna nagrada** se računa i beleži

### Grafici

- ✅ **Graf 1**: Nagrade tokom treniranja
- ✅ **Graf 2**: Prosečna nagrada u 10 test epizoda 📊
- ✅ **Graf 3**: Parametri politike θ(s,a) 📊
- ✅ **Graf 4**: Naučena politika (grid)

### Parametri

- ✅ **γ = 0.9** (kako je traženo)
- ✅ **2000 epizoda** treniranja
- ✅ **Test svakih 100 epizoda**

---

## 📊 KLJUČNI GRAFICI

### Graf 2: Prosečna nagrada u 10 test epizoda

**Ovo pokazuje kako se nagrada menja tokom učenja!**

- X-osa: Epizoda (100, 200, ..., 2000)
- Y-osa: Prosečna nagrada
- Trebalo bi da raste: 0.0 → 1.0 → 2.0+

### Graf 3: Parametri politike

**Ovo pokazuje kako se θ(s,a) menjaju!**

- Prikazuje prosečne θ vrednosti za ne-terminalna stanja
- Konvergiraju ka stabilnim vrednostima
- Različite linije za različita stanja (A1, A2, A3, ...)

---

## 🎯 RAZLIKE OD Q-LEARNING

|               | Q-Learning      | REINFORCE    |
|---------------|-----------------|--------------|
| **Tip**       | Value-based     | Policy-based |
| **Uči**       | Q(s,a)          | π_θ(a,s)     |
| **Politika**  | Deterministička | Stohastička  |
| **Update**    | TD (po koraku)  | MC (epizoda) |
| **Brzina**    | Brža            | Sporija      |
| **Varijansa** | Manja           | Veća         |

---

## 🚀 EKSPERIMENTI

### Eksperiment 1: Promenljiva α

- Stopa učenja: α = ln(e+1)/(e+1)
- Počinje ~0.69, pada na ~0.14
- Stabilnija konvergencija

### Eksperiment 2: Konstantna α = 0.01

- Fiksna stopa učenja
- Za poređenje brzine konvergencije

### Output

Za svaki eksperiment:

- PNG grafik sa 4 panela
- Konzolni ispis napretka
- Finalna prosečna nagrada

---

## 💡 OČEKIVANI REZULTATI

### Prosečna nagrada u toku učenja:

```
Epizoda    Očekivana nagrada
  100      0.0 - 0.5  (slučajne akcije)
  500      0.5 - 1.0  (rano učenje)
 1000      1.0 - 1.5  (sredina)
 1500      1.5 - 2.0  (kasno učenje)
 2000      2.0 - 2.5  (dobra politika)
```

### Naučena politika:

- ✅ Agent ide ka **B5** (nagrada +3)
- ✅ Izbegava **B1 i B3** (nagrada -1)
- ✅ Stohastička (daje verovatnoće, ne fiksne akcije)
- ✅ Uzima u obzir stohastičnost okruženja (0.7)

---

## 🎓 TEORIJA (kratko)

### REINFORCE update:

```python
for t in range(T):
    Gₜ = Σ
    γᵏ·rₜ₊ₖ  # Monte Carlo return
    for akciju a:
        θ(sₜ, a) += α·Gₜ·∇log
        π(a | sₜ)
```

### Softmax gradijent:

```python
if a == akcija_uzeta:
    ∇log
    π = 1 - π(a | s)
else:
    ∇log
    π = -π(a | s)
```

Ovo je **score function gradient estimator** - ne treba znati dinamiku okruženja!

---

## ✅ COMPLIANCE SA ZADATKOM

Zadatak je tražio:

| Zahtev                             | Implementirano              |
|------------------------------------|-----------------------------|
| REINFORCE algoritam                | ✅                           |
| Zamrzavati naučenu politiku        | ✅ `run_test_episodes()`     |
| 10 epizoda interakcije             | ✅ `num_episodes=10`         |
| Prosečna ukupna nagrada            | ✅ Računa se i beleži        |
| **Grafički:** Nagrada u 10 epizoda | ✅ Graf 2                    |
| **Grafički:** Parametri θ(s,a)     | ✅ Graf 3                    |
| Eksperimentisati sa α              | ✅ Promenljiva vs konstantna |
| γ = 0.9                            | ✅                           |

---

## 🐛 TROUBLESHOOTING

**Q: "No module named 'numpy'"**  
A: `pip install numpy matplotlib`

**Q: Sporo izvršavanje**  
A: Normalno! 2000 epizoda × 2 eksperimenta = 2-3 min. REINFORCE je sporiji jer koristi Monte Carlo.

**Q: Nagrade osciliraju**  
A: Normalno! REINFORCE ima veću varijansu. Zato pratimo prosek u 10 epizoda.

**Q: Politika nije deterministička**  
A: To je FEATURE! REINFORCE uči stohastičku politiku. Verovatnoće pokazuju "sigurnost".

**Q: Agent ne postiže +3 uvek**  
A: Normalno! Okruženje je stohastično (0.7), teško je uvek ići ka B5.

---

## 🎯 ZAKLJUČAK

### ✅ STATUS: KOMPLETNO IMPLEMENTIRANO

Svi zahtevi iz zadatka su ispunjeni:

- ✅ REINFORCE algoritam
- ✅ Praćenje napretka (zamrzavanje, 10 epizoda)
- ✅ Grafički prikaz nagrada
- ✅ Grafički prikaz parametara θ
- ✅ Eksperimenti sa α
- ✅ γ = 0.9

### Dodatno:

- ✅ Mypy tipizacija (no errors)
- ✅ 4 grafikona po eksperimentu
- ✅ Grid vizualizacija politike
- ✅ Testovi
- ✅ Kompletna dokumentacija

---

## 🚀 BRZI START (još jednom)

```bash
pip install numpy matplotlib
python src/02-reinforce-reinforce-main.py
```

**Trajanje**: 2-3 minuta  
**Output**: 2 PNG grafika + analiza
