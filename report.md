---
title: "Mašinsko učenje - Izveštaj o učenju sa potkrepljivanjem"
author: "Grid World okruženje"
date: "Februar 2026"
geometry: margin=2cm
lang: sr-Latn
---

# Uvod

Ovaj izveštaj prikazuje implementaciju i rezultate dva algoritma učenja sa potkrepljivanjem primenjena na stohastičko
Grid World okruženje:

- **Q-učenje** (off-policy TD kontrola)
- **REINFORCE** (Monte Carlo policy gradient)

## Opis okruženja

Okruženje je 2×5 grid world sa rupama na pozicijama B2 i B4:

```
  A1(S)  A2    A3    A4    A5
  B1(T)  --    B3(T) --    B5(T)
```

**Ključne karakteristike:**

- Početno stanje: A1 (označeno sa S)
- Terminalna stanja: B1 (nagrada: -1), B3 (nagrada: -1), B5 (nagrada: +3)
- Akcije: GORE, DOLE, LEVO, DESNO
- Stohastičke tranzicije: 70% izabrani smer, po 10% svaki drugi smer
- Zidovi blokiraju kretanje (agent ostaje u istom stanju)
- Nagrada se dobija odmah po ulasku u terminalno stanje

---

# Rezultati Q-učenja

Q-učenje je off-policy temporal difference algoritam koji uči optimalnu funkciju akcione vrednosti Q*(s,a).

## Eksperiment 1: Promenljiva stopa učenja ($\gamma$ = 0.9)

Stopa učenja: $\alpha$_e = ln(e+1)/(e+1), gde je e redni broj epizode

![Q-učenje sa promenljivim alpha i game=0.9](out/01-q-learning/(gama=0.9,_variable_alpha).png)

**Zapažanja:**

- Konvergencija postignuta nakon ~1500 epizoda
- V-vrednosti se stabilizuju kako agent uči optimalnu politiku
- Prosečna nagrada konvergira ka optimalnoj vrednosti

## Eksperiment 2: Konstantna stopa učenja ($\gamma$ = 0.9)

Stopa učenja: $\alpha$ = 0.1 (konstantna)

![Q-učenje sa konstantnim alpha=0.1 i gama=0.9](out/01-q-learning/(gama=0.9,_constant_alpha=0.1).png)

**Poređenje:**

- Konstantna stopa učenja pokazuje slično ponašanje konvergencije
- Promenljiva stopa učenja obezbeđuje stabilniju konvergenciju u kasnijim epizodama
- Oba pristupa uspešno uče optimalnu politiku

## Eksperiment 3: Visok faktor umanjenja ($\gamma$ = 0.999)

Stopa učenja: $\alpha$_e = ln(e+1)/(e+1) (promenljiva)

![Q-učenje sa promenljivim alpha i gama=0.999](out/01-q-learning/(gama=0.999,_variable_alpha).png)

**Analiza:**

- Veće $\gamma$ više ceni buduće nagrade
- Agent uči da konzistentnije navigira ka B5 (nagrada: +3)
- Konvergencija traje nešto duže zbog propagacije dugoročnih nagrada

---

# Rezultati REINFORCE algoritma

REINFORCE je Monte Carlo policy gradient algoritam koji direktno optimizuje parametre politike koristeći kompletne
trajektorije epizoda.

## Eksperiment 1: Promenljiva stopa učenja ($\gamma$ = 0.9)

Stopa učenja: $\alpha$_e = ln(e+1)/(e+1)

![REINFORCE sa promenljivim alpha i gama=0.9](out/02-reinforce/(gama=0.9,_variable_alpha).png)

**Zapažanja:**

- Parametri politike evoluiraju da favorizuju akcije koje vode ka B5
- Test nagrade pokazuju visoku varijansu tipičnu za policy gradient metode
- Prosečna nagrada se poboljšava tokom epizoda treniranja
- Politika konvergira da eksploatiše terminalno stanje sa visokom nagradom (B5)

## Eksperiment 2: Poređenje konstantne stope učenja

![REINFORCE sa konstantnim alpha (poređenje sa Q-učenjem)](out/02-reinforce/(gama=0.9,_constant_αlpha=0.01).png)

**Poređenje sa Q-učenjem:**

- REINFORCE obično zahteva više epizoda za konvergenciju
- Veća varijansa u nagradama epizoda zbog stohastičke politike
- Policy gradient metode prirodno uče stohastičke politike
- Q-učenje brže uči determinističku optimalnu politiku

---

# Zaključci

## Q-učenje

- **Prednosti:** Brza konvergencija, stabilno učenje, uči optimalnu determinističku politiku
- **Parametri:** $\epsilon$-gramzivo istraživanje ključno za pronalaženje optimalnih putanja
- **Faktor umanjenja:** $\gamma$ = 0.9 vs $\gamma$ = 0.999 pokazuje kompromis između trenutnih i dugoročnih nagrada

## REINFORCE

- **Prednosti:** Direktna optimizacija politike, prirodno rukuje stohastičkim politikama
- **Izazovi:** Visoka varijansa zahteva više epizoda, osetljiv na stopu učenja
- **Performanse:** Uspešno uči da navigira ka terminalnom stanju sa visokom nagradom

## Opšti nalazi

Oba algoritma uspešno rešavaju Grid World zadatak:

- Q-učenje konvergira brže sa nižom varijansom
- REINFORCE obezbeđuje fleksibilniju reprezentaciju politike
- Izbor zavisi od zahteva problema: deterministička vs stohastička politika

---

# Detalji implementacije

- **Jezik:** Python 3.12
- **Provera tipova:** mypy za statičku verifikaciju tipova
- **Simulator:** Prilagođena Grid World implementacija sa Action enum-om
- **Vizualizacija:** matplotlib za krive treniranja i vizualizaciju politike
- **Reproduktivnost:** Testiranje bazirano na epizodama svakih 100 epizoda treniranja

**Struktura koda:**

- `simulator.py` - Implementacija okruženja
- `q-learn-main.py` - Implementacija Q-učenja
- `reinforce-main.py` - Implementacija REINFORCE algoritma

