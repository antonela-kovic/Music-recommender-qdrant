# 📚 Content Recommendation System with Qdrant

---

## 🧠 Opis projekta
Ovaj projekt implementira sustav za preporuku sadržaja koristeći **Qdrant** vektorsku bazu podataka. Sustav temelji preporuke na **semantičkoj sličnosti** između tekstualnih sadržaja, pri čemu koristi **AI modele** za generiranje **tekstualnih embeddinga**.

Koriste se **prethodno istrenirani NLP modeli** (poput `sentence-transformers`) za pretvaranje teksta u numeričke vektore koji predstavljaju značenje teksta. Ti se vektori pohranjuju u Qdrant, koja zatim omogućuje brzo i efikasno pronalaženje sličnog sadržaja temeljenog na unosu korisnika.

---

## 🎯 Ciljevi projekta
- Razviti preporučni sustav temeljen na semantičkoj sličnosti teksta
- Integrirati **Qdrant** kao NoSQL vektorsku bazu
- Primijeniti **AI embedding model** za obradu tekstualnih podataka
- Omogućiti korisniku da unese tekst i dobije relevantne preporuke
- Demonstrirati razlikovanje i suradnju između SQL (opcionalno) i NoSQL baza

---

## 🛠️ Korištene tehnologije
- **Qdrant** (vektorska baza podataka)
- **Python** + `qdrant-client` (integracija s bazom)
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) za generiranje embeddinga
- **Google Colab** kao razvojno okruženje
- (Opcionalno) **SQLite** za relacijsku bazu s metapodacima

---

## 🔄 Kratki opis arhitekture
1. Korisnik unosi tekstualni upit
2. AI model generira embedding tog teksta
3. Qdrant pretražuje najbliže embeddinge po semantičkoj sličnosti
4. Sustav vraća relevantne sadržaje kao preporuke

---

## 📦 Ulazni podaci (primjeri)
- Naslovi i opisi članaka
- Kratki tekstovi (npr. opisi recepata, vijesti, filmova)
- Svaki podatak sadrži: `id`, `tekst`, (opcionalno: `kategorija`, `datum`)

---

## 📈 Očekivani rezultati
- Funkcionalan demo u Colabu
- Primjer kako AI i NoSQL mogu raditi zajedno u sustavu preporuka
- Potencijalna ekspanzija s relacijskom bazom (za dodatni sloj podataka)

---

## ✅ Status
Projekt je u fazi izrade i testiranja.

---




