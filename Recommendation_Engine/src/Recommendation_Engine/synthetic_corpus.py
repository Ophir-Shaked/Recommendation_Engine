# -*- coding: utf-8 -*-  # Ensures compatibility with Unicode text (titles, comments, sources).

from dataclasses import dataclass  # Provides lightweight class syntax for data containers.
from typing import List, Dict, Optional, Tuple  # Type hints for structured readability and validation.
import random  # Used for deterministic pseudo-random generation of synthetic articles.
from .env_and_imports import PRESET, SECONDS_PER_DAY  # Imports runtime mode and time constants.

# -------------------- Synthetic corpus --------------------
@dataclass
class Article:
    """Represents a single synthetic article in the generated corpus."""
    article_id: str                          # Unique string ID for this article.
    title: str                               # Article title (short text).
    text: str                                # Article body or abstract.
    category: Optional[str] = None           # Top-level topic (e.g., 'cars', 'tech').
    published_ts: Optional[float] = None     # UNIX timestamp of publication.
    source: Optional[str] = None             # Simulated source or publication outlet.

def build_big_corpus(seed: int = 42, now_anchor: Optional[float] = None) -> Tuple[List[Article], float]:
    """
    Build a reproducible synthetic article corpus across multiple domains.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    now_anchor : float, optional
        Optional "current time" timestamp to anchor publication times.

    Returns
    -------
    arts : list[Article]
        List of generated Article dataclass instances.
    NOW : float
        Reference timestamp used as 'now'.
    """
    random.seed(seed)  # Initialize RNG for reproducible results.
    NOW = now_anchor if now_anchor is not None else 1726000000.0  # Default anchor timestamp (~Sep 2024).
    rng = random  # Local alias for readability.

    # --- Define possible news sources per category ---
    sources = {
        "cars": ["AutoNow", "MotorDaily", "GearHead", "RacingLine"],
        "movies": ["FilmDaily", "CinemaWorld", "BoxOffice", "ScreenBuzz"],
        "tech": ["TechBiz", "AIWeekly", "CloudPost", "DataWire"],
        "basketball": ["HoopsWire", "BasketInsider", "CourtReport", "BballDaily"],
        "finance": ["MarketWatch", "FinDaily", "BizLedger", "CapitalPost"],
        "music": ["SoundWave", "BeatTimes", "MusicNow", "TuneReport"],
        "science": ["SciDaily", "NatureWire", "LabNotes", "PhysicsPost"],
        "gaming": ["GameWire", "PixelPress", "EsportsNet", "PlayDaily"],
        "health": ["MedPulse", "WellnessNow", "HealthBeat", "CareReport"],
        "education": ["EduWorld", "CampusPost", "TeachDaily", "LearnWire"],
        "travel": ["TravelNow", "GlobeGuide", "WanderPost", "TrailTimes"],
        "food": ["TastePress", "GourmetWire", "FoodDaily", "KitchenPost"],
    }

    # --- Define token vocabularies and semantic fields per category ---
    brands = ["Ferrari","Porsche","Lamborghini","BMW","Audi","Mercedes","Tesla","Toyota","Nissan","Ford",
              "Chevrolet","McLaren","Aston Martin","Jaguar","Bugatti","Maserati","Kia","Hyundai","Peugeot","Volkswagen"]
    ev_terms = ["electric","EV","battery","charging","range","sustainability","fast-charging","eco-friendly"]
    perf_terms = ["turbo","horsepower","aerodynamics","track","handling","0-100","performance","carbon-fiber"]
    movie_genres = ["action","thriller","drama","comedy","sci-fi","adventure","mystery","family","animated","crime"]
    tech_topics = ["AI","machine learning","cloud","edge","5G","IoT","blockchain","cybersecurity","data centers",
                   "chips","LLM","foundation model","MLOps","vector search","FAISS","RAG","quantization","compilers","systems"]
    bball_teams = ["Lakers","Celtics","Warriors","Heat","Bulls","Mavericks","Nuggets","Kings","Knicks","Suns"]
    fin_terms = ["earnings","revenue","guidance","inflation","rates","IPO","merger","acquisition","dividend",
                 "buyback","volatility","ETF","index","bond","credit","liquidity","macro","forecast"]

    # --- Define category-specific corpus sizes depending on preset mode ---
    counts = dict(
        cars = 3600 if PRESET == "FAST" else 4800,
        movies = 3200 if PRESET == "FAST" else 4200,
        tech = 3200 if PRESET == "FAST" else 4200,
        basketball = 2500 if PRESET == "FAST" else 3400,
        finance = 2200 if PRESET == "FAST" else 3000,
        music = 2100 if PRESET == "FAST" else 2900,
        science = 1900 if PRESET == "FAST" else 2600,
        gaming = 1900 if PRESET == "FAST" else 2600,
        health = 1900 if PRESET == "FAST" else 2600,
        education = 1700 if PRESET == "FAST" else 2300,
        travel = 1700 if PRESET == "FAST" else 2300,
        food = 1700 if PRESET == "FAST" else 2300,
    )
    # Each category generates more or fewer samples depending on preset (FAST = smaller dataset).

    arts: List[Article] = []  # Container for all generated Article objects.
    aid = 1  # Sequential article ID counter.

    def _uniq(aid, seed):  # Helper for appending unique suffixes to titles (for reproducibility traceability).
        return f" • id{aid}-s{seed}"

    # --- Example loop: Generate car articles ---
    for _ in range(counts["cars"]):
        brand = random.choice(brands)  # Choose random car brand.
        focus = random.choice(["ev","perf","mix"])  # Randomly focus on EV, performance, or mix topics.

        # Select keywords and construct title according to focus type.
        if focus == "ev":
            topic = ev_terms + random.sample(perf_terms, k=2)
            title = f"{brand} {random.choice(['Electric','EV'])} {random.choice(['Revolution','Update','Expansion'])}"
        elif focus == "perf":
            topic = perf_terms + random.sample(ev_terms, k=2)
            title = f"{brand} {random.choice(['Turbo','Performance','Track'])} {random.choice(['Arrives','Edition','Package'])}"
        else:
            topic = ev_terms + perf_terms
            title = f"{brand} {random.choice(['Hybrid','Pro','Series'])} {random.choice(['Unveiled','Launched','Revealed'])}"

        # Generate pseudo-realistic specs and numeric attributes.
        hp = random.randint(150, 1100)  # Horsepower.
        z2h = round(random.uniform(2.2, 7.9), 2)  # 0–100 km/h acceleration time.
        rng_km = random.randint(220, 820)  # Range (for EV).
        price = random.randint(25, 350) * 1000  # Approximate price in USD.
        t = f"{title}{_uniq(aid, seed)}"  # Append unique suffix to title.
        text = f"{brand} announces {', '.join(random.sample(topic, k=4))}. Specs: {hp} hp, 0-100 in {z2h}s, {rng_km} km. ≈ ${price:,}."
        ts = NOW - random.randint(0, 210) * SECONDS_PER_DAY  # Assign publication timestamp within past ~7 months.
        src = random.choice(sources["cars"])  # Randomly assign one car news source.

        # Create an Article instance and append it to the list.
        arts.append(Article(str(aid), t, text, "cars", ts, src))
        aid += 1  # Increment article ID.

    # (Truncated in this scaffold: additional loops for movies, tech, sports, etc.)
    # You can replicate this logic for each category defined in `sources` to complete the corpus.

    # Return the generated articles and the reference "current" time.
    return arts, NOW
