"""Domain curriculum — 48 probe prompts with expected-keyword sets.

Each probe is a question in the model's own textbook domains; the keyword
sets are the concepts a competent answer should touch. Scoring is lenient
substring matching — this measures topic coverage, not prose quality.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    domain: str
    prompt: str
    keywords: tuple[str, ...]


DOMAINS: dict[str, tuple[Probe, ...]] = {
    "cybersecurity": (
        Probe("cybersecurity", "Explain what a firewall does.",
              ("traffic", "network", "filter", "block")),
        Probe("cybersecurity", "Why should people use strong passwords?",
              ("password", "guess", "crack", "long", "unique")),
        Probe("cybersecurity", "What is phishing?",
              ("email", "fake", "trick", "steal", "click")),
        Probe("cybersecurity", "What does two-factor authentication add?",
              ("code", "second", "phone", "extra", "step")),
        Probe("cybersecurity", "How does encryption keep messages secret?",
              ("key", "secret", "encrypt", "scramble", "read")),
        Probe("cybersecurity", "What does a VPN do?",
              ("tunnel", "private", "hide", "internet", "encrypt")),
        Probe("cybersecurity", "Why are software updates important?",
              ("fix", "security", "hole", "patch", "bug")),
        Probe("cybersecurity", "What is malware?",
              ("harmful", "software", "virus", "computer", "install")),
        Probe("cybersecurity", "What happens in a data breach?",
              ("stolen", "data", "information", "company", "leak")),
        Probe("cybersecurity", "Why is public wifi risky?",
              ("same", "network", "see", "traffic", "attacker")),
        Probe("cybersecurity", "What does ransomware do?",
              ("files", "lock", "pay", "encrypt", "money")),
        Probe("cybersecurity", "What is social engineering?",
              ("people", "trick", "trust", "information", "lie")),
    ),
    "python": (
        Probe("python", "What is a variable in python?",
              ("store", "value", "name", "data")),
        Probe("python", "What does a loop do in python?",
              ("repeat", "again", "list", "times")),
        Probe("python", "What is a function in python?",
              ("code", "call", "reuse", "name", "return")),
        Probe("python", "What is a python list?",
              ("items", "order", "collection", "hold", "many")),
        Probe("python", "What is a dictionary in python?",
              ("key", "value", "pairs", "lookup")),
        Probe("python", "Why do programmers test their code?",
              ("bugs", "mistakes", "check", "works", "catch")),
        Probe("python", "What does debugging mean?",
              ("find", "fix", "mistake", "error", "problem")),
        Probe("python", "What is an API?",
              ("ask", "program", "data", "interface", "other")),
        Probe("python", "What is a library in python?",
              ("code", "reuse", "import", "others", "written")),
        Probe("python", "What is recursion?",
              ("calls", "itself", "smaller", "base", "repeat")),
        Probe("python", "What does it mean to sort a list?",
              ("order", "arrange", "small", "big", "alphabet")),
        Probe("python", "Why write readable code?",
              ("others", "read", "understand", "later", "team")),
    ),
    "computers": (
        Probe("computers", "How does the internet carry a message?",
              ("packets", "routers", "across", "network", "split")),
        Probe("computers", "What does a server do?",
              ("serves", "requests", "waits", "responds", "stores")),
        Probe("computers", "What does DNS translate?",
              ("name", "address", "domain", "ip", "numbers")),
        Probe("computers", "How does a CPU compute?",
              ("instructions", "fast", "calculate", "steps", "small")),
        Probe("computers", "What is the difference between memory and storage?",
              ("temporary", "permanent", "ram", "disk", "forget")),
        Probe("computers", "What does an operating system decide?",
              ("programs", "run", "resources", "between", "manages")),
        Probe("computers", "How does wifi send data through walls?",
              ("radio", "waves", "air", "signal", "wireless")),
        Probe("computers", "What is an IP address?",
              ("number", "identify", "device", "internet", "location")),
        Probe("computers", "How do search engines rank pages?",
              ("links", "popular", "keywords", "relevance", "order")),
        Probe("computers", "What is a database?",
              ("stores", "organized", "data", "query", "tables")),
        Probe("computers", "Why do computers need fans?",
              ("hot", "heat", "cool", "chips", "warm")),
        Probe("computers", "What does open source mean?",
              ("code", "free", "anyone", "public", "read")),
    ),
    "stories": (
        Probe("stories", "Tell me a story about a lost mitten.",
              ("mitten", "lost", "found", "looked", "cold")),
        Probe("stories", "Tell me a story about the first snow.",
              ("snow", "winter", "white", "fell", "cold")),
        Probe("stories", "Tell me a story about a paper boat.",
              ("boat", "water", "sail", "floated", "river")),
        Probe("stories", "Tell me a story about a sleepless firefly.",
              ("firefly", "light", "night", "glow", "awake")),
        Probe("stories", "Tell me a story about a puddle after rain.",
              ("puddle", "rain", "jump", "water", "wet")),
        Probe("stories", "Tell me a story about the bakery cat.",
              ("cat", "bakery", "bread", "sleep", "warm")),
        Probe("stories", "Tell me a story about a kite that wouldn't land.",
              ("kite", "wind", "fly", "string", "sky")),
        Probe("stories", "Tell me a story about grandma's radio.",
              ("radio", "music", "grandma", "old", "song")),
        Probe("stories", "Tell me a story about a snail race.",
              ("snail", "race", "slow", "finish", "won")),
        Probe("stories", "Tell me a story about the missing button.",
              ("button", "lost", "coat", "sew", "found")),
        Probe("stories", "Tell me a story about a cardboard rocket.",
              ("rocket", "space", "cardboard", "fly", "stars")),
        Probe("stories", "Tell me a story about a windy Tuesday.",
              ("wind", "blew", "hat", "day", "flew")),
    ),
}

DOMAIN_NAMES: tuple[str, ...] = tuple(DOMAINS)


def probes_for(domain: str) -> tuple[Probe, ...]:
    """The 12 probes of one domain (KeyError on an unknown domain)."""
    return DOMAINS[domain]


def all_probes() -> list[Probe]:
    """Every probe, domains in DOMAINS order — the full 48-probe curriculum."""
    return [p for domain in DOMAIN_NAMES for p in DOMAINS[domain]]


# Seed texts the planner draws study material from — the same declarative
# sentences evalforge mines for eval pairs, one chapter per domain. In the
# real loop the Academy's teachers rewrite/expand these into the textbook.
_SEED_LIBRARY: dict[str, tuple[str, ...]] = {
    "cybersecurity": (
        "A firewall filters network traffic between trusted and untrusted networks and blocks unsafe connections.",
        "Two-factor authentication adds a second layer of security beyond the password by requiring a code or device.",
        "Phishing tricks users into revealing credentials through fake emails that look trustworthy.",
        "Encryption protects data by turning it into ciphertext that only the key holder can read.",
        "A software patch fixes a known vulnerability, so updating closes the door on attackers.",
        "Antivirus software scans files, detects malware signatures and removes infected programs.",
    ),
    "python": (
        "A function is a reusable piece of code that you call by name and it can return a value.",
        "A dictionary stores key and value pairs and looks up a value by its key.",
        "A loop repeats a block of code once for every item in a list.",
        "A variable is a name that stores a value so the code can use it later.",
        "Programmers test their code to catch bugs before users find the mistakes.",
        "Readable code is code that others can read and understand later.",
    ),
    "computers": (
        "The internet splits a message into packets and routers forward each packet across the network.",
        "A server waits for requests and responds to them, often storing the data too.",
        "DNS translates a domain name into the IP address numbers that identify a device on the internet.",
        "The CPU computes by following instructions in small fast steps.",
        "Memory is temporary and forgets when the power goes off, storage is permanent.",
        "An operating system manages the resources between the programs that run.",
    ),
    "stories": (
        "The lost mitten was found in the snow and the bakery cat slept on it, warm at last.",
        "A paper boat sailed down the river and floated past the sleepy town.",
        "The firefly glowed all night, the only light awake in the garden.",
        "The snail raced the kettle and, slow and steady, won by a nose.",
        "The cardboard rocket flew past the stars and landed back in the yard by daybreak.",
        "On a windy Tuesday the breeze blew the hat clean off and it flew like a kite.",
    ),
}


def textbook_library() -> dict[str, list[str]]:
    """Seed texts per domain the planner commissions study material from."""
    return {domain: list(texts) for domain, texts in _SEED_LIBRARY.items()}
