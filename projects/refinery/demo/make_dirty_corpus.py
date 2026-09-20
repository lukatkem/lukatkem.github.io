#!/usr/bin/env python3
"""Generate a deliberately dirty corpus to demonstrate the Refinery."""
from __future__ import annotations

import random
from pathlib import Path

random.seed(7)
OUT = Path(__file__).parent / "dirty"

NAMES = ["Mila", "Theo", "Nia", "Sam", "Ada", "Leo", "Ivy", "Max"]
THINGS = ["a red mitten", "a paper boat", "a lost puppy", "a small kite",
          "an old map", "a shiny shell", "a warm scarf", "a tiny lantern"]
PLACES = ["the school garden", "the quiet beach", "the pine forest",
          "the corner bakery", "the rooftop", "the river bank"]

STORY = (
    "{name} found {thing} near {place}. It looked special, so {name} took it "
    "home and cleaned it carefully. Every day after school {name} worked on it, "
    "fixing and polishing until it shone again. Friends came by to see it and "
    "everyone agreed it was the best one they had ever seen. On Sunday, {name} "
    "returned to {place} and showed it to the whole neighborhood. Everybody "
    "clapped and {name} felt proud. That night {name} fell asleep smiling, "
    "dreaming of the next small adventure to come. The end."
)

# four independent story parts → 4×4×4×4 = 256 genuinely distinct documents
OPENINGS = [
    "{name} woke up early and walked to {place}. The morning was cool and the streets were quiet.",
    "On a bright Saturday, {name} wandered to {place} with a sandwich and a book in a canvas bag.",
    "It had rained all night, and {name} wanted to see what {place} looked like after a storm.",
    "{name} had one job that afternoon: deliver a package to {place} before the sun went down.",
]
MIDDLES = [
    "That was where {name} found {thing}, tucked behind a bench as if someone had left it in a hurry.",
    "While cutting across the grass, {name} spotted {thing} glinting between two paving stones.",
    "An old man at {place} handed {name} {thing} and said it had been waiting a long time.",
    "{name} almost stepped on {thing}, stopped, knelt down, and picked it up with both hands.",
]
RESOLUTIONS = [
    "It took three evenings of patient work to clean and repair, but every minute felt worth it.",
    "At home, {name} wiped off the dirt, tightened the loose parts, and painted a tiny blue star on it.",
    "{name} asked a neighbor for glue and help, and together they made it better than before.",
    "The repair was not perfect, but the small scar in the corner became {name}'s favorite detail.",
]
CLOSINGS = [
    "When friends finally saw it, they agreed it was one of a kind. {name} slept well that night.",
    "On Sunday everyone gathered at {place} to see it. There was cake, laughter, and a round of applause.",
    "A week later {name} returned it to the very spot, for the next curious person to find and love.",
    "The story spread through the neighborhood, and for months people called it {name}'s little miracle.",
]

FOREIGN = ("გამარჯობა, ეს არის ქართული ტექსტი ბევრი სიტყვებით და წინადადებებით. "
           "ქართული ენა ძალიან ლამაზია და მდიდარია თავისი ანბანით. "
           "ბავშვები კითხულობენ წიგნებს და სწავლობენ ახალ სიტყვებს ყოველ დღე.")
CYRILLIC = ("Это небольшой текст на русском языке про кошку и её котят. "
            "Кошка спала на окне, а котята играли с клубком ниток весь день.")


def clean_doc(i: int) -> str:
    return " ".join(
        part.format(name=random.choice(NAMES), thing=random.choice(THINGS),
                    place=random.choice(PLACES))
        for part in (random.choice(OPENINGS), random.choice(MIDDLES),
                     random.choice(RESOLUTIONS), random.choice(CLOSINGS))
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in OUT.glob("*.txt"):
        f.unlink()

    n = 0
    def write(body: str) -> None:
        nonlocal n
        Path(OUT / f"doc{n:03d}.txt").write_text(body, encoding="utf-8")
        n += 1

    # 45 clean stories
    cleans = [clean_doc(i) for i in range(45)]
    for body in cleans:
        write(body)
    # 12 exact duplicates
    for i in range(12):
        write(random.choice(cleans))
    # 10 near-duplicates: one word swapped, or a tail appended
    for i in range(10):
        d = random.choice(cleans)
        for old, new in random.sample([("found", "discovered"), ("smiling", "grinning"),
                                       ("proud", "happy"), ("home", "house")], 1):
            d = d.replace(old, new, 1)
        write(d)
    # 6 near-duplicates with a small edit + tail (harder case)
    for i in range(6):
        d = random.choice(cleans)
        d = d.replace("Everyone", "All of them", 1) + " And then they had cake."
        write(d)
    # 8 junk docs
    for i in range(4):
        write("!!! $$$ ### %%% &&& !!! " * 40)
    for i in range(4):
        line = "Buy our product now, limited offer, click the link below!!!"
        write((line + "\n") * 15)
    # 5 lorem filler
    for i in range(5):
        write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
              "eiusmod tempor incididunt ut labore et dolore magna aliqua. " * 6)
    # 8 docs with leaked PII / secrets
    for i in range(4):
        write(clean_doc(i) + f"\n\nQuestions? Email admin{i}@corpus-leak.example or call +1 (555) 010-{1000+i:04d}.")
    for i in range(2):
        write(clean_doc(i + 10) + "\n\nConfig: OPENROUTER_API_KEY=sk-or-v1-0000000000000000000000000000000000000DEADBEEF")
    for i in range(2):
        write(clean_doc(i + 20) + "\n\nAuth: eyJhbGciOiJIUzI1NiJ9.eyJraWxvIjp0cnVlfQ.fakefakefakefakefakefakefake")
    # 6 foreign-language docs
    for i in range(4):
        write(FOREIGN)
    for i in range(2):
        write(CYRILLIC)

    print(f"wrote {n} dirty documents → {OUT}")


if __name__ == "__main__":
    main()
