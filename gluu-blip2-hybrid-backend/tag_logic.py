# tag_logic.py
from typing import List

OBJECT_LABELS = ["Shoe", "Watch", "Wallet", "Bag", "Purse", "Glasses", "Hat", "Phone", "Jacket", "Belt"]
MATERIAL_LABELS = ["Leather", "Canvas", "Suede", "Synthetic", "Rubber", "Metal", "Plastic", "Glass", "Wood", "Fabric"]

PRODUCT_TAGS = {
    "Shoe": ["creased toe box", "worn sole", "heel wear", "misaligned heel", "sole intact", "upper intact", "visible scuff marks", "clean surface", "smooth finish", "like new"],
    "Watch": ["shiny bezel", "scratched glass", "buckle rusted", "strap discolored", "dial clean", "metal polished", "clean surface", "dial dirty", "well maintained"],
    "Wallet": ["soft leather", "strap cracked", "zipper broken", "corners frayed", "lining clean", "logo faded"],
    "Bag": ["strap cracked", "zipper broken", "lining clean", "corners frayed", "logo faded", "torn material"],
    "Purse": ["zipper broken", "soft leather", "strap wrinkled", "pristine condition"],
    "Glasses": ["frame intact", "lens scratched", "hinge loose", "nose pad clean", "lens spotless"],
    "Hat": ["brim firm", "fabric stretched", "collar worn", "fabric smooth"],
    "Phone": ["screen pristine", "cracked screen", "back cover intact", "body dented", "buttons responsive", "frame scuffed"],
    "Jacket": ["buttons intact", "fabric smooth", "collar worn", "lining damaged"],
    "Belt": ["buckle polished", "buckle rusted", "holes stretched", "strap wrinkled", "leather supple", "strap smooth"]
}

def map_condition_to_score(tags: List[str]) -> int:
    damage_tags = {
        "creased toe box", "worn sole", "heel wear", "misaligned heel", "visible scuff marks",
        "scratched glass", "buckle rusted", "strap discolored", "strap cracked", "zipper broken", "dial dirty",
        "corners frayed", "logo faded", "torn material", "strap wrinkled", "lens scratched",
        "hinge loose", "fabric stretched", "collar worn", "lining damaged", "cracked screen",
        "body dented", "frame scuffed", "holes stretched"
    }
    count = sum(1 for tag in tags if tag in damage_tags)
    if count == 0:
        return 10
    elif count == 1:
        return 8
    elif count == 2:
        return 6
    else:
        return 4
