import numpy as np

NUM_SUPERCLASSES = 9

SUPERCLASS_NAMES = {
    0: "background",
    1: "upper_clothes",
    2: "lower_clothes",
    3: "dress",
    4: "shoes",
    5: "accessories",
    6: "skin",
    7: "hair",
    8: "suit",
}

# Original 59-class id -> 9-class superclass id.
CLASS_TO_SUPERCLASS = {
    0: 0,   # null -> background
    1: 5,   # accessories -> accessories
    2: 5,   # bag -> accessories
    3: 5,   # belt -> accessories
    4: 1,   # blazer -> upper_clothes
    5: 1,   # blouse -> upper_clothes
    6: 3,   # bodysuit -> dress
    7: 4,   # boots -> shoes
    8: 1,   # bra -> upper_clothes
    9: 5,   # bracelet -> accessories
    10: 1,  # cape -> upper_clothes
    11: 1,  # cardigan -> upper_clothes
    12: 4,  # clogs -> shoes
    13: 1,  # coat -> upper_clothes
    14: 3,  # dress -> dress
    15: 5,  # earrings -> accessories
    16: 4,  # flats -> shoes
    17: 5,  # glasses -> accessories
    18: 5,  # gloves -> accessories
    19: 7,  # hair -> hair
    20: 5,  # hat -> accessories
    21: 4,  # heels -> shoes
    22: 1,  # hoodie -> upper_clothes
    23: 1,  # intimate -> upper_clothes
    24: 1,  # jacket -> upper_clothes
    25: 2,  # jeans -> lower_clothes
    26: 1,  # jumper -> upper_clothes
    27: 2,  # leggings -> lower_clothes
    28: 4,  # loafers -> shoes
    29: 5,  # necklace -> accessories
    30: 2,  # panties -> lower_clothes
    31: 2,  # pants -> lower_clothes
    32: 4,  # pumps -> shoes
    33: 5,  # purse -> accessories
    34: 5,  # ring -> accessories
    35: 3,  # romper -> dress
    36: 4,  # sandals -> shoes
    37: 5,  # scarf -> accessories
    38: 1,  # shirt -> upper_clothes
    39: 4,  # shoes -> shoes
    40: 2,  # shorts -> lower_clothes
    41: 6,  # skin -> skin
    42: 2,  # skirt -> lower_clothes
    43: 4,  # sneakers -> shoes
    44: 4,  # socks -> shoes
    45: 2,  # stockings -> lower_clothes
    46: 8,  # suit -> suit
    47: 5,  # sunglasses -> accessories
    48: 1,  # sweater -> upper_clothes
    49: 1,  # sweatshirt -> upper_clothes
    50: 3,  # swimwear -> dress
    51: 1,  # t-shirt -> upper_clothes
    52: 5,  # tie -> accessories
    53: 2,  # tights -> lower_clothes
    54: 1,  # top -> upper_clothes
    55: 1,  # vest -> upper_clothes
    56: 5,  # wallet -> accessories
    57: 5,  # watch -> accessories
    58: 4,  # wedges -> shoes
}

assert set(CLASS_TO_SUPERCLASS.keys()) == set(range(59)), \
    "CLASS_TO_SUPERCLASS must cover exactly original class ids 0-58"
assert set(CLASS_TO_SUPERCLASS.values()) == set(range(NUM_SUPERCLASSES)), \
    "CLASS_TO_SUPERCLASS must map onto exactly superclass ids 0-8"

# Precompute a lookup table for fast vectorized remapping: LOOKUP[original_id] = superclass_id
_LOOKUP_TABLE = np.zeros(59, dtype=np.uint8)
for _orig_id, _super_id in CLASS_TO_SUPERCLASS.items():
    _LOOKUP_TABLE[_orig_id] = _super_id


def remap_mask(mask):
    mask = np.asarray(mask)
    if mask.max() > 58:
        raise ValueError(
            f"Mask contains value {mask.max()} outside expected range [0, 58] "
            "— is this already a remapped/superclass mask?"
        )
    return _LOOKUP_TABLE[mask]


def load_superclass_names(as_list=True):
    names = [SUPERCLASS_NAMES[i] for i in range(NUM_SUPERCLASSES)]
    return names if as_list else SUPERCLASS_NAMES


if __name__ == "__main__":
    test_mask = np.array([0, 1, 19, 41, 46, 58], dtype=np.uint8)
    remapped = remap_mask(test_mask)
    print("Original:", test_mask)
    print("Remapped:", remapped)
    print("Expected: [0 5 7 6 8 4]")
    assert list(remapped) == [0, 5, 7, 6, 8, 4]
    print("PASSED")