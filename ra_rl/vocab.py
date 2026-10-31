"""Fixed vocabularies for the network (Red Alert rules, OpenRA/mods/ra/rules).

Part of the feature contract: changing an entry changes every input/output index,
so it is a new feature version (FEATURE_VERSION in ra_rl.features).
"""

BUILDINGS = (
    "fact", "powr", "apwr", "proc", "silo", "tent", "barr", "kenn", "weap", "fix", "dome", "hpad",
    "afld", "atek", "stek", "spen", "syrd", "gap", "iron", "pdox", "mslo",
    "pbox", "hbox", "gun", "ftur", "tsla", "agun", "sam", "sbag", "fenc", "brik",
)
INFANTRY = ("dog", "e1", "e2", "e3", "e4", "e6", "spy", "e7", "medi", "mech", "thf", "shok")
VEHICLES = (
    "harv", "mcv", "jeep", "apc", "1tnk", "2tnk", "3tnk", "4tnk", "arty", "v2rl", "ftrk", "mnly",
    "truk", "mgg", "mrj", "ttnk", "dtrk", "ctnk", "qtnk", "stnk",
)
AIRCRAFT = ("heli", "hind", "mh60", "tran", "yak", "mig")
SHIPS = ("ss", "msub", "dd", "ca", "lst", "pt")
UNITS = INFANTRY + VEHICLES + AIRCRAFT + SHIPS

# Everything a production queue can offer: the output vocabulary of the item head.
ITEMS = BUILDINGS + UNITS

QUEUES = ("Building", "Defense", "Infantry", "Vehicle", "Aircraft", "Ship")
TARGET_KINDS = ("enemy_spawn", "enemy_building", "unexplored", "stale")

BUILDING_INDEX = {t: i for i, t in enumerate(BUILDINGS)}
UNIT_INDEX = {t: i for i, t in enumerate(UNITS)}
ITEM_INDEX = {t: i for i, t in enumerate(ITEMS)}
QUEUE_INDEX = {q: i for i, q in enumerate(QUEUES)}
KIND_INDEX = {k: i for i, k in enumerate(TARGET_KINDS)}

# Valued.Cost from the rules.
COST = {
    "mslo": 2500, "gap": 800, "spen": 800, "syrd": 1000, "iron": 2000, "pdox": 1500, "tsla": 1200,
    "agun": 800, "dome": 1500, "pbox": 600, "hbox": 750, "gun": 800, "ftur": 600, "sam": 700,
    "atek": 1500, "weap": 2000, "fact": 2000, "proc": 1400, "silo": 150, "hpad": 500, "afld": 500,
    "powr": 300, "apwr": 500, "stek": 1500, "barr": 500, "kenn": 200, "tent": 500, "fix": 1200,
    "sbag": 30, "fenc": 30, "brik": 200,
    "dog": 200, "e1": 100, "e2": 150, "e3": 300, "e4": 300, "e6": 400, "spy": 500, "e7": 1800,
    "medi": 200, "mech": 500, "thf": 500, "shok": 350,
    "v2rl": 900, "1tnk": 700, "2tnk": 850, "3tnk": 1150, "4tnk": 2000, "arty": 850, "harv": 1100,
    "mcv": 2000, "jeep": 500, "apc": 850, "mnly": 800, "truk": 500, "mgg": 1000, "mrj": 1000,
    "ttnk": 1350, "ftrk": 600, "dtrk": 2500, "ctnk": 1350, "qtnk": 2000, "stnk": 1000,
    "mig": 2000, "yak": 1350, "tran": 900, "heli": 2000, "hind": 1500, "mh60": 1500,
    "ss": 950, "msub": 2000, "dd": 1000, "ca": 2400, "lst": 500, "pt": 500,
}

# Units that do not fight (left out of army values).
NON_COMBAT = frozenset({"harv", "mcv", "truk", "mnly", "tran", "lst", "e6", "spy", "thf", "medi", "mech"})


def base_type(actor_type: str) -> str:
    """Faction variants share the base type (e.g. "spy.england" -> "spy")."""
    return actor_type.split(".", 1)[0]


def combat_value(actor_type: str) -> int:
    t = base_type(actor_type)
    return 0 if t in NON_COMBAT else COST.get(t, 0)
