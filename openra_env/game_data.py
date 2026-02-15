"""Static Red Alert mod data for game knowledge tools.

Provides unit stats, building stats, tech tree, and faction information
extracted from OpenRA Red Alert mod rules. This gives an LLM agent the same
reference knowledge a human player would have from experience.
"""

from typing import Optional


# ─── Unit Data ────────────────────────────────────────────────────────────────

RA_UNITS: dict[str, dict] = {
    # Infantry
    "e1": {
        "name": "Rifle Infantry",
        "category": "infantry",
        "cost": 100,
        "hp": 5000,
        "speed": 56,
        "armor": "none",
        "side": "both",
        "prerequisites": ["barr|tent"],
        "description": "Basic infantry unit. Cheap and fast to produce.",
    },
    "e2": {
        "name": "Grenadier",
        "category": "infantry",
        "cost": 160,
        "hp": 5000,
        "speed": 56,
        "armor": "none",
        "side": "both",
        "prerequisites": ["barr|tent"],
        "description": "Anti-structure infantry. Grenades deal area damage.",
    },
    "e3": {
        "name": "Rocket Soldier",
        "category": "infantry",
        "cost": 300,
        "hp": 4500,
        "speed": 56,
        "armor": "none",
        "side": "both",
        "prerequisites": ["barr|tent"],
        "description": "Anti-armor and anti-air infantry.",
    },
    "e4": {
        "name": "Flamethrower",
        "category": "infantry",
        "cost": 300,
        "hp": 4000,
        "speed": 56,
        "armor": "none",
        "side": "soviet",
        "prerequisites": ["barr", "ftur"],
        "description": "Short-range anti-infantry/structure. Soviet only.",
    },
    "e6": {
        "name": "Engineer",
        "category": "infantry",
        "cost": 500,
        "hp": 4000,
        "speed": 56,
        "armor": "none",
        "side": "both",
        "prerequisites": ["barr|tent"],
        "description": "Captures enemy buildings. Cannot attack.",
    },
    "e7": {
        "name": "Tanya",
        "category": "infantry",
        "cost": 1200,
        "hp": 10000,
        "speed": 68,
        "armor": "none",
        "side": "allied",
        "prerequisites": ["tent", "atek"],
        "build_limit": 1,
        "description": "Elite commando. Destroys buildings with C4, kills infantry instantly. Allied only.",
    },
    "medi": {
        "name": "Medic",
        "category": "infantry",
        "cost": 200,
        "hp": 6000,
        "speed": 49,
        "armor": "none",
        "side": "allied",
        "prerequisites": ["tent"],
        "description": "Heals nearby infantry. Cannot attack.",
    },
    "mech": {
        "name": "Mechanic",
        "category": "infantry",
        "cost": 500,
        "hp": 8000,
        "speed": 49,
        "armor": "none",
        "side": "allied",
        "prerequisites": ["tent", "fix"],
        "description": "Repairs nearby vehicles. Cannot attack.",
    },
    "spy": {
        "name": "Spy",
        "category": "infantry",
        "cost": 500,
        "hp": 2500,
        "speed": 56,
        "armor": "none",
        "side": "allied",
        "prerequisites": ["tent", "dome"],
        "description": "Disguises as enemy infantry. Infiltrates buildings for bonuses.",
    },
    "thf": {
        "name": "Thief",
        "category": "infantry",
        "cost": 500,
        "hp": 5000,
        "speed": 68,
        "armor": "none",
        "side": "allied",
        "prerequisites": ["tent", "dome"],
        "description": "Steals credits from enemy refineries.",
    },
    "shok": {
        "name": "Shock Trooper",
        "category": "infantry",
        "cost": 350,
        "hp": 5000,
        "speed": 49,
        "armor": "none",
        "side": "soviet",
        "prerequisites": ["barr", "stek", "tsla"],
        "description": "Tesla infantry. High damage vs all targets. Soviet only.",
    },
    "dog": {
        "name": "Attack Dog",
        "category": "infantry",
        "cost": 200,
        "hp": 2000,
        "speed": 99,
        "armor": "none",
        "side": "soviet",
        "prerequisites": ["kenn"],
        "description": "Fast anti-infantry unit. Kills spies. Soviet only.",
    },

    # Vehicles
    "1tnk": {
        "name": "Light Tank",
        "category": "vehicle",
        "cost": 700,
        "hp": 23000,
        "speed": 113,
        "armor": "heavy",
        "side": "allied",
        "prerequisites": ["weap"],
        "description": "Fast medium tank. Good all-around. Allied only.",
    },
    "2tnk": {
        "name": "Medium Tank",
        "category": "vehicle",
        "cost": 800,
        "hp": 30000,
        "speed": 72,
        "armor": "heavy",
        "side": "allied",
        "prerequisites": ["weap"],
        "description": "Main battle tank. Balanced stats. Allied only.",
    },
    "3tnk": {
        "name": "Heavy Tank",
        "category": "vehicle",
        "cost": 950,
        "hp": 46000,
        "speed": 64,
        "armor": "heavy",
        "side": "soviet",
        "prerequisites": ["weap"],
        "description": "Powerful main battle tank. Dual cannons. Soviet only.",
    },
    "4tnk": {
        "name": "Mammoth Tank",
        "category": "vehicle",
        "cost": 1700,
        "hp": 60000,
        "speed": 43,
        "armor": "heavy",
        "side": "soviet",
        "prerequisites": ["weap", "stek"],
        "description": "Heaviest tank. Dual cannons + missiles. Self-healing. Soviet only.",
    },
    "v2rl": {
        "name": "V2 Rocket Launcher",
        "category": "vehicle",
        "cost": 700,
        "hp": 15000,
        "speed": 72,
        "armor": "light",
        "side": "soviet",
        "prerequisites": ["weap", "dome"],
        "description": "Long-range artillery. High damage, inaccurate. Soviet only.",
    },
    "jeep": {
        "name": "Ranger",
        "category": "vehicle",
        "cost": 600,
        "hp": 15000,
        "speed": 164,
        "armor": "light",
        "side": "allied",
        "prerequisites": ["weap"],
        "description": "Fast scout vehicle with machine gun. Allied only.",
    },
    "apc": {
        "name": "APC",
        "category": "vehicle",
        "cost": 800,
        "hp": 20000,
        "speed": 128,
        "armor": "heavy",
        "side": "both",
        "prerequisites": ["weap"],
        "description": "Armored troop transport. Carries 5 infantry.",
    },
    "arty": {
        "name": "Artillery",
        "category": "vehicle",
        "cost": 600,
        "hp": 7500,
        "speed": 54,
        "armor": "light",
        "side": "allied",
        "prerequisites": ["weap", "dome"],
        "description": "Long-range siege weapon. Allied only.",
    },
    "harv": {
        "name": "Ore Truck",
        "category": "vehicle",
        "cost": 1400,
        "hp": 60000,
        "speed": 72,
        "armor": "heavy",
        "side": "both",
        "prerequisites": ["proc"],
        "description": "Harvests ore and delivers to refinery. Free with refinery.",
    },
    "mcv": {
        "name": "MCV",
        "category": "vehicle",
        "cost": 2500,
        "hp": 60000,
        "speed": 60,
        "armor": "heavy",
        "side": "both",
        "prerequisites": ["weap", "fix"],
        "description": "Deploys into Construction Yard. Mobile base.",
    },
    "ftrk": {
        "name": "Flak Truck",
        "category": "vehicle",
        "cost": 500,
        "hp": 15000,
        "speed": 113,
        "armor": "light",
        "side": "soviet",
        "prerequisites": ["weap"],
        "description": "Mobile anti-air unit. Soviet only.",
    },
    "mnly": {
        "name": "Minelayer",
        "category": "vehicle",
        "cost": 800,
        "hp": 30000,
        "speed": 113,
        "armor": "heavy",
        "side": "both",
        "prerequisites": ["weap", "fix"],
        "description": "Lays anti-tank mines.",
    },
    "ttnk": {
        "name": "Tesla Tank",
        "category": "vehicle",
        "cost": 1500,
        "hp": 30000,
        "speed": 92,
        "armor": "heavy",
        "side": "soviet",
        "prerequisites": ["weap", "stek", "tsla"],
        "description": "Tesla weapon on tracks. Effective vs all targets. Soviet only.",
    },
    "ctnk": {
        "name": "Chrono Tank",
        "category": "vehicle",
        "cost": 1200,
        "hp": 20000,
        "speed": 86,
        "armor": "light",
        "side": "allied",
        "prerequisites": ["weap", "atek"],
        "description": "Teleporting tank. Hit and run tactics. Allied only.",
    },
    "stnk": {
        "name": "Phase Transport",
        "category": "vehicle",
        "cost": 900,
        "hp": 11000,
        "speed": 128,
        "armor": "light",
        "side": "allied",
        "prerequisites": ["weap", "atek"],
        "description": "Cloaked APC. Invisible when not firing. Allied only.",
    },
    "qtnk": {
        "name": "MAD Tank",
        "category": "vehicle",
        "cost": 2300,
        "hp": 22000,
        "speed": 46,
        "armor": "heavy",
        "side": "soviet",
        "prerequisites": ["weap", "stek"],
        "description": "Deploys seismic charge, destroying self and nearby vehicles. Soviet only.",
    },
    "dtrk": {
        "name": "Demolition Truck",
        "category": "vehicle",
        "cost": 1500,
        "hp": 11000,
        "speed": 113,
        "armor": "light",
        "side": "soviet",
        "prerequisites": ["weap", "stek"],
        "description": "Suicide vehicle. Massive area nuclear explosion on death. Soviet only.",
    },
    "mgg": {
        "name": "Mobile Gap Generator",
        "category": "vehicle",
        "cost": 600,
        "hp": 11000,
        "speed": 72,
        "armor": "light",
        "side": "allied",
        "prerequisites": ["weap", "atek"],
        "description": "Creates mobile shroud area. Allied only.",
    },
    "mrj": {
        "name": "Mobile Radar Jammer",
        "category": "vehicle",
        "cost": 600,
        "hp": 11000,
        "speed": 68,
        "armor": "light",
        "side": "allied",
        "prerequisites": ["weap", "atek"],
        "description": "Jams enemy radar in area. Allied only.",
    },
    "truk": {
        "name": "Supply Truck",
        "category": "vehicle",
        "cost": 800,
        "hp": 11000,
        "speed": 113,
        "armor": "light",
        "side": "both",
        "prerequisites": ["weap"],
        "description": "Delivers cash when reaching allied structures.",
    },

    # Aircraft
    "heli": {
        "name": "Longbow",
        "category": "aircraft",
        "cost": 1200,
        "hp": 12000,
        "speed": 149,
        "armor": "light",
        "side": "allied",
        "prerequisites": ["hpad"],
        "description": "Anti-armor helicopter with missiles. Allied only.",
    },
    "hind": {
        "name": "Hind",
        "category": "aircraft",
        "cost": 1200,
        "hp": 12000,
        "speed": 112,
        "armor": "light",
        "side": "soviet",
        "prerequisites": ["afld"],
        "description": "Anti-ground attack helicopter. Soviet only.",
    },
    "mh60": {
        "name": "Black Hawk",
        "category": "aircraft",
        "cost": 1200,
        "hp": 12000,
        "speed": 112,
        "armor": "light",
        "side": "allied",
        "prerequisites": ["hpad"],
        "description": "Transport/attack helicopter. Allied only.",
    },
    "tran": {
        "name": "Chinook",
        "category": "aircraft",
        "cost": 900,
        "hp": 14000,
        "speed": 128,
        "armor": "light",
        "side": "both",
        "prerequisites": ["hpad|afld"],
        "description": "Transport helicopter. Carries 5 infantry.",
    },
    "yak": {
        "name": "Yak",
        "category": "aircraft",
        "cost": 800,
        "hp": 6000,
        "speed": 178,
        "armor": "light",
        "side": "soviet",
        "prerequisites": ["afld"],
        "description": "Fast anti-infantry attack plane. Soviet only.",
    },
    "mig": {
        "name": "MiG",
        "category": "aircraft",
        "cost": 2000,
        "hp": 8000,
        "speed": 223,
        "armor": "light",
        "side": "soviet",
        "prerequisites": ["afld", "stek"],
        "description": "Anti-structure/armor attack plane with missiles. Soviet only.",
    },

    # Ships
    "ss": {
        "name": "Submarine",
        "category": "ship",
        "cost": 950,
        "hp": 25000,
        "speed": 78,
        "armor": "light",
        "side": "soviet",
        "prerequisites": ["spen"],
        "description": "Invisible anti-ship unit. Soviet only.",
    },
    "dd": {
        "name": "Destroyer",
        "category": "ship",
        "cost": 1000,
        "hp": 40000,
        "speed": 92,
        "armor": "heavy",
        "side": "allied",
        "prerequisites": ["syrd", "dome"],
        "description": "Multi-role warship. Anti-sub, anti-air, anti-surface. Allied only.",
    },
    "ca": {
        "name": "Cruiser",
        "category": "ship",
        "cost": 2000,
        "hp": 80000,
        "speed": 44,
        "armor": "heavy",
        "side": "allied",
        "prerequisites": ["syrd", "atek"],
        "description": "Heavy bombardment ship. Long range. Allied only.",
    },
    "pt": {
        "name": "Gunboat",
        "category": "ship",
        "cost": 500,
        "hp": 20000,
        "speed": 142,
        "armor": "heavy",
        "side": "both",
        "prerequisites": ["syrd|spen"],
        "description": "Fast patrol boat.",
    },
    "lst": {
        "name": "Transport",
        "category": "ship",
        "cost": 700,
        "hp": 40000,
        "speed": 115,
        "armor": "heavy",
        "side": "both",
        "prerequisites": ["syrd|spen"],
        "description": "Naval transport. Carries vehicles and infantry.",
    },
    "msub": {
        "name": "Missile Submarine",
        "category": "ship",
        "cost": 2000,
        "hp": 40000,
        "speed": 44,
        "armor": "light",
        "side": "soviet",
        "prerequisites": ["spen", "stek"],
        "description": "Long-range missile submarine. Soviet only.",
    },
}


# ─── Building Data ────────────────────────────────────────────────────────────

RA_BUILDINGS: dict[str, dict] = {
    "fact": {
        "name": "Construction Yard",
        "cost": 2500,
        "hp": 150000,
        "power": 0,
        "side": "both",
        "prerequisites": [],
        "produces": ["Building", "Defense"],
        "description": "Primary base structure. Required to build other structures.",
    },
    "powr": {
        "name": "Power Plant",
        "cost": 300,
        "hp": 40000,
        "power": 100,
        "side": "both",
        "prerequisites": [],
        "produces": [],
        "description": "Basic power supply. Most structures need power to function.",
    },
    "apwr": {
        "name": "Advanced Power Plant",
        "cost": 500,
        "hp": 70000,
        "power": 200,
        "side": "both",
        "prerequisites": ["dome"],
        "produces": [],
        "description": "Double power output. Requires radar dome tech.",
    },
    "barr": {
        "name": "Soviet Barracks",
        "cost": 500,
        "hp": 60000,
        "power": -20,
        "side": "soviet",
        "prerequisites": ["powr"],
        "produces": ["Infantry"],
        "description": "Soviet infantry production. Required for all Soviet infantry.",
    },
    "tent": {
        "name": "Allied Barracks",
        "cost": 500,
        "hp": 60000,
        "power": -20,
        "side": "allied",
        "prerequisites": ["powr"],
        "produces": ["Infantry"],
        "description": "Allied infantry production. Required for all Allied infantry.",
    },
    "proc": {
        "name": "Ore Refinery",
        "cost": 2000,
        "hp": 90000,
        "power": -30,
        "side": "both",
        "prerequisites": ["powr"],
        "produces": [],
        "description": "Processes ore into credits. Comes with a free Ore Truck.",
    },
    "weap": {
        "name": "War Factory",
        "cost": 2000,
        "hp": 150000,
        "power": -30,
        "side": "both",
        "prerequisites": ["proc"],
        "produces": ["Vehicle"],
        "description": "Vehicle production facility. Required for all vehicles.",
    },
    "dome": {
        "name": "Radar Dome",
        "cost": 1000,
        "hp": 100000,
        "power": -40,
        "side": "both",
        "prerequisites": ["proc"],
        "produces": [],
        "description": "Provides minimap radar. Unlocks advanced tech.",
    },
    "fix": {
        "name": "Service Depot",
        "cost": 1200,
        "hp": 80000,
        "power": -30,
        "side": "both",
        "prerequisites": ["weap"],
        "produces": [],
        "description": "Repairs vehicles. Unlocks MCV and Minelayer.",
    },
    "atek": {
        "name": "Allied Tech Center",
        "cost": 1500,
        "hp": 60000,
        "power": -200,
        "side": "allied",
        "prerequisites": ["dome", "weap"],
        "produces": [],
        "description": "Unlocks advanced Allied units. GPS satellite.",
    },
    "stek": {
        "name": "Soviet Tech Center",
        "cost": 1500,
        "hp": 80000,
        "power": -100,
        "side": "soviet",
        "prerequisites": ["dome", "weap"],
        "produces": [],
        "description": "Unlocks advanced Soviet units.",
    },
    "hpad": {
        "name": "Helipad",
        "cost": 1500,
        "hp": 80000,
        "power": -10,
        "side": "allied",
        "prerequisites": ["dome"],
        "produces": ["Aircraft"],
        "description": "Allied aircraft production. Rearming pad.",
    },
    "afld": {
        "name": "Airfield",
        "cost": 1000,
        "hp": 100000,
        "power": -20,
        "side": "soviet",
        "prerequisites": ["dome"],
        "produces": ["Aircraft"],
        "description": "Soviet aircraft production. Rearming strip.",
    },
    "spen": {
        "name": "Sub Pen",
        "cost": 650,
        "hp": 100000,
        "power": -20,
        "side": "soviet",
        "prerequisites": ["powr"],
        "produces": ["Ship"],
        "description": "Soviet naval production. Repairs ships.",
    },
    "syrd": {
        "name": "Naval Yard",
        "cost": 650,
        "hp": 100000,
        "power": -20,
        "side": "allied",
        "prerequisites": ["powr"],
        "produces": ["Ship"],
        "description": "Allied naval production. Repairs ships.",
    },
    "silo": {
        "name": "Ore Silo",
        "cost": 150,
        "hp": 30000,
        "power": -10,
        "side": "both",
        "prerequisites": ["proc"],
        "produces": [],
        "description": "Additional ore storage capacity.",
    },
    "kenn": {
        "name": "Kennel",
        "cost": 200,
        "hp": 30000,
        "power": -10,
        "side": "soviet",
        "prerequisites": ["powr"],
        "produces": ["Infantry"],
        "description": "Produces attack dogs. Soviet only.",
    },

    # Defenses
    "pbox": {
        "name": "Pillbox",
        "cost": 400,
        "hp": 40000,
        "power": 0,
        "side": "allied",
        "prerequisites": ["tent"],
        "produces": [],
        "description": "Anti-infantry defense turret. Allied only.",
    },
    "hbox": {
        "name": "Camo Pillbox",
        "cost": 600,
        "hp": 40000,
        "power": 0,
        "side": "allied",
        "prerequisites": ["tent"],
        "produces": [],
        "description": "Hidden anti-infantry defense. Allied only.",
    },
    "gun": {
        "name": "Turret",
        "cost": 600,
        "hp": 40000,
        "power": -20,
        "side": "allied",
        "prerequisites": ["weap"],
        "produces": [],
        "description": "Anti-armor defense turret. Allied only.",
    },
    "ftur": {
        "name": "Flame Tower",
        "cost": 600,
        "hp": 40000,
        "power": -20,
        "side": "soviet",
        "prerequisites": ["barr"],
        "produces": [],
        "description": "Short-range anti-infantry defense. Soviet only.",
    },
    "tsla": {
        "name": "Tesla Coil",
        "cost": 1500,
        "hp": 40000,
        "power": -75,
        "side": "soviet",
        "prerequisites": ["weap"],
        "produces": [],
        "description": "Powerful anti-ground defense. High power cost. Soviet only.",
    },
    "agun": {
        "name": "AA Gun",
        "cost": 600,
        "hp": 40000,
        "power": -50,
        "side": "allied",
        "prerequisites": ["dome"],
        "produces": [],
        "description": "Anti-air defense turret. Allied only.",
    },
    "sam": {
        "name": "SAM Site",
        "cost": 750,
        "hp": 40000,
        "power": -20,
        "side": "soviet",
        "prerequisites": ["dome"],
        "produces": [],
        "description": "Anti-air missile defense. Soviet only.",
    },
    "gap": {
        "name": "Gap Generator",
        "cost": 500,
        "hp": 50000,
        "power": -60,
        "side": "allied",
        "prerequisites": ["atek"],
        "produces": [],
        "description": "Creates shroud area over your base. Allied only.",
    },

    # Superweapons
    "iron": {
        "name": "Iron Curtain",
        "cost": 2800,
        "hp": 100000,
        "power": -200,
        "side": "soviet",
        "prerequisites": ["stek"],
        "produces": [],
        "build_limit": 1,
        "description": "Superweapon: Makes one unit/building invulnerable temporarily.",
    },
    "pdox": {
        "name": "Chronosphere",
        "cost": 2800,
        "hp": 100000,
        "power": -200,
        "side": "allied",
        "prerequisites": ["atek"],
        "produces": [],
        "build_limit": 1,
        "description": "Superweapon: Teleports units across the map.",
    },
    "mslo": {
        "name": "Missile Silo",
        "cost": 2500,
        "hp": 100000,
        "power": -150,
        "side": "soviet",
        "prerequisites": ["stek"],
        "produces": [],
        "build_limit": 1,
        "description": "Superweapon: Launches nuclear missile at target location.",
    },
}


# ─── Tech Tree ────────────────────────────────────────────────────────────────

RA_TECH_TREE: dict[str, list[str]] = {
    "soviet": [
        "powr",     # Power Plant (base)
        "barr",     # Barracks → infantry (requires powr)
        "kenn",     # Kennel → dogs (requires powr)
        "proc",     # Ore Refinery (requires powr)
        "weap",     # War Factory (requires proc)
        "spen",     # Sub Pen (requires powr, needs water)
        "dome",     # Radar Dome (requires proc)
        "fix",      # Service Depot (requires weap)
        "afld",     # Airfield (requires dome)
        "stek",     # Tech Center (requires dome + weap)
        "tsla",     # Tesla Coil (requires weap)
        "sam",      # SAM Site (requires dome)
        "ftur",     # Flame Tower (requires barr)
        "iron",     # Iron Curtain (requires stek)
        "mslo",     # Missile Silo (requires stek)
    ],
    "allied": [
        "powr",     # Power Plant (base)
        "tent",     # Barracks → infantry (requires powr)
        "proc",     # Ore Refinery (requires powr)
        "weap",     # War Factory (requires proc)
        "syrd",     # Naval Yard (requires powr, needs water)
        "dome",     # Radar Dome (requires proc)
        "fix",      # Service Depot (requires weap)
        "hpad",     # Helipad (requires dome)
        "atek",     # Tech Center (requires dome + weap)
        "gun",      # Turret (requires weap)
        "pbox",     # Pillbox (requires tent)
        "agun",     # AA Gun (requires dome)
        "gap",      # Gap Generator (requires atek)
        "pdox",     # Chronosphere (requires atek)
    ],
}


# ─── Faction Data ─────────────────────────────────────────────────────────────

RA_FACTIONS: dict[str, dict] = {
    "england": {
        "side": "allied",
        "display_name": "England",
        "unique_units": [],
        "description": "Standard Allied faction.",
    },
    "france": {
        "side": "allied",
        "display_name": "France",
        "unique_units": ["stnk"],
        "description": "Allied faction with Phase Transport (cloaked APC).",
    },
    "germany": {
        "side": "allied",
        "display_name": "Germany",
        "unique_units": ["ctnk"],
        "description": "Allied faction with Chrono Tank (teleporting tank).",
    },
    "russia": {
        "side": "soviet",
        "display_name": "Russia",
        "unique_units": ["ttnk"],
        "description": "Soviet faction with Tesla Tank.",
    },
    "ukraine": {
        "side": "soviet",
        "display_name": "Ukraine",
        "unique_units": ["dtrk"],
        "description": "Soviet faction with Demolition Truck (nuclear suicide vehicle).",
    },
}


# ─── Query Functions ──────────────────────────────────────────────────────────


def get_unit_stats(unit_type: str) -> Optional[dict]:
    """Get stats for a unit type. Returns None if not found."""
    return RA_UNITS.get(unit_type.lower())


def get_building_stats(building_type: str) -> Optional[dict]:
    """Get stats for a building type. Returns None if not found."""
    return RA_BUILDINGS.get(building_type.lower())


def get_tech_tree(faction: Optional[str] = None) -> dict:
    """Get the tech tree build order.

    Args:
        faction: Faction name (e.g., 'russia') or side ('allied', 'soviet').
                If None, returns both sides.
    """
    if faction is None:
        return RA_TECH_TREE

    # Map faction to side
    side = faction.lower()
    if side in RA_FACTIONS:
        side = RA_FACTIONS[side]["side"]

    if side in RA_TECH_TREE:
        return {side: RA_TECH_TREE[side]}

    return {}


def get_faction_info(faction: str) -> Optional[dict]:
    """Get faction info including available units and buildings."""
    faction = faction.lower()
    info = RA_FACTIONS.get(faction)
    if info is None:
        return None

    side = info["side"]

    # Collect units available to this faction
    available_units = []
    for unit_type, data in RA_UNITS.items():
        unit_side = data.get("side", "")
        if unit_side == "both" or unit_side == side:
            available_units.append(unit_type)

    # Add faction-unique units
    for u in info.get("unique_units", []):
        if u not in available_units and u in RA_UNITS:
            available_units.append(u)

    # Collect buildings
    available_buildings = []
    for bldg_type, data in RA_BUILDINGS.items():
        bldg_side = data.get("side", "")
        if bldg_side == "both" or bldg_side == side:
            available_buildings.append(bldg_type)

    return {
        **info,
        "faction": faction,
        "available_units": sorted(available_units),
        "available_buildings": sorted(available_buildings),
    }


def get_all_unit_types() -> list[str]:
    """Get all available unit type names."""
    return sorted(RA_UNITS.keys())


def get_all_building_types() -> list[str]:
    """Get all available building type names."""
    return sorted(RA_BUILDINGS.keys())
