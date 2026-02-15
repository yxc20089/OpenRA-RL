"""Pydantic models for the OpenRA-RL environment.

Defines the Action, Observation, and State types used across
the OpenEnv client-server boundary.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


# ─── Action Types ─────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    """Available command types matching the protobuf ActionType enum."""

    NO_OP = "no_op"
    MOVE = "move"
    ATTACK_MOVE = "attack_move"
    ATTACK = "attack"
    STOP = "stop"
    HARVEST = "harvest"
    BUILD = "build"
    TRAIN = "train"
    DEPLOY = "deploy"
    SELL = "sell"
    REPAIR = "repair"
    PLACE_BUILDING = "place_building"
    CANCEL_PRODUCTION = "cancel_production"
    SET_RALLY_POINT = "set_rally_point"
    GUARD = "guard"
    SET_STANCE = "set_stance"
    ENTER_TRANSPORT = "enter_transport"
    UNLOAD = "unload"
    POWER_DOWN = "power_down"
    SET_PRIMARY = "set_primary"


class CommandModel(Action):
    """A single command to issue to the game engine."""

    action: ActionType = Field(..., description="Type of command to execute")
    actor_id: int = Field(default=0, description="Subject actor ID (for unit commands)")
    target_actor_id: int = Field(default=0, description="Target actor ID (for attack, etc.)")
    target_x: int = Field(default=0, description="Target cell X coordinate")
    target_y: int = Field(default=0, description="Target cell Y coordinate")
    item_type: str = Field(default="", description="Actor type for build/train commands")
    queued: bool = Field(default=False, description="Queue after current activity vs interrupt")


class OpenRAAction(Action):
    """Action sent from the agent to the OpenRA environment.

    Contains a list of commands to execute in a single game step.
    Multiple commands can be issued per step (e.g., move unit A and build unit B).
    """

    commands: List[CommandModel] = Field(
        default_factory=list, description="List of commands to execute this step"
    )


# ─── Observation Types ────────────────────────────────────────────────────────


class EconomyInfo(Action):
    """Player economic state."""

    cash: int = Field(default=0, description="Available cash")
    ore: int = Field(default=0, description="Raw ore in silos")
    power_provided: int = Field(default=0, description="Total power generation")
    power_drained: int = Field(default=0, description="Total power consumption")
    resource_capacity: int = Field(default=0, description="Maximum resource storage")
    harvester_count: int = Field(default=0, description="Number of active harvesters")


class MilitaryInfo(Action):
    """Player military statistics."""

    units_killed: int = Field(default=0, description="Enemy units destroyed")
    units_lost: int = Field(default=0, description="Own units lost")
    buildings_killed: int = Field(default=0, description="Enemy buildings destroyed")
    buildings_lost: int = Field(default=0, description="Own buildings lost")
    army_value: int = Field(default=0, description="Total value of active army")
    active_unit_count: int = Field(default=0, description="Number of active units")


class UnitInfoModel(Action):
    """Information about a single unit."""

    actor_id: int = Field(..., description="Unique actor ID")
    type: str = Field(..., description="Actor type (e.g., 'e1', '1tnk', 'harv')")
    pos_x: int = Field(default=0, description="World position X")
    pos_y: int = Field(default=0, description="World position Y")
    cell_x: int = Field(default=0, description="Cell position X")
    cell_y: int = Field(default=0, description="Cell position Y")
    hp_percent: float = Field(default=1.0, description="Health percentage 0.0-1.0")
    is_idle: bool = Field(default=True, description="Whether the unit is idle")
    current_activity: str = Field(default="", description="Current activity name")
    owner: str = Field(default="", description="Owner player internal name")
    can_attack: bool = Field(default=False, description="Whether the unit can attack")

    # Sprint 4: enriched unit data
    facing: int = Field(default=0, description="WAngle 0-1023 direction unit faces")
    experience_level: int = Field(default=0, description="Veterancy level (0=none)")
    stance: int = Field(default=0, description="0=HoldFire, 1=ReturnFire, 2=Defend, 3=AttackAnything")
    speed: int = Field(default=0, description="Base movement speed")
    attack_range: int = Field(default=0, description="Max attack range in WDist units")
    passenger_count: int = Field(default=-1, description="Cargo count (0 if transport empty, -1 if N/A)")
    is_building: bool = Field(default=False, description="False for units, helps distinguish in visible_enemies")


class BuildingInfoModel(Action):
    """Information about a single building."""

    actor_id: int = Field(..., description="Unique actor ID")
    type: str = Field(..., description="Actor type (e.g., 'powr', 'barr', 'weap')")
    pos_x: int = Field(default=0, description="World position X")
    pos_y: int = Field(default=0, description="World position Y")
    hp_percent: float = Field(default=1.0, description="Health percentage 0.0-1.0")
    owner: str = Field(default="", description="Owner player internal name")
    is_producing: bool = Field(default=False, description="Whether actively producing")
    production_progress: float = Field(default=0.0, description="Production progress 0.0-1.0")
    producing_item: str = Field(default="", description="Item currently being produced")
    is_powered: bool = Field(default=True, description="Whether powered")

    # Sprint 4: enriched building data
    is_repairing: bool = Field(default=False, description="Actively being repaired")
    sell_value: int = Field(default=0, description="Refund amount if sold")
    rally_x: int = Field(default=-1, description="Rally point cell X (-1 if none)")
    rally_y: int = Field(default=-1, description="Rally point cell Y (-1 if none)")
    power_amount: int = Field(default=0, description="Power provided (+) or consumed (-)")
    can_produce: List[str] = Field(default_factory=list, description="Items this building can produce")
    cell_x: int = Field(default=0, description="Cell position X")
    cell_y: int = Field(default=0, description="Cell position Y")


class ProductionInfoModel(Action):
    """Information about a production queue entry."""

    queue_type: str = Field(..., description="Queue type: Building, Infantry, Vehicle, Aircraft")
    item: str = Field(..., description="Actor type being produced")
    progress: float = Field(default=0.0, description="Progress 0.0-1.0")
    remaining_ticks: int = Field(default=0, description="Ticks until completion")
    remaining_cost: int = Field(default=0, description="Remaining cost")
    paused: bool = Field(default=False, description="Whether production is paused")


class MapInfoModel(Action):
    """Basic map information."""

    width: int = Field(default=0, description="Map width in cells")
    height: int = Field(default=0, description="Map height in cells")
    map_name: str = Field(default="", description="Map display name")


class OpenRAObservation(Observation):
    """Observation returned from the OpenRA environment each step.

    Contains structured game state data matching the protobuf GameObservation.
    """

    tick: int = Field(default=0, description="Current game tick")
    economy: EconomyInfo = Field(default_factory=EconomyInfo, description="Economic state")
    military: MilitaryInfo = Field(default_factory=MilitaryInfo, description="Military statistics")
    units: List[UnitInfoModel] = Field(default_factory=list, description="Own units")
    buildings: List[BuildingInfoModel] = Field(default_factory=list, description="Own buildings")
    production: List[ProductionInfoModel] = Field(default_factory=list, description="Active production queues")
    visible_enemies: List[UnitInfoModel] = Field(default_factory=list, description="Visible enemy units")
    visible_enemy_buildings: List[BuildingInfoModel] = Field(
        default_factory=list, description="Visible enemy buildings"
    )
    map_info: MapInfoModel = Field(default_factory=MapInfoModel, description="Map metadata")
    available_production: List[str] = Field(
        default_factory=list, description="Actor types available for production"
    )
    result: str = Field(default="", description="Game result: 'win', 'lose', 'draw', or ''")

    # Spatial map tensor (base64-encoded float32 array for JSON transport)
    spatial_map: str = Field(default="", description="Base64-encoded spatial tensor: H×W×C float32 array")
    spatial_channels: int = Field(default=0, description="Number of spatial channels")

    # Inherited from Observation:
    # done: bool = False
    # reward: float | None = None
    # metadata: Dict[str, Any] = {}


# ─── State ────────────────────────────────────────────────────────────────────


class OpenRAState(State):
    """Environment state tracking episode metadata.

    Extends the base State with OpenRA-specific fields.
    """

    game_tick: int = Field(default=0, description="Current game tick")
    map_name: str = Field(default="", description="Active map name")
    opponent_type: str = Field(default="bot_normal", description="Opponent type: bot_easy, bot_normal, bot_hard")

    # Inherited from State:
    # episode_id: Optional[str] = None
    # step_count: int = 0
