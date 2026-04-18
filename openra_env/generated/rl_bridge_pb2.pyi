from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NO_OP: _ClassVar[ActionType]
    MOVE: _ClassVar[ActionType]
    ATTACK_MOVE: _ClassVar[ActionType]
    ATTACK: _ClassVar[ActionType]
    STOP: _ClassVar[ActionType]
    HARVEST: _ClassVar[ActionType]
    BUILD: _ClassVar[ActionType]
    TRAIN: _ClassVar[ActionType]
    DEPLOY: _ClassVar[ActionType]
    SELL: _ClassVar[ActionType]
    REPAIR: _ClassVar[ActionType]
    PLACE_BUILDING: _ClassVar[ActionType]
    CANCEL_PRODUCTION: _ClassVar[ActionType]
    SET_RALLY_POINT: _ClassVar[ActionType]
    GUARD: _ClassVar[ActionType]
    SET_STANCE: _ClassVar[ActionType]
    ENTER_TRANSPORT: _ClassVar[ActionType]
    UNLOAD: _ClassVar[ActionType]
    POWER_DOWN: _ClassVar[ActionType]
    SET_PRIMARY: _ClassVar[ActionType]
    SURRENDER: _ClassVar[ActionType]
    FAST_ADVANCE: _ClassVar[ActionType]
    PATROL: _ClassVar[ActionType]
NO_OP: ActionType
MOVE: ActionType
ATTACK_MOVE: ActionType
ATTACK: ActionType
STOP: ActionType
HARVEST: ActionType
BUILD: ActionType
TRAIN: ActionType
DEPLOY: ActionType
SELL: ActionType
REPAIR: ActionType
PLACE_BUILDING: ActionType
CANCEL_PRODUCTION: ActionType
SET_RALLY_POINT: ActionType
GUARD: ActionType
SET_STANCE: ActionType
ENTER_TRANSPORT: ActionType
UNLOAD: ActionType
POWER_DOWN: ActionType
SET_PRIMARY: ActionType
SURRENDER: ActionType
FAST_ADVANCE: ActionType
PATROL: ActionType

class GameObservation(_message.Message):
    __slots__ = ("tick", "episode_id", "economy", "military", "units", "buildings", "production", "visible_enemies", "map_info", "spatial_map", "spatial_channels", "done", "reward", "result", "available_production", "visible_enemy_buildings", "explored_percent", "interrupted", "interrupt_reason", "actual_ticks_advanced", "kill_events")
    TICK_FIELD_NUMBER: _ClassVar[int]
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    ECONOMY_FIELD_NUMBER: _ClassVar[int]
    MILITARY_FIELD_NUMBER: _ClassVar[int]
    UNITS_FIELD_NUMBER: _ClassVar[int]
    BUILDINGS_FIELD_NUMBER: _ClassVar[int]
    PRODUCTION_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_ENEMIES_FIELD_NUMBER: _ClassVar[int]
    MAP_INFO_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_MAP_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_PRODUCTION_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_ENEMY_BUILDINGS_FIELD_NUMBER: _ClassVar[int]
    EXPLORED_PERCENT_FIELD_NUMBER: _ClassVar[int]
    INTERRUPTED_FIELD_NUMBER: _ClassVar[int]
    INTERRUPT_REASON_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_TICKS_ADVANCED_FIELD_NUMBER: _ClassVar[int]
    KILL_EVENTS_FIELD_NUMBER: _ClassVar[int]
    tick: int
    episode_id: str
    economy: RlEconomy
    military: RlMilitary
    units: _containers.RepeatedCompositeFieldContainer[RlUnitInfo]
    buildings: _containers.RepeatedCompositeFieldContainer[RlBuildingInfo]
    production: _containers.RepeatedCompositeFieldContainer[RlProductionInfo]
    visible_enemies: _containers.RepeatedCompositeFieldContainer[RlUnitInfo]
    map_info: RlMapInfo
    spatial_map: bytes
    spatial_channels: int
    done: bool
    reward: float
    result: str
    available_production: _containers.RepeatedScalarFieldContainer[str]
    visible_enemy_buildings: _containers.RepeatedCompositeFieldContainer[RlBuildingInfo]
    explored_percent: float
    interrupted: bool
    interrupt_reason: str
    actual_ticks_advanced: int
    kill_events: _containers.RepeatedCompositeFieldContainer[RlKillEvent]
    def __init__(self, tick: _Optional[int] = ..., episode_id: _Optional[str] = ..., economy: _Optional[_Union[RlEconomy, _Mapping]] = ..., military: _Optional[_Union[RlMilitary, _Mapping]] = ..., units: _Optional[_Iterable[_Union[RlUnitInfo, _Mapping]]] = ..., buildings: _Optional[_Iterable[_Union[RlBuildingInfo, _Mapping]]] = ..., production: _Optional[_Iterable[_Union[RlProductionInfo, _Mapping]]] = ..., visible_enemies: _Optional[_Iterable[_Union[RlUnitInfo, _Mapping]]] = ..., map_info: _Optional[_Union[RlMapInfo, _Mapping]] = ..., spatial_map: _Optional[bytes] = ..., spatial_channels: _Optional[int] = ..., done: bool = ..., reward: _Optional[float] = ..., result: _Optional[str] = ..., available_production: _Optional[_Iterable[str]] = ..., visible_enemy_buildings: _Optional[_Iterable[_Union[RlBuildingInfo, _Mapping]]] = ..., explored_percent: _Optional[float] = ..., interrupted: bool = ..., interrupt_reason: _Optional[str] = ..., actual_ticks_advanced: _Optional[int] = ..., kill_events: _Optional[_Iterable[_Union[RlKillEvent, _Mapping]]] = ...) -> None: ...

class RlKillEvent(_message.Message):
    __slots__ = ("tick", "victim_actor_id", "victim_type", "victim_cell_x", "victim_cell_y", "attacker_actor_id", "attacker_type", "attacker_cell_x", "attacker_cell_y", "victim_is_own", "attacker_is_own", "victim_is_building")
    TICK_FIELD_NUMBER: _ClassVar[int]
    VICTIM_ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    VICTIM_TYPE_FIELD_NUMBER: _ClassVar[int]
    VICTIM_CELL_X_FIELD_NUMBER: _ClassVar[int]
    VICTIM_CELL_Y_FIELD_NUMBER: _ClassVar[int]
    ATTACKER_ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    ATTACKER_TYPE_FIELD_NUMBER: _ClassVar[int]
    ATTACKER_CELL_X_FIELD_NUMBER: _ClassVar[int]
    ATTACKER_CELL_Y_FIELD_NUMBER: _ClassVar[int]
    VICTIM_IS_OWN_FIELD_NUMBER: _ClassVar[int]
    ATTACKER_IS_OWN_FIELD_NUMBER: _ClassVar[int]
    VICTIM_IS_BUILDING_FIELD_NUMBER: _ClassVar[int]
    tick: int
    victim_actor_id: int
    victim_type: str
    victim_cell_x: int
    victim_cell_y: int
    attacker_actor_id: int
    attacker_type: str
    attacker_cell_x: int
    attacker_cell_y: int
    victim_is_own: bool
    attacker_is_own: bool
    victim_is_building: bool
    def __init__(self, tick: _Optional[int] = ..., victim_actor_id: _Optional[int] = ..., victim_type: _Optional[str] = ..., victim_cell_x: _Optional[int] = ..., victim_cell_y: _Optional[int] = ..., attacker_actor_id: _Optional[int] = ..., attacker_type: _Optional[str] = ..., attacker_cell_x: _Optional[int] = ..., attacker_cell_y: _Optional[int] = ..., victim_is_own: bool = ..., attacker_is_own: bool = ..., victim_is_building: bool = ...) -> None: ...

class RlEconomy(_message.Message):
    __slots__ = ("cash", "ore", "power_provided", "power_drained", "resource_capacity", "harvester_count")
    CASH_FIELD_NUMBER: _ClassVar[int]
    ORE_FIELD_NUMBER: _ClassVar[int]
    POWER_PROVIDED_FIELD_NUMBER: _ClassVar[int]
    POWER_DRAINED_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    HARVESTER_COUNT_FIELD_NUMBER: _ClassVar[int]
    cash: int
    ore: int
    power_provided: int
    power_drained: int
    resource_capacity: int
    harvester_count: int
    def __init__(self, cash: _Optional[int] = ..., ore: _Optional[int] = ..., power_provided: _Optional[int] = ..., power_drained: _Optional[int] = ..., resource_capacity: _Optional[int] = ..., harvester_count: _Optional[int] = ...) -> None: ...

class RlMilitary(_message.Message):
    __slots__ = ("units_killed", "units_lost", "buildings_killed", "buildings_lost", "army_value", "active_unit_count", "kills_cost", "deaths_cost", "assets_value", "experience", "order_count")
    UNITS_KILLED_FIELD_NUMBER: _ClassVar[int]
    UNITS_LOST_FIELD_NUMBER: _ClassVar[int]
    BUILDINGS_KILLED_FIELD_NUMBER: _ClassVar[int]
    BUILDINGS_LOST_FIELD_NUMBER: _ClassVar[int]
    ARMY_VALUE_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_UNIT_COUNT_FIELD_NUMBER: _ClassVar[int]
    KILLS_COST_FIELD_NUMBER: _ClassVar[int]
    DEATHS_COST_FIELD_NUMBER: _ClassVar[int]
    ASSETS_VALUE_FIELD_NUMBER: _ClassVar[int]
    EXPERIENCE_FIELD_NUMBER: _ClassVar[int]
    ORDER_COUNT_FIELD_NUMBER: _ClassVar[int]
    units_killed: int
    units_lost: int
    buildings_killed: int
    buildings_lost: int
    army_value: int
    active_unit_count: int
    kills_cost: int
    deaths_cost: int
    assets_value: int
    experience: int
    order_count: int
    def __init__(self, units_killed: _Optional[int] = ..., units_lost: _Optional[int] = ..., buildings_killed: _Optional[int] = ..., buildings_lost: _Optional[int] = ..., army_value: _Optional[int] = ..., active_unit_count: _Optional[int] = ..., kills_cost: _Optional[int] = ..., deaths_cost: _Optional[int] = ..., assets_value: _Optional[int] = ..., experience: _Optional[int] = ..., order_count: _Optional[int] = ...) -> None: ...

class RlUnitInfo(_message.Message):
    __slots__ = ("actor_id", "type", "pos_x", "pos_y", "cell_x", "cell_y", "hp_percent", "is_idle", "current_activity", "owner", "ammo", "can_attack", "facing", "experience_level", "stance", "speed", "attack_range", "passenger_count", "is_building")
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    POS_X_FIELD_NUMBER: _ClassVar[int]
    POS_Y_FIELD_NUMBER: _ClassVar[int]
    CELL_X_FIELD_NUMBER: _ClassVar[int]
    CELL_Y_FIELD_NUMBER: _ClassVar[int]
    HP_PERCENT_FIELD_NUMBER: _ClassVar[int]
    IS_IDLE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ACTIVITY_FIELD_NUMBER: _ClassVar[int]
    OWNER_FIELD_NUMBER: _ClassVar[int]
    AMMO_FIELD_NUMBER: _ClassVar[int]
    CAN_ATTACK_FIELD_NUMBER: _ClassVar[int]
    FACING_FIELD_NUMBER: _ClassVar[int]
    EXPERIENCE_LEVEL_FIELD_NUMBER: _ClassVar[int]
    STANCE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    ATTACK_RANGE_FIELD_NUMBER: _ClassVar[int]
    PASSENGER_COUNT_FIELD_NUMBER: _ClassVar[int]
    IS_BUILDING_FIELD_NUMBER: _ClassVar[int]
    actor_id: int
    type: str
    pos_x: int
    pos_y: int
    cell_x: int
    cell_y: int
    hp_percent: float
    is_idle: bool
    current_activity: str
    owner: str
    ammo: int
    can_attack: bool
    facing: int
    experience_level: int
    stance: int
    speed: int
    attack_range: int
    passenger_count: int
    is_building: bool
    def __init__(self, actor_id: _Optional[int] = ..., type: _Optional[str] = ..., pos_x: _Optional[int] = ..., pos_y: _Optional[int] = ..., cell_x: _Optional[int] = ..., cell_y: _Optional[int] = ..., hp_percent: _Optional[float] = ..., is_idle: bool = ..., current_activity: _Optional[str] = ..., owner: _Optional[str] = ..., ammo: _Optional[int] = ..., can_attack: bool = ..., facing: _Optional[int] = ..., experience_level: _Optional[int] = ..., stance: _Optional[int] = ..., speed: _Optional[int] = ..., attack_range: _Optional[int] = ..., passenger_count: _Optional[int] = ..., is_building: bool = ...) -> None: ...

class RlBuildingInfo(_message.Message):
    __slots__ = ("actor_id", "type", "pos_x", "pos_y", "hp_percent", "owner", "is_producing", "production_progress", "producing_item", "is_powered", "is_repairing", "sell_value", "rally_x", "rally_y", "power_amount", "can_produce", "cell_x", "cell_y")
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    POS_X_FIELD_NUMBER: _ClassVar[int]
    POS_Y_FIELD_NUMBER: _ClassVar[int]
    HP_PERCENT_FIELD_NUMBER: _ClassVar[int]
    OWNER_FIELD_NUMBER: _ClassVar[int]
    IS_PRODUCING_FIELD_NUMBER: _ClassVar[int]
    PRODUCTION_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    PRODUCING_ITEM_FIELD_NUMBER: _ClassVar[int]
    IS_POWERED_FIELD_NUMBER: _ClassVar[int]
    IS_REPAIRING_FIELD_NUMBER: _ClassVar[int]
    SELL_VALUE_FIELD_NUMBER: _ClassVar[int]
    RALLY_X_FIELD_NUMBER: _ClassVar[int]
    RALLY_Y_FIELD_NUMBER: _ClassVar[int]
    POWER_AMOUNT_FIELD_NUMBER: _ClassVar[int]
    CAN_PRODUCE_FIELD_NUMBER: _ClassVar[int]
    CELL_X_FIELD_NUMBER: _ClassVar[int]
    CELL_Y_FIELD_NUMBER: _ClassVar[int]
    actor_id: int
    type: str
    pos_x: int
    pos_y: int
    hp_percent: float
    owner: str
    is_producing: bool
    production_progress: float
    producing_item: str
    is_powered: bool
    is_repairing: bool
    sell_value: int
    rally_x: int
    rally_y: int
    power_amount: int
    can_produce: _containers.RepeatedScalarFieldContainer[str]
    cell_x: int
    cell_y: int
    def __init__(self, actor_id: _Optional[int] = ..., type: _Optional[str] = ..., pos_x: _Optional[int] = ..., pos_y: _Optional[int] = ..., hp_percent: _Optional[float] = ..., owner: _Optional[str] = ..., is_producing: bool = ..., production_progress: _Optional[float] = ..., producing_item: _Optional[str] = ..., is_powered: bool = ..., is_repairing: bool = ..., sell_value: _Optional[int] = ..., rally_x: _Optional[int] = ..., rally_y: _Optional[int] = ..., power_amount: _Optional[int] = ..., can_produce: _Optional[_Iterable[str]] = ..., cell_x: _Optional[int] = ..., cell_y: _Optional[int] = ...) -> None: ...

class RlProductionInfo(_message.Message):
    __slots__ = ("queue_type", "item", "progress", "remaining_ticks", "remaining_cost", "paused")
    QUEUE_TYPE_FIELD_NUMBER: _ClassVar[int]
    ITEM_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    REMAINING_TICKS_FIELD_NUMBER: _ClassVar[int]
    REMAINING_COST_FIELD_NUMBER: _ClassVar[int]
    PAUSED_FIELD_NUMBER: _ClassVar[int]
    queue_type: str
    item: str
    progress: float
    remaining_ticks: int
    remaining_cost: int
    paused: bool
    def __init__(self, queue_type: _Optional[str] = ..., item: _Optional[str] = ..., progress: _Optional[float] = ..., remaining_ticks: _Optional[int] = ..., remaining_cost: _Optional[int] = ..., paused: bool = ...) -> None: ...

class RlMapInfo(_message.Message):
    __slots__ = ("width", "height", "map_name")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    MAP_NAME_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    map_name: str
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., map_name: _Optional[str] = ...) -> None: ...

class AgentAction(_message.Message):
    __slots__ = ("commands",)
    COMMANDS_FIELD_NUMBER: _ClassVar[int]
    commands: _containers.RepeatedCompositeFieldContainer[Command]
    def __init__(self, commands: _Optional[_Iterable[_Union[Command, _Mapping]]] = ...) -> None: ...

class Command(_message.Message):
    __slots__ = ("action", "actor_id", "target_actor_id", "target_x", "target_y", "item_type", "queued", "ticks")
    ACTION_FIELD_NUMBER: _ClassVar[int]
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_X_FIELD_NUMBER: _ClassVar[int]
    TARGET_Y_FIELD_NUMBER: _ClassVar[int]
    ITEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    QUEUED_FIELD_NUMBER: _ClassVar[int]
    TICKS_FIELD_NUMBER: _ClassVar[int]
    action: ActionType
    actor_id: int
    target_actor_id: int
    target_x: int
    target_y: int
    item_type: str
    queued: bool
    ticks: int
    def __init__(self, action: _Optional[_Union[ActionType, str]] = ..., actor_id: _Optional[int] = ..., target_actor_id: _Optional[int] = ..., target_x: _Optional[int] = ..., target_y: _Optional[int] = ..., item_type: _Optional[str] = ..., queued: bool = ..., ticks: _Optional[int] = ...) -> None: ...

class GameState(_message.Message):
    __slots__ = ("episode_id", "tick", "phase", "winner", "player_count", "player_faction", "enemy_faction")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    WINNER_FIELD_NUMBER: _ClassVar[int]
    PLAYER_COUNT_FIELD_NUMBER: _ClassVar[int]
    PLAYER_FACTION_FIELD_NUMBER: _ClassVar[int]
    ENEMY_FACTION_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    tick: int
    phase: str
    winner: str
    player_count: int
    player_faction: str
    enemy_faction: str
    def __init__(self, episode_id: _Optional[str] = ..., tick: _Optional[int] = ..., phase: _Optional[str] = ..., winner: _Optional[str] = ..., player_count: _Optional[int] = ..., player_faction: _Optional[str] = ..., enemy_faction: _Optional[str] = ...) -> None: ...

class StateRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class FastAdvanceRequest(_message.Message):
    __slots__ = ("ticks", "commands", "session_id", "check_events_every", "enabled_interrupts")
    TICKS_FIELD_NUMBER: _ClassVar[int]
    COMMANDS_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CHECK_EVENTS_EVERY_FIELD_NUMBER: _ClassVar[int]
    ENABLED_INTERRUPTS_FIELD_NUMBER: _ClassVar[int]
    ticks: int
    commands: _containers.RepeatedCompositeFieldContainer[Command]
    session_id: str
    check_events_every: int
    enabled_interrupts: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ticks: _Optional[int] = ..., commands: _Optional[_Iterable[_Union[Command, _Mapping]]] = ..., session_id: _Optional[str] = ..., check_events_every: _Optional[int] = ..., enabled_interrupts: _Optional[_Iterable[str]] = ...) -> None: ...

class CreateSessionRequest(_message.Message):
    __slots__ = ("map_name", "bots", "seed")
    MAP_NAME_FIELD_NUMBER: _ClassVar[int]
    BOTS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    map_name: str
    bots: str
    seed: int
    def __init__(self, map_name: _Optional[str] = ..., bots: _Optional[str] = ..., seed: _Optional[int] = ...) -> None: ...

class CreateSessionResponse(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class DestroySessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class DestroySessionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
