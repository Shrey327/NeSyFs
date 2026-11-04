"""Small deterministic, partially observed kitchen tasks for local comparisons."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Scenario:
    id: str
    goal_object: str
    goal_container: str
    objects: tuple[tuple[str, str], ...]
    open_containers: tuple[str, ...]
    label: str = "Kitchen task"
    description: str = "Move an object using the observed kitchen state."
    difficulty: str = "basic"
    min_actions: int = 0
    locked_containers: tuple[str, ...] = ()
    additional_goals: tuple[tuple[str, str], ...] = ()
    clean_goals: tuple[str, ...] = ()
    closed_goals: tuple[str, ...] = ()
    tap_off_goal: bool = False
    # Partial observability: contents of a closed container stay unseen until opened.
    hidden: bool = False

    @property
    def task(self) -> str:
        if self.difficulty == "basic":
            return (
                f"Put the {self.goal_object} in the {self.goal_container}. "
                "Commands: open <fridge or cupboard>; take <object>; "
                "put <object> in <fridge or cupboard>. "
                "A container must be open to take an object from it or put an object into it. "
                "You can hold only one object. Later observations report only the action outcome."
            )
        goals = [f"the {obj} in the {container}" for obj, container in
                 ((self.goal_object, self.goal_container), *self.additional_goals)]
        goals.extend(f"the {obj} clean" for obj in self.clean_goals)
        goals.extend(f"the {container} closed" for container in self.closed_goals)
        if self.tap_off_goal:
            goals.append("the tap off")
        return (
            "Goal: " + "; ".join(goals) + ". "
            "Commands: open <fridge or cupboard>; close <fridge or cupboard>; "
            "take <object>; put <object> in <fridge or cupboard>; "
            "put <object> on <table or counter>; unlock <fridge or cupboard> with key; "
            "turn on tap; turn off tap; wash <object>. "
            "A locked container must be unlocked while holding the key before it can be opened. "
            "A container must be open to take an object from it or put an object in it. "
            "Only one object can be held at once. Washing requires holding the object and the tap on. "
            "Turning the tap on or off and opening or closing a container do not require free hands. "
            + ("Objects inside a closed fridge or cupboard cannot be seen until you open it. "
               if self.hidden else "")
            + "Later observations report only the action outcome."
        )


SCENARIOS = (
    Scenario("apple_table_fridge", "apple", "fridge", (("apple", "table"), ("pear", "counter")), ("cupboard",), label="Apple to fridge", min_actions=3),
    Scenario("pear_cupboard_fridge", "pear", "fridge", (("pear", "cupboard"), ("apple", "table")), ("fridge",), label="Pear from cupboard", min_actions=3),
    Scenario("mug_counter_cupboard", "mug", "cupboard", (("mug", "counter"), ("apple", "table")), ("cupboard",), label="Mug to cupboard", min_actions=2),
    Scenario("apple_fridge_cupboard", "apple", "cupboard", (("apple", "fridge"), ("pear", "table")), (), label="Apple transfer", min_actions=4),
    Scenario("key_unlock_cupboard", "apple", "cupboard", (("apple", "counter"), ("key", "table"), ("pear", "fridge")), ("fridge",),
             label="Unlock and secure", description="Use a key, free your hand, store the apple, and close the cupboard.", difficulty="complex", min_actions=7,
             locked_containers=("cupboard",), closed_goals=("cupboard",)),
    Scenario("wash_pear_fridge", "pear", "fridge", (("pear", "table"), ("mug", "counter")), ("cupboard",),
             label="Wash and store", description="Wash the pear, store it, and leave the tap off.", difficulty="complex", min_actions=6,
             clean_goals=("pear",), tap_off_goal=True),
    Scenario("two_items_cupboard", "apple", "cupboard", (("apple", "fridge"), ("pear", "table"), ("mug", "counter")), (),
             label="Two-object transfer", description="Move two objects with one free hand and close the fridge.", difficulty="complex", min_actions=7,
             additional_goals=(("pear", "cupboard"),), closed_goals=("fridge",)),
    Scenario("wash_mug_secure", "mug", "fridge", (("mug", "cupboard"), ("key", "counter"), ("apple", "table")), ("fridge",),
             label="Unlock, wash and reset", description="Unlock the cupboard, wash the mug, store it, and restore the cupboard and tap.", difficulty="complex", min_actions=10,
             locked_containers=("cupboard",), clean_goals=("mug",), closed_goals=("cupboard",), tap_off_goal=True),
    # Partial-observability suite. Designed before any model was run on it: every
    # configuration must search closed containers and remember what it found.
    Scenario("hidden_apple_search", "apple", "fridge", (("apple", "cupboard"), ("pear", "fridge"), ("mug", "counter")), (),
             label="Search for the apple", description="Find the hidden apple, store it in the fridge, and close both containers.",
             difficulty="hidden", min_actions=6, closed_goals=("fridge", "cupboard"), hidden=True),
    Scenario("hidden_key_search", "apple", "cupboard", (("apple", "table"), ("key", "fridge"), ("pear", "cupboard")), (),
             label="Find the key", description="The cupboard is locked and the key is hidden. Store the apple and close everything.",
             difficulty="hidden", min_actions=9, locked_containers=("cupboard",), closed_goals=("fridge", "cupboard"), hidden=True),
    Scenario("hidden_swap_wash", "mug", "cupboard", (("mug", "fridge"), ("pear", "cupboard"), ("apple", "counter")), (),
             label="Hidden swap and wash", description="Swap two hidden objects between containers, wash one, and restore the kitchen.",
             difficulty="hidden", min_actions=11, additional_goals=(("pear", "fridge"),), clean_goals=("mug",),
             closed_goals=("fridge", "cupboard"), tap_off_goal=True, hidden=True),
)


class EvaluationKitchen:
    """A fresh instance exposes the initial layout, then only action outcomes."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self._objects = dict(scenario.objects)
        self._open = {"fridge": "fridge" in scenario.open_containers,
                      "cupboard": "cupboard" in scenario.open_containers}
        self._locked = {container: container in scenario.locked_containers for container in self._open}
        self._tap_on = False
        self._clean: set[str] = set()
        self._held: str | None = None
        self.invalid_actions = 0
        visible = [(obj, location) for obj, location in scenario.objects
                   if not (scenario.hidden and location in self._open and not self._open[location])]
        self._observation = (
            "Initial kitchen: "
            + "; ".join(f"the {obj} is in the {location}" if location in self._open
                        else f"the {obj} is on the {location}"
                        for obj, location in visible)
            + "".join(f"; you cannot see inside the closed {container}"
                      for container, opened in self._open.items() if scenario.hidden and not opened)
            + "; " + "; ".join(f"the {container} is {'open' if opened else 'closed'}"
                                  for container, opened in self._open.items())
            + ("; " + "; ".join(f"the {container} is {'locked' if locked else 'unlocked'}"
                                  for container, locked in self._locked.items())
               + "; the tap is off; all objects are unwashed" if scenario.difficulty != "basic" else "")
            + "; your hands are empty."
        )

    @property
    def is_complete(self) -> bool:
        goals = ((self.scenario.goal_object, self.scenario.goal_container), *self.scenario.additional_goals)
        return (all(self._objects[obj] == container for obj, container in goals)
                and all(obj in self._clean for obj in self.scenario.clean_goals)
                and all(not self._open[container] for container in self.scenario.closed_goals)
                and (not self.scenario.tap_off_goal or not self._tap_on))

    def observe(self) -> str:
        return self._observation

    def _invalid(self, message: str) -> str:
        if "nothing changed" not in message:
            message = message.rstrip(".") + "; nothing changed."
        self.invalid_actions += 1
        self._observation = message
        return message

    def execute(self, action: str) -> str:
        if not isinstance(action, str):
            return self._invalid("Invalid action: use the listed commands.")
        command = re.sub(r"[.!?]+$", "", action.strip().casefold())
        command = re.sub(r"\bthe\s+", "", command)
        # Models sometimes wrap a concrete command argument in the example's brackets.
        command = re.sub(r"<([a-z]+)>", r"\1", command)
        command = re.sub(r"\s+", " ", command)

        match = re.fullmatch(r"open (fridge|cupboard)", command)
        if match:
            container = match.group(1)
            if self._open[container]:
                return self._invalid(f"The {container} is already open; nothing changed.")
            if self._locked[container]:
                return self._invalid(f"The {container} is locked; nothing changed.")
            self._open[container] = True
            self._observation = f"You opened the {container}."
            if self.scenario.hidden:
                inside = [f"the {obj}" for obj, place in self._objects.items() if place == container]
                self._observation += f" Inside: {', '.join(inside) if inside else 'nothing'}."
            return self._observation

        match = re.fullmatch(r"close (fridge|cupboard)", command)
        if match and self.scenario.difficulty != "basic":
            container = match.group(1)
            if not self._open[container]:
                return self._invalid(f"The {container} is already closed; nothing changed.")
            self._open[container] = False
            self._observation = f"You closed the {container}."
            return self._observation

        match = re.fullmatch(r"unlock (fridge|cupboard) with key", command)
        if match and self.scenario.difficulty != "basic":
            container = match.group(1)
            if not self._locked[container]:
                return self._invalid(f"The {container} is already unlocked; nothing changed.")
            if self._held != "key":
                return self._invalid(f"You must hold the key to unlock the {container}.")
            self._locked[container] = False
            self._observation = f"You unlocked the {container} with the key."
            return self._observation

        match = re.fullmatch(r"turn (on|off) tap", command)
        if match and self.scenario.difficulty != "basic":
            desired = match.group(1) == "on"
            if self._tap_on == desired:
                return self._invalid(f"The tap is already {match.group(1)}; nothing changed.")
            self._tap_on = desired
            self._observation = f"You turned the tap {match.group(1)}."
            return self._observation

        match = re.fullmatch(r"wash ([a-z]+)", command)
        if match and self.scenario.difficulty != "basic":
            obj = match.group(1)
            if self._held != obj:
                return self._invalid(f"You must hold the {obj} to wash it.")
            if not self._tap_on:
                return self._invalid("The tap is off; nothing was washed.")
            if obj in self._clean:
                return self._invalid(f"The {obj} is already clean; nothing changed.")
            self._clean.add(obj)
            self._observation = f"You washed the {obj}; it is now clean."
            return self._observation

        match = re.fullmatch(r"(?:take|pick up|grab) ([a-z]+)(?: from (fridge|cupboard|table|counter))?", command)
        if match:
            obj = match.group(1)
            if self.scenario.hidden:
                # Check visible conditions before looking up an object's hidden location.
                if self._held is not None:
                    return self._invalid("Your hands are already full; nothing changed.")
                source = match.group(2)
                if source in self._open and not self._open[source]:
                    return self._invalid(f"The {source} is closed; nothing changed.")
                location = self._objects.get(obj)
                if source is not None and source != location:
                    return self._invalid(f"The {obj} is not in or on the {source}; nothing changed.")
                if location is None or (location in self._open and not self._open[location]):
                    return self._invalid(f"You cannot see the {obj} anywhere you can reach; nothing changed.")
            if obj not in self._objects:
                return self._invalid(f"There is no {obj} in this kitchen.")
            if self._held is not None:
                return self._invalid("Your hands are already full; nothing changed.")
            location = self._objects[obj]
            if match.group(2) is not None and match.group(2) != location:
                return self._invalid(f"The {obj} is not in or on the {match.group(2)}; nothing changed.")
            if location in self._open and not self._open[location]:
                return self._invalid(f"The {location} is closed; you cannot take the {obj}.")
            self._objects[obj] = "held"
            self._held = obj
            self._observation = f"You took the {obj} and are holding it."
            return self._observation

        match = re.fullmatch(r"(?:put|place|store) ([a-z]+) (?:in|inside|into) (fridge|cupboard)", command)
        if match:
            obj, container = match.groups()
            if self._held != obj:
                return self._invalid(f"You are not holding the {obj}; nothing changed.")
            if not self._open[container]:
                return self._invalid(f"The {container} is closed; you are still holding the {obj}.")
            self._objects[obj] = container
            self._held = None
            self._observation = f"You put the {obj} in the {container}."
            return self._observation

        match = re.fullmatch(r"(?:put|place|set) ([a-z]+) (?:on|onto) (table|counter)", command)
        if match and self.scenario.difficulty != "basic":
            obj, surface = match.groups()
            if self._held != obj:
                return self._invalid(f"You are not holding the {obj}; nothing changed.")
            self._objects[obj] = surface
            self._held = None
            self._observation = f"You put the {obj} on the {surface}."
            return self._observation

        return self._invalid("Invalid action: use the listed commands.")
