from __future__ import annotations


class KitchenDemoEnvironment:
    """Tiny local environment for trying the full action/observation loop."""

    def __init__(self):
        self.apple_location = "table"
        self.holding_apple = False
        self.fridge_open = False

    @property
    def is_complete(self) -> bool:
        return self.apple_location == "fridge"

    def observe(self) -> str:
        if self.is_complete:
            return "The apple is inside the fridge."
        if self.holding_apple:
            door = "open" if self.fridge_open else "closed"
            return f"You are holding the apple. The fridge is {door}."
        return "You are in the kitchen. An apple is on the table. The fridge is closed."

    def execute(self, action: str) -> str:
        normalized = action.casefold()
        if "apple" in normalized and any(word in normalized for word in ("pick", "take", "grab")):
            if self.apple_location == "table":
                self.apple_location = "held"
                self.holding_apple = True
                return "You picked up the apple and are holding it."
        if any(word in normalized for word in ("open", "unlock")) and any(
            word in normalized for word in ("fridge", "refrigerator")
        ):
            self.fridge_open = True
            return "The fridge is open."
        if any(word in normalized for word in ("put", "place", "store")) and "apple" in normalized and any(
            word in normalized for word in ("fridge", "refrigerator")
        ):
            if not self.holding_apple:
                return "You are not holding the apple."
            if not self.fridge_open:
                return "The fridge is closed. You are still holding the apple."
            self.apple_location = "fridge"
            self.holding_apple = False
            return "You put the apple in the fridge. The task is complete."
        return self.observe() + f" The action had no effect: {action}"
