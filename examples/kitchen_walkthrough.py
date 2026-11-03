"""Exercise the kitchen environment with scripted actions, without LLM calls."""

from nesyfs import KitchenDemoEnvironment


def main() -> None:
    environment = KitchenDemoEnvironment()
    print("Scripted environment walkthrough (no agent or model calls).")
    print(f"Initial observation: {environment.observe()}")
    for action in ("take apple", "open fridge", "put apple in fridge"):
        print(f"Action: {action}")
        print(f"Observation: {environment.execute(action)}")
    if not environment.is_complete:
        raise RuntimeError("The scripted kitchen walkthrough did not complete")
    print("Complete: True")


if __name__ == "__main__":
    main()
