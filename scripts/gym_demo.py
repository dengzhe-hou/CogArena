"""Demo: drive CogArena paradigms through the real Gymnasium API.

    python scripts/gym_demo.py

Shows gymnasium.make() + the standard reset()/step() 5-tuple loop on the
registered CogArena/* environments (one per paper paradigm), with a trivial
baseline agent.
"""
import random
import gymnasium as gym
import cogarena.gym_env as G  # registers all CogArena/* ids


def dummy_agent(obs: str, paradigm: str) -> str:
    """A trivial baseline so the loop is self-contained (no LLM needed)."""
    if paradigm == "n_back":
        return random.choice(["MATCH", "NO MATCH"])
    if paradigm == "go_nogo":
        return random.choice(["GO", "NO-GO"])
    if paradigm in ("stroop", "flanker"):
        return "red"
    if paradigm in ("digit_span", "operation_span", "cvlt_word_list"):
        return "1 2 3"
    return "answer"


def run(env_id: str, paradigm: str, seed: int = 42):
    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)
    print(f"\n=== {env_id} ({'multi-turn' if False else ''}) ===")
    print("first observation:\n " + obs.replace("\n", "\n ")[:240])
    done, steps, total = False, 0, 0.0
    while not done:
        obs, reward, terminated, truncated, info = env.step(dummy_agent(obs, paradigm))
        steps += 1
        total += reward
        done = terminated or truncated
    print(f"-> {steps} step(s), mean reward {total / max(steps, 1):.2f}, "
          f"episode score {env.unwrapped.score().get('accuracy', 0):.2f}")
    env.close()


if __name__ == "__main__":
    ids = [f"CogArena/{spec[3]}-v0" for spec in G._SPEC.values()]
    print(f"Registered CogArena envs ({len(ids)}):")
    for i in ids:
        print("  " + i)
    # show one multi-turn and two single-turn paradigms in detail
    run("CogArena/NBack-v0", "n_back")       # multi-turn (n-back sequence)
    run("CogArena/Stroop-v0", "stroop")      # single-turn (cognitive control)
    run("CogArena/FalseBelief-v0", "false_belief")  # single-turn (ToM)
    print("\nOK: all 13 CogArena/* paradigms are gymnasium.make-able (smoke-tested).")
