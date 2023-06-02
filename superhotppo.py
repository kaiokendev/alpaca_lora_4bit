import re
from typing import Any, Tuple
import torch

# Automated PPO Reward Policies


def compute_rewards(outputs: list[Tuple[str, dict]]):
    rewards = []
    for out in outputs:
        reward = []
        reward += [sequence(out["response"], out["expectations"]["sequence"])]
        reward = torch.sigmoid(torch.FloatTensor(reward))
        reward = sum(reward) / len(reward)
        rewards.append(reward)
    return rewards


def reward(condition, if_true=1, if_false=-1):
    return [if_true] if condition is True else [if_false]


def reward_val(expected, actual, deviation_scale=1.0, deviation_penalty=1.0):
    return


def sequence(response, expectations: dict[str, Any]):
    """
    Rewards the model for producing correct orderings when following instructions
    """
    rewards = []

    """Do we list exactly 1..K entries?"""
    if "k" in expectations:
        matches = re.findall(r"\n([1-9]\.)+.", response)
        rewards += reward(len(matches) == expectations["k"])

    """Do we list at least 1..K entries?"""
    if "min" in expectations:
        matches = re.findall(r"\n([1-9]\.)+.", response)
        rewards += reward(len(matches) >= expectations["k"])

    """Do we reiterate the number K as "K..."?"""
    if "reiterate" in expectations:
        if isinstance(expectations["reiterate"], bool):
            k = "1-9"
        else:
            k = expectations["reiterate"]
        matches = re.match(rf"[{k}].*:", response)
        rewards += reward(matches is not None)

    """Do we perform an expected sequence of tasks in the correct order?"""
    if "order" in expectations:
        pass

    return sum(rewards)


def chat(response, expections: dict[str, Any]):
    """
    Rewards the model for good roleplay chat responses
    """

    rewards = []
    char = expections["charname"]
    user = expections["username"]
    structure = []
    messages = []
    in_dialogue = False
    in_action = False
    was_in_dialogue_at_least_once = False
    buf = ""
    for c in response:
        buf += c
        match c:
            case '"':
                was_in_dialogue_at_least_once = True
                if in_dialogue:
                    structure.append("dialogue")
                    if buf.startswith(char):
                        messages.append(
                            {"sender": char, "message": buf, "complete": True}
                        )
                    elif buf.startswith(user):
                        messages.append(
                            {"sender": user, "message": buf, "complete": True}
                        )
                    else:
                        messages.append(
                            {"sender": "<unk>", "message": buf, "complete": True}
                        )
                    buf = ""
                in_dialogue = not in_dialogue
            case "*":
                if in_action and not in_dialogue:
                    structure.append("action")
                    buf = ""
                in_action = not in_action

        if buf.startswith(char):
            messages.append({"sender": char, "message": buf, "complete": False})
        elif buf.startswith(user):
            messages.append({"sender": user, "message": buf, "complete": False})
        else:
            messages.append({"sender": "<unk>", "message": buf, "complete": False})

    """Do we properly wrap our actions and dialogue?"""
    if "proper" in expections:
        rewards += reward(response[0] == '"' or response[0] == "*")

    """Do we use a different response structure each time?"""
    if "structures" in expections:
        rewards += reward(structure not in expections["structures"])

    """Do we generate the user's name after our response?"""
    if "pass_turn" in expections:
        pass

    """Are our replies long?"""
    if "long" in expections:
        pass

    """Are our replies not too long?"""
    if "should_fit" in expections:
        rewards += reward(all([m for m in messages if m["complete"]]))

    """Did we introduce any unexpected characters?"""
    if "only_expected_chars" in expections:
        pass

    """Did we respond with dialogue?"""
    if "at_least_one_dialogue" in expections:
        rewards += reward(was_in_dialogue_at_least_once)

    """Do we start or end our messages as instructed?"""
    if "start_end_with" in expections:
        pass

    """Are our messages long enough?"""
    if "min_length" in expections:
        pass

    return sum(rewards)


def storytelling(response, expectations: dict[str, Any]):
    """
    Rewards the model for good storytelling
    """
    pass


def step_by_step(response, expectations: dict[str, Any]):
    """
    Rewards good and accurate step-by-step problem solving
    """
    pass


def generalization(response, expectations: dict[str, Any]):
    """
    Rewards generalizing behaviors to new data
    """
    pass


def characterization(response, expectations: dict[str, Any]):
    """
    Rewards accurate portrayal of characters
    """
    pass


def spatial_awareness(response, expectations: dict[str, Any]):
    """
    Rewards solid understanding of the positions of entities and how they move in a scene
    """
    pass


def coherence(response, expectations: dict[str, Any]):
    """
    Rewards staying on topic
    """
    pass


def creativity(response, expectations: dict[str, Any]):
    """
    Rewards deviating from a topic when applicable
    """
    pass
