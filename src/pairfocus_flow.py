import random


FLOW_UNKNOWN = [0, 0]
FLOW_FREE = [1, 0]
FLOW_CONGESTED = [0, 1]
VALID_FLOW_STATES = (FLOW_UNKNOWN, FLOW_FREE, FLOW_CONGESTED)


def exact_flow_bits(is_congested):
    return list(FLOW_CONGESTED if is_congested else FLOW_FREE)


def is_valid_2bit_state(bits):
    return list(bits) in VALID_FLOW_STATES


def is_valid_2bit_sections(bits):
    return (
        len(bits) % 2 == 0
        and all(is_valid_2bit_state(bits[i:i + 2]) for i in range(0, len(bits), 2))
    )


def random_flow4():
    left = random.choice(VALID_FLOW_STATES)
    right = random.choice(VALID_FLOW_STATES)
    return list(left) + list(right)


def random_flow4_except(true_flow4):
    choices = [
        list(left) + list(right)
        for left in VALID_FLOW_STATES
        for right in VALID_FLOW_STATES
    ]
    choices = [flow for flow in choices if flow != list(true_flow4)]
    return random.choice(choices)