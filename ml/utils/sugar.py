def check_div(num: int, denom: int) -> int:
    assert num % denom == 0, f"Can't evenly divide {num} by {denom}"
    return num // denom
