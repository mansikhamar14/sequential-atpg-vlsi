"""
9-Valued Logic Engine for Sequential ATPG (Muth's Algorithm)
==============================================================
Each signal is represented as a pair (fault_free_value / faulty_value).
Component values: 0, 1, X (unknown)

The 9 values:
  0/0, 0/1, 1/0, 1/1, 0/X, X/0, 1/X, X/1, X/X

Correspondence to D-calculus:
  D    = 1/0  (good=1, faulty=0)
  D_BAR= 0/1  (good=0, faulty=1)
  X/X  = fully unknown

The advantage over 5-valued logic: partial knowledge states like
0/X, X/0, 1/X, X/1 allow progress when the 5-valued system gets stuck
on initialization of sequential circuits.
"""

# ── All 9 valid symbols ──────────────────────────────────────────────────────

NINE_VALUES = ["0/0", "0/1", "1/0", "1/1", "0/X", "X/0", "1/X", "X/1", "X/X"]


# ── Encoding / Decoding ─────────────────────────────────────────────────────

def decode(v):
    """Decode "g/f" string → (good, faulty) tuple. Components are int or "X"."""
    g, f = v.split("/")
    g = int(g) if g != "X" else "X"
    f = int(f) if f != "X" else "X"
    return g, f

def encode(g, f):
    """Encode (good, faulty) → "g/f" string."""
    return f"{g}/{f}"


# ── 3-Valued Primitives ─────────────────────────────────────────────────────

def _and3(a, b):
    if a == 0 or b == 0: return 0
    if a == 1 and b == 1: return 1
    return "X"

def _or3(a, b):
    if a == 1 or b == 1: return 1
    if a == 0 and b == 0: return 0
    return "X"

def _not3(a):
    if a == 0: return 1
    if a == 1: return 0
    return "X"

def _xor3(a, b):
    if a == "X" or b == "X": return "X"
    return int(a) ^ int(b)


# ── 9-Valued Gate Operations ────────────────────────────────────────────────
# Each applies the 3-valued operation independently to good and faulty halves.

def and_9val(*args):
    g, f = decode(args[0])
    for v in args[1:]:
        vg, vf = decode(v)
        g = _and3(g, vg)
        f = _and3(f, vf)
    return encode(g, f)

def or_9val(*args):
    g, f = decode(args[0])
    for v in args[1:]:
        vg, vf = decode(v)
        g = _or3(g, vg)
        f = _or3(f, vf)
    return encode(g, f)

def not_9val(a):
    g, f = decode(a)
    return encode(_not3(g), _not3(f))

def nand_9val(*args):
    return not_9val(and_9val(*args))

def nor_9val(*args):
    return not_9val(or_9val(*args))

def xor_9val(*args):
    g, f = decode(args[0])
    for v in args[1:]:
        vg, vf = decode(v)
        g = _xor3(g, vg)
        f = _xor3(f, vf)
    return encode(g, f)

def xnor_9val(*args):
    return not_9val(xor_9val(*args))

def buf_9val(a):
    return a


# ── Gate Dispatch ────────────────────────────────────────────────────────────

GATE_FN_9V = {
    "and":  and_9val,
    "or":   or_9val,
    "not":  not_9val,
    "nand": nand_9val,
    "nor":  nor_9val,
    "xor":  xor_9val,
    "xnor": xnor_9val,
    "buf":  buf_9val,
}

def evaluate_gate_9v(gate_type, input_values):
    """Evaluate a gate given its type and list of 9-valued input strings."""
    fn = GATE_FN_9V.get(gate_type.lower())
    if fn is None:
        raise ValueError(f"Unknown gate type: {gate_type}")
    return fn(*input_values)


# ── Helpers ──────────────────────────────────────────────────────────────────

def fault_value_9v(fault_type):
    """
    Return the 9-valued activation value for a stuck-at fault.
      SA0: good=1, faulty=0 → '1/0' (D)
      SA1: good=0, faulty=1 → '0/1' (D̄)
    """
    if fault_type.upper() == "SA0":
        return "1/0"
    elif fault_type.upper() == "SA1":
        return "0/1"
    raise ValueError(f"Unknown fault type: {fault_type}")

def is_discrepant(v):
    """True if good ≠ faulty AND both are known (fault effect present)."""
    g, f = decode(v)
    if g == "X" or f == "X":
        return False
    return g != f

def is_unknown(v):
    """True if either component is X."""
    g, f = decode(v)
    return g == "X" or f == "X"

def good_val(v):
    """Extract the fault-free component."""
    return decode(v)[0]

def faulty_val(v):
    """Extract the faulty component."""
    return decode(v)[1]

def ctrl_val_9v(gate_type):
    """Controlling value for AND/NAND/OR/NOR gates."""
    if gate_type in ("and", "nand"): return 0
    if gate_type in ("or", "nor"):   return 1
    return None

def inv_flag_9v(gate_type):
    """True if gate inverts output."""
    return gate_type in ("nand", "nor", "not", "xnor")

def merge_9val(a, b):
    """
    Merge two 9-valued signals (intersection of information).
    Returns None on conflict.
    """
    if a == "X/X": return b
    if b == "X/X": return a
    if a == b: return a
    ag, af = decode(a)
    bg, bf = decode(b)
    rg = ag if bg == "X" else (bg if ag == "X" else (ag if ag == bg else None))
    rf = af if bf == "X" else (bf if af == "X" else (af if af == bf else None))
    if rg is None or rf is None:
        return None
    return encode(rg, rf)


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 9-Valued Logic Self-Test ===\n")

    tests = [
        ("AND(D,1)",    and_9val("1/0", "1/1"),   "1/0"),
        ("AND(D,0)",    and_9val("1/0", "0/0"),   "0/0"),
        ("AND(D,D')",   and_9val("1/0", "0/1"),   "0/0"),
        ("AND(X/X,0)",  and_9val("X/X", "0/0"),   "0/0"),
        ("OR(D,0)",     or_9val("1/0", "0/0"),    "1/0"),
        ("OR(X/X,1)",   or_9val("X/X", "1/1"),    "1/1"),
        ("NOT(D)",      not_9val("1/0"),           "0/1"),
        ("NOT(X/X)",    not_9val("X/X"),           "X/X"),
        ("NAND(1,D)",   nand_9val("1/1","1/0"),   "0/1"),
        ("NOR(0,D)",    nor_9val("0/0","1/0"),     "0/1"),
        ("Merge",       merge_9val("1/X","X/0"),   "1/0"),
        ("Merge(c)",    merge_9val("1/0","0/1"),   None),
    ]

    ok = True
    for label, result, expected in tests:
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL": ok = False
        print(f"  [{status}] {label}: got={result}, expected={expected}")
    print(f"\n{'All passed ✓' if ok else 'FAILURES ✗'}")
