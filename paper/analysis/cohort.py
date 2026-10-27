"""The identities analyzed in the manuscript and where their photographs live."""

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def stim_dir():
    """The DOPP_STIM folder (see paper/README.md)."""
    value = os.environ.get("DOPP_STIM")
    if not value:
        sys.exit("Set the DOPP_STIM environment variable (see paper/README.md).")
    return value


def identities():
    """The 76 identities in identities.txt, as (group, number) pairs."""
    with open(os.path.join(HERE, "identities.txt")) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return [tuple(line.split("-")) for line in lines]


def original(group, num):
    """Path of the neutral CFD photograph for an identity."""
    (path,) = glob.glob(f"{stim_dir()}/{group}/CFD-{group}-{num}-*-N.jpg")
    return path
