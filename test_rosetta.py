
import faulthandler
faulthandler.enable()

import pyrosetta
from pyrosetta import rosetta

PDB = "/lustre/grp/cmclab/leiy/harness/runs/clean7n0a-ab-ppiflow-04/ppiflow-14f9003395870eff/rosetta-fix/input/candidate.pdb"

print("1. Initializing PyRosetta", flush=True)
pyrosetta.init("-ignore_zero_occupancy false")

print("2. Loading PDB", flush=True)
pose = pyrosetta.pose_from_pdb(PDB)
print("Residues:", pose.total_residue(), flush=True)

print("3. Scoring", flush=True)
scorefxn = pyrosetta.get_fa_scorefxn()
print("Score:", scorefxn(pose), flush=True)

print("4. Starting FastRelax", flush=True)
relax = rosetta.protocols.relax.FastRelax(scorefxn, 1)
relax.apply(pose)

print("5. Relax completed", flush=True)
pose.dump_pdb("/tmp/rosetta_test_relaxed.pdb")
