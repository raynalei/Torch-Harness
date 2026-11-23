
import sys
import faulthandler
faulthandler.enable()

import pyrosetta
from pyrosetta import rosetta

pdb_path = sys.argv[1]
xml_path = sys.argv[2]

pyrosetta.init("-ignore_zero_occupancy false")

print("Loading pose", flush=True)
pose = pyrosetta.pose_from_pdb(pdb_path)

print("Parsing original XML", flush=True)
xml_objects = rosetta.protocols.rosetta_scripts.XmlObjects.create_from_file(xml_path)
protocol = xml_objects.get_mover("ParsedProtocol")

print("Applying original RosettaScripts", flush=True)
protocol.apply(pose)

print("XML protocol completed", flush=True)
pose.dump_pdb("test_xml_relaxed.pdb")
