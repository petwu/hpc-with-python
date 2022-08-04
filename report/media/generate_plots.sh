#!/bin/sh

THIS_DIR=$(dirname "$0")
cd "$THIS_DIR/../.."

# delete old output
rm -f out/*.png out/*.webm

# activate venv
. ./venv/bin/activate

# run milestones that generate the wanted plots
echo -e "\n[milestone 3: shear wave decay]"
python3 milestone.py m3
echo -e "\n[milestone 4: couette]"
python3 milestone.py m4
echo -e "\n[milestone 5: poiseuille]"
python3 milestone.py m5
echo -e "\n[milestone 6: lid-driven cavity]"
python3 milestone.py m6 --dpi=150
echo -e "\n[milestone 7: serial]"
python3 milestone.py m7
echo -e "\n[milestone 7: parallel]"
mpiexec --mca opal_warn_on_missing_libcuda 0 --use-hwthread-cpus -n 16 python milestone.py m7
echo -e "\n[milestone 7: scaling]"
python3 report/scaling/scaling.py

# copy outputs
cp out/*.png out/*.webm "$THIS_DIR/"
