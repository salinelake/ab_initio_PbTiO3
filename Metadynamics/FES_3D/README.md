
data folder contains metadynamics simulation results for different temperatures:
- 300K9x9x9: Contains metadynamics simulation results at 300K temperature with 9x9x9 supercell of PbTiO3
- 600K9x9x9: Contains metadynamics simulation results at 600K temperature with 9x9x9 supercell of PbTiO3
- 700K9x9x9: Contains metadynamics simulation results at 700K temperature with 9x9x9 supercell of PbTiO3
- 800K9x9x9: Contains metadynamics simulation results at 800K temperature with 9x9x9 supercell of PbTiO3
- 820K9x9x9: Contains metadynamics simulation results at 820K temperature with 9x9x9 supercell of PbTiO3
- 900K9x9x9: Contains metadynamics simulation results at 900K temperature with 9x9x9 supercell of PbTiO3
- 1000K9x9x9: Contains metadynamics simulation results at 1000K temperature with 9x9x9 supercell of PbTiO3

Each temperature folder contains:
- COLVAR: trajectory file of collective variable output
- HILLS: Gaussian hills deposited during metadynamics
- in.lammps: LAMMPS input file for metadynamics
- plumed.dat: PLUMED input file for metadynamics
- run.slurm: SLURM script for submitting the simulation task to computing cluster
- plumed.out: output file of PLUMED
- pto.log: logging file of LAMMPS
- cubic.lmp: initial configuration file of LAMMPS








