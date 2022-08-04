# Scaling

The files in this directory were used to run the scaling test on the
[bwUniCluster](https://wiki.bwhpc.de/e/Category:BwUniCluster\_2.0):

1. Copy the necessary scripts to the repository root.
   ```sh
   $ cp report/scaling/bwunicluster.sh report/scaling/job.sh .
   ```
2. Create the Slurm jobs.
   ```sh
   $ sh bwunicluster.sh
   ```
3. Wait for all jobs to finish.
4. Create a tarball from the created JSON data.
   ```sh
   $ tar cf - -C out/scaling . | gzip -9 - > scaling.tar.gz
   ```
5. Use `scp` to copy the tarball to you local machine.
6. Plot the _#processes vs MLUPS_ graph.
   ```sh
   $ python3 report/scaling/scaling.py [...]
   ```
