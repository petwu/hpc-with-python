#!/bin/sh
# https://wiki.bwhpc.de/e/BwUniCluster_2.0_Batch_Queues


# parameters
SUFFIX="$1"


# 100x100
sbatch --nodes=1  --ntasks-per-node=1  --time=00:10:00 --partition=single   --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_1$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=2  --time=00:05:00 --partition=single   --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_2$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=4  --time=00:04:00 --partition=single   --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_4$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=8  --time=00:03:00 --partition=single   --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_8$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=16 --time=00:03:00 --partition=single   --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_16$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=32 --time=00:03:00 --partition=single   --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_32$SUFFIX.json"
sbatch --nodes=2  --ntasks-per-node=32 --time=00:03:00 --partition=multiple --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_64$SUFFIX.json"
sbatch --nodes=4  --ntasks-per-node=32 --time=00:03:00 --partition=multiple --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_128$SUFFIX.json"
sbatch --nodes=8  --ntasks-per-node=32 --time=00:03:00 --partition=multiple --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_256$SUFFIX.json"
sbatch --nodes=16 --ntasks-per-node=32 --time=00:03:00 --partition=multiple --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_512$SUFFIX.json"
sbatch --nodes=32 --ntasks-per-node=32 --time=00:04:00 --partition=multiple --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_1024$SUFFIX.json"
sbatch --nodes=64 --ntasks-per-node=32 --time=00:05:00 --partition=multiple --mail-type=FAIL job.sh -x 100 -y 100 -o "out/scaling/100x100_2048$SUFFIX.json"


# 300x300
sbatch --nodes=1  --ntasks-per-node=1  --time=01:20:00 --partition=single   --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_1$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=2  --time=00:40:00 --partition=single   --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_2$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=4  --time=00:20:00 --partition=single   --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_4$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=8  --time=00:10:00 --partition=single   --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_8$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=16 --time=00:10:00 --partition=single   --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_16$SUFFIX.json"
sbatch --nodes=1  --ntasks-per-node=32 --time=00:05:00 --partition=single   --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_32$SUFFIX.json"
sbatch --nodes=2  --ntasks-per-node=32 --time=00:05:00 --partition=multiple --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_64$SUFFIX.json"
sbatch --nodes=4  --ntasks-per-node=32 --time=00:05:00 --partition=multiple --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_128$SUFFIX.json"
sbatch --nodes=8  --ntasks-per-node=32 --time=00:05:00 --partition=multiple --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_256$SUFFIX.json"
sbatch --nodes=16 --ntasks-per-node=32 --time=00:10:00 --partition=multiple --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_512$SUFFIX.json"
sbatch --nodes=32 --ntasks-per-node=32 --time=00:20:00 --partition=multiple --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_1024$SUFFIX.json"
sbatch --nodes=64 --ntasks-per-node=32 --time=00:40:00 --partition=multiple --mail-type=FAIL job.sh -x 300 -y 300 -o "out/scaling/300x300_2048$SUFFIX.json"


# 500x500
sbatch --nodes=1   --ntasks-per-node=1  --time=04:00:00 --partition=single   --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_1$SUFFIX.json"
sbatch --nodes=1   --ntasks-per-node=2  --time=02:00:00 --partition=single   --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_2$SUFFIX.json"
sbatch --nodes=1   --ntasks-per-node=4  --time=01:00:00 --partition=single   --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_4$SUFFIX.json"
sbatch --nodes=1   --ntasks-per-node=8  --time=00:40:00 --partition=single   --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_8$SUFFIX.json"
sbatch --nodes=1   --ntasks-per-node=16 --time=00:20:00 --partition=single   --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_16$SUFFIX.json"
sbatch --nodes=1   --ntasks-per-node=32 --time=00:15:00 --partition=single   --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_32$SUFFIX.json"
sbatch --nodes=2   --ntasks-per-node=32 --time=00:10:00 --partition=multiple --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_64$SUFFIX.json"
sbatch --nodes=4   --ntasks-per-node=32 --time=00:10:00 --partition=multiple --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_128$SUFFIX.json"
sbatch --nodes=8   --ntasks-per-node=32 --time=00:10:00 --partition=multiple --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_256$SUFFIX.json"
sbatch --nodes=16  --ntasks-per-node=32 --time=00:10:00 --partition=multiple --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_512$SUFFIX.json"
sbatch --nodes=32  --ntasks-per-node=32 --time=00:15:00 --partition=multiple --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_1024$SUFFIX.json"
sbatch --nodes=64  --ntasks-per-node=32 --time=00:20:00 --partition=multiple --mail-type=FAIL job.sh -x 500 -y 500 -o "out/scaling/500x500_2048$SUFFIX.json"
