#!/usr/bin/env bash

#### copy from cloud ####

gcloud compute scp --recurse xli4217@rlfps-small:/home/xli4217/docker/docker_home/robot_cooking/docker/dirs-to-copy/learning/experiments/* ${LEARNING_PATH}learning/experiments/cloud/rlfps-small/

# gcloud compute scp --recurse xli4217@rlfps1:/home/xli4217/docker/docker_home/rlfps/examples/iros2019/experiments/test/* ${RLFPS_PATH}/examples/iros2019/experiments/cloud/rlfps1/

# gcloud compute scp --recurse xli4217@rlfps2:/home/xli4217/docker/docker_home/rlfps/examples/iros2019/experiments/test/* ${RLFPS_PATH}/examples/iros2019/experiments/cloud/rlfps2/

# gcloud compute scp --recurse xli4217@rlfps3:/home/xli4217/docker/docker_home/rlfps/examples/iros2019/experiments/test/* ${RLFPS_PATH}/examples/iros2019/experiments/cloud/rlfps3/

#### copy to Seagate ####
# gcloud compute scp --recurse xli4217@rlfps:/home/xli4217/docker/docker_home/rlfps/examples/icml2019/experiments/* /media/xli4217/Seagate_Drive/rlfps_data/icml2019_data/


#### copy to cloud ####
# gcloud compute scp --recurse ${RLFPS_PATH}/examples/rss2019/experiments/t2.1_ddpg/* xli4217@rlfps:/home/xli4217/docker/docker_home/rlfps/examples/rss2019/experiments/t2.1_ddpg/

# gcloud compute scp --recurse ${RLFPS_PATH}/examples/rss2019/experiments/t2.2_ddpg/* xli4217@rlfps:/home/xli4217/docker/docker_home/rlfps/examples/rss2019/experiments/t2.2_ddpg/

