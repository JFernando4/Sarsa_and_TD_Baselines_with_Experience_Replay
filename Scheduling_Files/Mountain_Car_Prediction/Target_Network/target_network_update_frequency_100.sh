#!/usr/bin/env bash

echo Enter the number of agents you want to train:
read number_of_agents
echo Enter the number of the first agent:
read first_agent_number
last_agent_number=$(($first_agent_number + $number_of_agents - 1))

export PYTHONPATH=.
for (( i=$first_agent_number; i <= $last_agent_number; i++ ))
do
    echo "Training Agent $i"
    python3 ./td_zero_mountain_car_prediction.py -episodes 2000 \
    -name Target_Network_Results/Target_Network_Update_Freq100/agent_$i \
    -quiet -tnetwork_update_freq 100 -alpha 0.00025 -hidden_units 135 -max_steps 1000
done