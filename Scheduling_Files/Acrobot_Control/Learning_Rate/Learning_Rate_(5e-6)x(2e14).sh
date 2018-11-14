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
    python3 ./sarsa_zero_acrobot_control.py -episodes 500 \
    -name Learning_Rate_Results/Learning_Rate_"(5e-6)x(2e14)"/agent_$i \
    -quiet -tnetwork_update_freq 100 -alpha "5e-6 * (2**14)" -onpolicy -hidden_units 2000 -max_steps 500 \
    -replay_start 100
done