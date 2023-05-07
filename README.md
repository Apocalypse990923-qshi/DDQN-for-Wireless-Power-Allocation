# DDQN-for-Wireless-Power-Allocation
EE597 Final Project\
Deep Reinforcement Learning for Resource Allocation in Multi-Agent Wireless Networks Proposal\
Qiushi Xu, Yiyi Li\
Power and wireless resource allocation problem has been studied since Long-Term Evolution (LTE)
system and 5G mobile communication system. However, with the rapid growth of network scale and the
dramatical increasement of the number of Base Stations (BSs), the problem of the optimal power
allocation becomes particularly acute.\
Due to the non-convex and combinatorial characteristics, it is challenging for the traditional optimization
decomposition method (such as Water Filling) to obtain an optimal strategy for the joint user association
and resource allocation issue. Compared with other methods, such as game-theoretic approach, linear
programming, and Markov approximation strategy, which require nearly complete information, some
research indicates that reinforcement learning can solve this information constraints and find a better
way to allocate the power resources.\
In this research project, firstly we would like to conduct the code reproduction of Deep-Q-Full- Connected-Network (DQFNet) inspired by [1]. Then we will try to revise the structure and parameters of
DQFNet, by which we expect to get a better solution of power allocation.\
It is also a potential research direction to optimize dueling double deep Q-network(D3QN) proposed by
[2] and apply it to DQFNet to find a better power allocation in cellular networks.

REFERENCES\
[1] Y. Zhang, C. Kang, T. Ma, Y. Teng and D. Guo, "Power Allocation in Multi-Cell Networks Using Deep
Reinforcement Learning," 2018 IEEE 88th Vehicular Technology Conference (VTC-Fall), Chicago, IL, USA, 2018, pp. 1-6, doi: 10.1109/VTCFall.2018.8690757.\
[2] N. Zhao, Y. -C. Liang, D. Niyato, Y. Pei, M. Wu and Y. Jiang, "Deep Reinforcement Learning for User
Association and Resource Allocation in Heterogeneous Cellular Networks," in IEEE Transactions on
Wireless Communications, vol. 18, no. 11, pp. 5141-5152, Nov. 2019, doi: 10.1109/TWC.2019.2933417.
