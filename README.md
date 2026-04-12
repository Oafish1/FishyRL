# Fishy RL

![Documentation](https://github.com/Oafish1/FishyRL/actions/workflows/build-docs.yml/badge.svg)

**Fishy RL** is a distributed reinforcement learning framework for model-based algorithms and Dreamer-V3. It is designed to be *flexible* and *user-friendly*, allowing researchers and practitioners to easily interchange components and environments. Please check out the [documentation](https://oafish1.github.io/FishyRL/) to get started.

<table>
    <tbody>
        <!-- Hopper -->
        <!-- <tr>
            <td colspan=4 style="text-align: center"><b>MuJoCo Hopper-v5</b></td>
        </tr>
        <tr>
            <td style="text-align: center">10K Steps</td>
            <td style="text-align: center">50K Steps</td>
            <td style="text-align: center">100K Steps</td>
            <td style="text-align: center">200K Steps</td>
        </tr>
        <tr>
            <td style="text-align: center"><img src="./examples/images/Hopper_10k.gif" width="100%" alt="Trained Dreamer-V3 agent after 10k steps on MuJoCo Hopper-v5 environment"></td>
            <td style="text-align: center"><img src="./examples/images/Hopper_50k.gif" width="100%" alt="Trained Dreamer-V3 agent after 50k steps on MuJoCo Hopper-v5 environment"></td>
            <td style="text-align: center"><img src="./examples/images/Hopper_100k.gif" width="100%" alt="Trained Dreamer-V3 agent after 100k steps on MuJoCo Hopper-v5 environment"></td>
            <td style="text-align: center"><img src="./examples/images/Hopper_200k.gif" width="100%" alt="Trained Dreamer-V3 agent after 200k steps on MuJoCo Hopper-v5 environment"></td>
        </tr> -->
        <!-- Ant -->
        <tr bgcolor="#2b2b68" >
            <td colspan=4 style="text-align: center"><b>MuJoCo Ant-v5</b></td>
        </tr>
        <tr>
            <td style="text-align: center">10K Steps</td>
            <td style="text-align: center">50K Steps</td>
            <td style="text-align: center">100K Steps</td>
            <td style="text-align: center">200K Steps</td>
        </tr>
        <tr>
            <td style="text-align: center"><img src="./examples/images/Ant_10k.gif" width="100%" alt="Trained Dreamer-V3 agent after 10k steps on MuJoCo Ant-v5 environment"></td>
            <td style="text-align: center"><img src="./examples/images/Ant_50k.gif" width="100%" alt="Trained Dreamer-V3 agent after 50k steps on MuJoCo Ant-v5 environment"></td>
            <td style="text-align: center"><img src="./examples/images/Ant_100k.gif" width="100%" alt="Trained Dreamer-V3 agent after 100k steps on MuJoCo Ant-v5 environment"></td>
            <td style="text-align: center"><img src="./examples/images/Ant_200k.gif" width="100%" alt="Trained Dreamer-V3 agent after 200k steps on MuJoCo Ant-v5 environment"></td>
        </tr>
        <!-- BipedalWalker -->
        <tr bgcolor="#2b2b68">
            <td colspan=4 style="text-align: center"><b>LunarLander-v3</b></td>
        </tr>
        <tr>
            <td style="text-align: center">10K Steps</td>
            <td style="text-align: center">50K Steps</td>
            <td style="text-align: center">100K Steps</td>
            <td style="text-align: center">200K Steps</td>
        </tr>
        <tr>
            <td style="text-align: center"><img src="./examples/images/BipedalWalker_10k.gif" width="100%" alt="Trained Dreamer-V3 agent after 10k steps on BipedalWalker-v3 environment"></td>
            <td style="text-align: center"><img src="./examples/images/BipedalWalker_50k.gif" width="100%" alt="Trained Dreamer-V3 agent after 50k steps on BipedalWalker-v3 environment"></td>
            <td style="text-align: center"><img src="./examples/images/BipedalWalker_100k.gif" width="100%" alt="Trained Dreamer-V3 agent after 100k steps on BipedalWalker-v3 environment"></td>
            <td style="text-align: center"><img src="./examples/images/BipedalWalker_200k.gif" width="100%" alt="Trained Dreamer-V3 agent after 200k steps on BipedalWalker-v3 environment"></td>
        </tr>
        <!-- LunarLander -->
        <tr bgcolor="#2b2b68">
            <td colspan=4 style="text-align: center"><b>LunarLander-v3</b></td>
        </tr>
        <tr>
            <td style="text-align: center">1K Steps</td>
            <td style="text-align: center">10K Steps</td>
            <td style="text-align: center">25K Steps</td>
            <td style="text-align: center">50K Steps</td>
        </tr>
        <tr>
            <td style="text-align: center"><img src="./examples/images/LunarLander_1k.gif" width="100%" alt="Trained Dreamer-V3 agent after 1k steps on LunarLander-v3 environment"></td>
            <td style="text-align: center"><img src="./examples/images/LunarLander_10k.gif" width="100%" alt="Trained Dreamer-V3 agent after 10k steps on LunarLander-v3 environment"></td>
            <td style="text-align: center"><img src="./examples/images/LunarLander_25k.gif" width="100%" alt="Trained Dreamer-V3 agent after 25k steps on LunarLander-v3 environment"></td>
            <td style="text-align: center"><img src="./examples/images/LunarLander_50k.gif" width="100%" alt="Trained Dreamer-V3 agent after 50k steps on LunarLander-v3 environment"></td>
        </tr>
        <!-- CartPole -->
        <tr bgcolor="#2b2b68">
            <td colspan=4 style="text-align: center"><b>CartPole-v1</b></td>
        </tr>
        <tr>
            <td style="text-align: center">1K Steps</td>
            <td style="text-align: center">8K Steps</td>
            <td style="text-align: center">15K Steps</td>
            <td style="text-align: center">26K Steps</td>
        </tr>
        <tr>
            <td style="text-align: center"><img src="./examples/images/CartPole_1k.gif" width="100%" alt="Trained Dreamer-V3 agent after 1k steps on CartPole-v1 environment"></td>
            <td style="text-align: center"><img src="./examples/images/CartPole_8k.gif" width="100%" alt="Trained Dreamer-V3 agent after 8k steps on CartPole-v1 environment"></td>
            <td style="text-align: center"><img src="./examples/images/CartPole_15k.gif" width="100%" alt="Trained Dreamer-V3 agent after 15k steps on CartPole-v1 environment"></td>
            <td style="text-align: center"><img src="./examples/images/CartPole_26k.gif" width="100%" alt="Trained Dreamer-V3 agent after 26k steps on CartPole-v1 environment"></td>
        </tr>
    </tbody>
</table>

<!-- | CartPole | LunarLander |
| :---: | :---: |
| <img src="./examples/CartPole.gif" width="200" alt="Trained Dreamer-V3 agent on CartPole-v1 environment"> | <img src="./examples/CartPole.gif" width="200" alt="Trained Dreamer-V3 agent on LunarLander-v3 environment"> | -->

> [!WARNING]
> This repository is still under construction. Please check back later for more information, features, and examples.
