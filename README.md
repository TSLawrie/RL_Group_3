<div id="top"></div>


<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



## About The Project

This project aims to solve [OpenAI](https://openai.com)'s [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2) problem using 2 Reinforcement Learning algorithms: DQN and DDPG.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

These are the languages and major libraries used in the project.

* [Python](https://python.org)
* [Pytorch](https://pytorch.org)
* [Gym](https://gym.openai.com)
* [Numpy](https://numpy.org)
* [Pandas](https://pandas.pydata.org)
* [Tensorboard](https://tensorflow.org/tensorboard)


<p align="right">(<a href="#top">back to top</a>)</p>



## Getting Started

Below is all the information required to train agents with DQN or DDPG.

### Prerequisites

The prerequisites are the same on all platforms however they will be listed as their installation commands on Windows. It is recommended to install these in the order given.
* Pytorch
  ```sh
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
  ```
* Wheel
  ```sh
  pip install wheel
  ```
* Gym requirements
  ```sh
  pip install gym[all]
  ```
* Box2D
  ```sh
  pip install box2d
  ```
* Box2D-py
  ```sh
  pip install box2d-py
  ```
* Gym
  ```sh
  pip install gym
  ```
* Pandas
  ```sh
  pip install pandas
  ```
* Numpy
  ```sh
  pip install numpy
  ```
* TensorBoard
  ```sh
  pip install tensorboard
  ```

### Installation

To get the code on your system simply clone the GitHub repository using the below command or download the zip file on the GitHub website.

   ```sh
   git clone https://github.bath.ac.uk/kd671/RL_Group_3.git
   ```
<p align="right">(<a href="#top">back to top</a>)</p>



## Usage

To run the code that trains an agent, navigate to the "/RL-Group-3" folder and run one of:

DQN:

```sh
(base) PS ~\Documents\GitHub\RL_Group_3> python /ContinuousV1/DQN_v3.py
```

DDPG:

```sh
(base) PS ~\Documents\GitHub\RL_Group_3> python /Discretev1/DDPG.py
```

To change the hyperparameters, open the relevant python file you want to run and change the hyperparameters in the following sections:

DQN:

![DQN Params Change Location](/dqn_param_change.png)

DDPG:

![DDPG Params Change Location](/ddpg_param_change.png)

Once the code finishes running, a log of the training run will be saved to the folder "/RL-Group-3/logs/*algo*" where *algo* is "dqn" or "ddpg" depending on which algorithm you wish to use. You can view the logs by running the following in the same directory as previously, replacing *folder_name* with the name of the folder your log is in (by default the folder name is the timestamp of when the code was first ran):
```sh
(base) PS C:\Users\totol\Documents\GitHub\RL_Group_3> tensorboard --logdir=./logs/folder_name
```

You can also copy multiple log folders into one folder within "/logs" and run tensorboard from that folder to compare multiple runs.

2 or 4 pytorch models will also be saved: a *best* model and *last* model for each neural network. DDPG uses 2 neural networks hence double the models compared to DQN's single neural network.

You can test your trained agent using the models saved. First open the file "agent_test.py" in a text editor and modifiy the paths for the models you wish to test, and whether you wish to use the *best* or *last* model(s). Once done, save and run the following in the same directory as previously.
```sh
(base) PS C:\Users\totol\Documents\GitHub\RL_Group_3> python agent_test.py
```

The DQN agent can be evaluated by calling the `eval` function.

<p align="right">(<a href="#top">back to top</a>)</p>
