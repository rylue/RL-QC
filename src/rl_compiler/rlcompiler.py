import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from noise_mapping_experiment.noise_mapping_exp import BackendErrors

@dataclass
class CompilerState:
    """
    Stores the current environment state for one compilation episode.
    """
    logical_to_physical: List[int]
    placed_logical: List[bool]
    used_physical: List[bool]
    current_logical_idx: int
    done: bool

class RLCompiler(gym.Env):
    """
    Reinforcement-learning environment for noise-aware quantum circuit mapping.

    The agent learns to assign logical qubits from an input circuit onto physical qubits
    of a backend topology, taking into consideration:

    - hardware connectivity
    - readout error
    - 2-qubit gate error
    - logical interaction structure of the circuit
    - possible execution-quality proxies such as Murali-style reliability scores
    - optional empirical canary success probability from transpiled execution

    The agent incrementally builds a layout. At each step, it chooses one physical qubit
    for the next logical qubit to be placed. Once it is done mapping all logical qubits,
    the layout is evaluated based on the reward.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend_info: BackendErrors,
        max_logical_qubits: Optional[int] = None,   # constraint for input size
        reward_mode: str = "murali_proxy",  # reward based on Murali et al. evaluations
        invalid_action_penalty: float = -5.0,   # default large deduction for invalid mapping
        completion_bonus: float = 1.0,     # signal for complete and feasible mapping
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.circuit = circuit
        self.backend_info = backend_info
        self.reward_mode = reward_mode
        self.invalid_action_penalty = invalid_action_penalty
        self.completion_bonus = completion_bonus
        self.render_mode = render_mode

        # number of logical qubits
        self.n_logical = self._infer_num_logical_qubits(circuit)
        # number of physical qubits
        self.n_physical = self._infer_num_physical_qubits(backend_info)

        if max_logical_qubits is not None:
            self.n_logical = min(self.n_logical, max_logical_qubits)

        # ACTION SPACE
        # defines the set of choices for the agent, in this case, the index of the physical
        # qubit for the next logical qubit to place.
        self.action_space = spaces.Discrete(self.n_physical)

        # OBSERVATION SPACE
        # defines the set of information fed to the agent. in this case, it may include:
        # - current partially complete layout
        # - which logical qubits have already been placed
        # - which physical qubits are already occupied
        # - readout error per physical qubit
        # - pairwise two-gate error statistics
        # - logical interaction info for the next qubit
        # - connectivity features of device
        # - path reliability summaries
        # the below dimensions are more simple
        obs_dim = (
            self.n_logical +    # current logical_to_physical mapping
            self.n_logical +    # placed_logical mask
            self.n_physical +   # used_physical mask
            self.n_physical     # readout error features
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1e6,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.state: Optional[CompilerState] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the compilation episode and intialize an empty partial layout. In this initialization,
        no logical qubits have been assigned yet, and the placement pointer is reset to logical
        qubit 0.

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        self.state = CompilerState(
            logical_to_physical=[-1] * self.n_logical,
            placed_logical=[False] * self.n_logical,
            used_physical=[False] * self.n_physical,
            current_logical_idx=0,
            done=False,
        )

        obs = self._get_observation()
        info = self._build_info()
        return obs, info

    def step(self, action: int):
        """
        One logical to physical placement action is taken. The action is an assignment of the current
        logical qubit to one physical qubit. The logic is as follows:
        1. Check whether the action (assignment) is valid. To be valid, the physical qubit must be
        unoccupied.
        2. If the action is valid, update the partial layout.
        3. If all logical qubits have now been assigned, evaluate the final mapping and terminate.
        4. Otherwise, continue to next step (next logical qubit).

        Returns:
            observation, reward, terminated (bool), truncated (bool), info
        """
        reward = 0.0
        terminated = False
        truncated = False

        if not self._is_valid_action(action):
            reward += self.invalid_action_penalty
            obs = self._get_observation()
            info = self._build_info(extra={"invalid_action": True})
            return obs, reward, terminated, truncated, info

        logical_idx = self.state.current_logical_idx
        self.state.logical_to_physical[logical_idx] = action
        self.state.placed_logical[logical_idx] = True
        self.state.used_physical[action] = True
        self.state.current_logical_idx += 1

        if self.state.current_logical_idx >= self.n_logical:    # done with all logical qubit mappings
            self.state.done = True
            terminated = True

            final_layout = list(self.state.logical_to_physical)
            reward += self.completion_bonus
            reward += self._evaluate_final_layout(final_layout)

        obs = self._get_observation()
        info = self._build_info(extra={"final_layout": self.state.logical_to_physical})
        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Optional visualization tool. Used to show evolution of mappings as the agent learns.
        Consider displaying the current partial mapping or a logical-to-physical assignment table.
        """
        pass

    def close(self):
        """
        Clean up and close resources if needed.
        """
        pass

    # HELPERS
    def _infer_num_logical_qubits(self, circuit: QuantumCircuit) -> int:
        return circuit.num_qubits

    def _infer_num_physical_qubits(self, backend_info: BackendErrors) -> int:
        return len(backend_info.readout_error)

    def _get_observation(self) -> np.ndarray:
        """
        Convert the current partial mapping state into a flat numeric observation. The
        current implementation is quite simple.

        Suggested components:
        - logical_to_physical with -1 for unassigned
        - placed_logical mask
        - used_physical mask
        - readout-error vector
        - optional graph-derived features
        - optional logical-interaction features for remaining qubits
        """
        assert self.state is not None

        logical_to_physical = np.array(self.state.logical_to_physical, dtype=np.float32)
        placed_logical = np.array(self.state.placed_logical, dtype=np.float32)
        used_physical = np.array(self.state.used_physical, dtype=np.float32)
        readout_error = np.array(self.backend_info.readout_error, dtype=np.float32)

        obs = np.concatenate([
            logical_to_physical,
            placed_logical,
            used_physical,
            readout_error,
        ])
        return obs

    def _is_valid_action(self, action: int) -> bool:
        """
        Check whether the chosen physical qubit can be assigned at this step. The action
        corresponds to the index of a physical qubit. This index must be in range, and the
        physical qubit must not be occupied already.
        """
        assert self.state is not None

        if action < 0 or action >= self.n_physical:
            return False
        if self.state.used_physical[action]:
            return False
        return True

    def _evaluate_final_layout(self, layout: List[int]) -> float:
        """
        Reward based on the completed logical-to-physical mapping. This reward may be inspired
        by techniques used in the noise mapping experiment. For example, some ideas:
        - Murali-inspired score that combines readout reliability of mapped qubits with the
        best path reliability for logical CX interactions.
        - Success probability based on success rate for circuit on a simulator.

        Note that reward needs to be normalized in [-1, 1] for training stability.
        """
        if self.reward_mode == "murali_proxy":
            return self._murali_proxy_reward(layout)
        elif self.reward_mode == "canary_success":
            return self._canary_success_reward(layout)
        else:
            raise ValueError(f"{self.reward_mode} is not implemented.")

    def _murali_proxy_reward(self, layout: List[int]) -> float:
        """
        Evaluates similarly to Murali-style evaluation in noise mapping experiment.
        """
        return 0.0

    def _canary_success_reward(self, layout: List[int]) -> float:
        """
        Evaluates similarly to success rate in noise mapping experiment, defined as the fraction
        of outputs that correspond to expected values.
        """
        return 0.0

    def _build_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Useful info for debugging or stats during training.
        """
        assert self.state is not None

        info = {
            "current_logical_idx": self.state.current_logical_idx,
            "logical_to_physical": list(self.state.logical_to_physical),
            "placed_logical": list(self.state.placed_logical),
            "used_physical": list(self.state.used_physical),
        }

        if extra is not None:
            info.update(extra)

        return info