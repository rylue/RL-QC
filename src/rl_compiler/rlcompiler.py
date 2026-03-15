import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

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
        circuit: Any,
        backend_info: Any,
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

        The reward may be based on the probability that the assigned mapping outputs an expected
        value, after running many simulations with Qiskit. Alternatively, other reward designs
        may be considered, such as a backend-aware noise penalty, circuit depth penalty, local
        placement quality, etc.

        Note that reward needs to be normalized in [-1, 1].

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

    # ----------------------------------------------------------------------
    # Helper methods to implement
    # ----------------------------------------------------------------------

    def _infer_num_logical_qubits(self, circuit: Any) -> int:
        """
        Extract the number of logical qubits from the input circuit.

        In your actual implementation, this should read from the QuantumCircuit
        object and possibly support filtering to a smaller benchmark instance.
        """
        return circuit.num_qubits

    def _infer_num_physical_qubits(self, backend_info: Any) -> int:
        """
        Extract the number of physical qubits from the backend / backend-error
        structure.
        """
        return len(backend_info.readout_error)

    def _get_observation(self) -> np.ndarray:
        """
        Convert the current partial mapping state into a flat numeric observation.

        Suggested components:
        - logical_to_physical with -1 for unassigned
        - placed_logical mask
        - used_physical mask
        - readout-error vector
        - optional graph-derived features
        - optional logical-interaction features for remaining qubits

        This scaffold only includes a simple base observation.
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
        Check whether the chosen physical qubit can be assigned at this step.

        Base validity:
        - action index must be in range
        - physical qubit must not already be occupied

        Possible future extensions:
        - restrict to candidate sets based on interaction graph
        - topology-aware feasibility filtering
        - symmetry-reduction logic
        """
        assert self.state is not None

        if action < 0 or action >= self.n_physical:
            return False
        if self.state.used_physical[action]:
            return False
        return True

    def _evaluate_final_layout(self, layout: List[int]) -> float:
        """
        Evaluate the completed logical-to-physical mapping.

        This is where your real experiment logic plugs in.

        Possible reward modes:
        ------------------------------------------------------------------
        1. 'murali_proxy'
           Use a Murali-inspired score combining:
           - readout reliability of mapped qubits
           - best path reliability for logical CX interactions

        2. 'transpile_cost'
           Transpile the circuit with the chosen layout and score based on:
           - depth
           - 2Q gate count
           - SWAP count
           - total operation count

        3. 'canary_success'
           Execute a canary / benchmark circuit on a simulator or fake backend
           and reward by empirical success probability.

        4. 'hybrid'
           Combine static proxy and empirical execution metrics.
        ------------------------------------------------------------------

        This method should call the experiment utilities you already wrote,
        rather than reimplementing them here.
        """
        if self.reward_mode == "murali_proxy":
            return self._murali_proxy_reward(layout)
        elif self.reward_mode == "transpile_cost":
            return self._transpile_cost_reward(layout)
        elif self.reward_mode == "canary_success":
            return self._canary_success_reward(layout)
        elif self.reward_mode == "hybrid":
            return self._hybrid_reward(layout)
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def _murali_proxy_reward(self, layout: List[int]) -> float:
        """
        Placeholder for Murali-inspired static layout scoring.

        Intended contents:
        - compute readout reliability term
        - compute logical-CX path reliability term
        - combine with weighting factor omega
        - return higher score for cleaner mappings

        This should wrap your existing Murali-style scoring code.
        """
        return 0.0

    def _transpile_cost_reward(self, layout: List[int]) -> float:
        """
        Placeholder for reward from transpiled circuit quality.

        Intended contents:
        - transpile with initial_layout=layout
        - inspect depth and gate counts
        - penalize SWAP-heavy or deep compilations
        - optionally normalize across benchmarks
        """
        return 0.0

    def _canary_success_reward(self, layout: List[int]) -> float:
        """
        Placeholder for empirical reward from execution.

        Intended contents:
        - compile the circuit or a canary under this layout
        - run on simulator / fake backend
        - compute success probability from measurement counts
        - reward mappings with better observed correctness
        """
        return 0.0

    def _hybrid_reward(self, layout: List[int]) -> float:
        """
        Placeholder for a weighted combination of static and empirical rewards.

        Example:
            alpha * murali_proxy
            + beta * canary_success
            - gamma * transpile_depth_penalty
        """
        return 0.0

    def _build_info(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build debugging / logging info returned by reset() and step().

        Useful for:
        - inspecting current partial mapping
        - monitoring invalid actions
        - tracking final layout and reward components
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