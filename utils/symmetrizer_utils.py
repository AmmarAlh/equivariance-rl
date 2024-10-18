import torch
import numpy as np

from symmetrizer.ops import GroupRepresentations
from symmetrizer.groups import MatrixRepresentation


def create_inverted_pendulum_actor_representations():
    """
    Creates the input and output representations for the actor in the InvertedPendulum environment.

    Returns:
        repr_in (MatrixRepresentation): Input representation for the actor.
        repr_out (MatrixRepresentation): Output representation for the actor.
    """
    # Actor input representation specific to InvertedPendulum
    actor_input_representations = [
        torch.FloatTensor(np.eye(4)), 
        torch.FloatTensor(-1 * np.eye(4))
    ]
    in_group_actor = GroupRepresentations(actor_input_representations, "StateGroupRepr")
    
    # Actor output representation
    actor_output_representations = [
        torch.FloatTensor(np.eye(1)), 
        torch.FloatTensor(-1 * np.eye(1))
    ]
    out_group_actor = GroupRepresentations(actor_output_representations, "ActionGroupRepr")
    
    repr_in = MatrixRepresentation(in_group_actor, out_group_actor)
    repr_out = MatrixRepresentation(out_group_actor, out_group_actor)
    
    return repr_in, repr_out

def create_inverted_pendulum_qfunction_representations():
    """
    Creates the input and output representations for the Q-function in the InvertedPendulum environment.

    Returns:
        repr_in_q (MatrixRepresentation): Input representation for the Q-function.
        repr_out_q (MatrixRepresentation): Output representation for the Q-function.
    """
    # Q-function input representation specific to InvertedPendulum
    qf_input_representations = [
        torch.FloatTensor(np.eye(5)), 
        torch.FloatTensor(-1 * np.eye(5))
    ]
    in_group_qf = GroupRepresentations(qf_input_representations, "StateGroupRepr")
    
    # Q-function output representation
    qf_output_representations =  [torch.FloatTensor(np.eye(5)), torch.FloatTensor( [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])]
    out_group_qf = GroupRepresentations(qf_output_representations, "ActionGroupRepr")
    
    repr_in_q = MatrixRepresentation(in_group_qf, out_group_qf)
    
    # Q-function invariant output representation
    invariant_output_representations = [
        torch.FloatTensor(np.eye(1)), 
        torch.FloatTensor(np.eye(1))
    ]
    out_group_qf_invariant = GroupRepresentations(invariant_output_representations, "InvariantGroupRepr")
    
    repr_out_q = MatrixRepresentation(out_group_qf, out_group_qf_invariant)
    
    return repr_in_q, repr_out_q
def create_cartpole_actor_representations():
    """
    Creates the input and output representations for the actor in the CartPole environment.

    Returns:
        repr_in (MatrixRepresentation): Input representation for the actor.
        repr_out (MatrixRepresentation): Output representation for the actor.
    """
    # Actor input representation specific to CartPole
    actor_input_representations = [
        torch.FloatTensor(np.eye(4)), 
        torch.FloatTensor(-1 * np.eye(4))
    ]
    in_group_actor = GroupRepresentations(actor_input_representations, "StateGroupRepr")
    
    # Actor output representation
    actor_output_representations = [
        torch.FloatTensor(np.eye(2)), 
        torch.FloatTensor([[0,1],[1,0]])
    ]
    out_group_actor = GroupRepresentations(actor_output_representations, "ActionGroupRepr")
    
    repr_in = MatrixRepresentation(in_group_actor, out_group_actor)
    repr_out = MatrixRepresentation(out_group_actor, out_group_actor)
    
    return repr_in, repr_out
def create_cartpole_vfunction_representations():
    """
    Creates the input and output representations for the V-function in the CartPole environment.

    Returns:
        repr_in_v (MatrixRepresentation): Input representation for the V-function.
        repr_out_v (MatrixRepresentation): Output representation for the V-function.
    """
    # V-function input representation specific to CartPole
    vf_input_representations = [
        torch.FloatTensor(np.eye(4)), 
        torch.FloatTensor(-1 * np.eye(4))
    ]
    in_group_vf = GroupRepresentations(vf_input_representations, "StateGroupRepr")
    
    # V-function output representation
    vf_output_representations = [
        torch.FloatTensor(np.eye(2)), 
        torch.FloatTensor([[0,1],[1,0]])
    ]
    out_group_vf = GroupRepresentations(vf_output_representations, "ActionGroupRepr")
    
    # V-function invariant output representation
    invariant_output_representations = [
        torch.FloatTensor(np.eye(1)), 
        torch.FloatTensor(np.eye(1))
    ]
    out_group_vf_invariant = GroupRepresentations(invariant_output_representations, "InvariantGroupRepr")
    repr_in_v = MatrixRepresentation(in_group_vf, out_group_vf)
    repr_out_v = MatrixRepresentation(out_group_vf, out_group_vf_invariant)
    
    return repr_in_v, repr_out_v

def actor_equivariance_mae(network, obs: torch.Tensor, repr_in: MatrixRepresentation, repr_out: MatrixRepresentation) -> float:
    """
    Calculate the MSE of the equivariance error for the actor network.
    """
    # Move the matrices to the network's device
    device = obs.device
    dtype = obs.dtype
    repr_in_matrices = [p_in.to(device, dtype) for p_in in repr_in._input_matrices]
    repr_out_matrices = [p_out.to(device, dtype) for p_out in repr_out._output_matrices]

    transformed_inputs = torch.stack([obs @ p_in for p_in in repr_in_matrices])
    
    def get_only_mean(x):
        if isinstance(x, tuple):
            return x[0]
        return x
    
    
    y1 = torch.stack([get_only_mean(network(p_x)) for p_x in transformed_inputs])
    y2 = torch.stack([get_only_mean(network(obs)) @ p_out  for p_out in repr_out_matrices])

    return (y1.abs() - y2.abs()).abs().mean().item()

def q_equivariance_mae(network, obs: torch.Tensor, actions: torch.Tensor, repr_in_q: MatrixRepresentation) -> float:
    """
    Calculate the MSE of the equivariance error for the Q-network.
    """
    device = obs.device
    dtype = obs.dtype
    obs_actions = torch.cat([obs, actions], dim=-1)
    # Move the matrices to the network's device
    repr_in_q_matrices = [p_in.to(device, dtype) for p_in in repr_in_q._input_matrices]
    
    transformed_inputs = torch.stack([obs_actions @ p_in for p_in in repr_in_q_matrices])
    y1 = torch.stack([network(p_obs_actions[:, :obs.size(-1)], p_obs_actions[:, obs.size(-1):]) for p_obs_actions in transformed_inputs])
    y2 = network(obs, actions).unsqueeze(0).expand_as(y1)

    return (y1.abs() - y2.abs()).abs().mean().item()
