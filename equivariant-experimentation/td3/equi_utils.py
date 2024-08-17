from emlp.reps import Scalar, Vector, Rep
from emlp.nn.flax import uniform_rep, EMLPBlock, Linear, Sequential
import jax
import jax.numpy as jnp
import numpy as np

class AngularVelocityTorqueRep(Rep):
    """Representation for reflecting angular velocities and torques across the x-axis."""
    
    def __init__(self, G):
        self.G = G  # The group to which this representation is associated
        super().__init__()

    def rho(self, M):
        """
        Group representation of the matrix M.
        M should be either the identity or reflection matrix.
        """
        if jnp.allclose(M, jnp.eye(2)):
            return jnp.eye(2)  # Identity matrix, no change
        elif jnp.allclose(M, jnp.array([[-1, 0], [0, 1]])):
            return -jnp.eye(2)   # Sign flip for angular velocities and torques
        else:
            raise ValueError("Unrecognized group element")

    def size(self):
        assert self.G is not None, f"must know G to find size for rep={self}"
        return self.G.d

    def __str__(self):
        return "AngularVelocityTorqueRep"
    def __call__(self,G):
        return self.__class__(G)
    
class ReflectRep(Rep):
    """Representation for reflecting angular velocities and torques across the x-axis."""
    
    def __init__(self, G):
        self.G = G  # The group to which this representation is associated
        super().__init__()

    def rho(self, M):
        """
        Group representation of the matrix M.
        M should be either the identity or reflection matrix.
        """
        if jnp.allclose(M, jnp.eye(2)):
            return jnp.eye(1)  # Identity matrix, no change
        elif jnp.allclose(M, jnp.array([[-1, 0], [0, -1]])):
            return -1*jnp.eye(1)   # Sign flip for angular velocities and torques
        else:
            raise ValueError("Unrecognized group element")

    def size(self):
        assert self.G is not None, f"must know G to find size for rep={self}"
        return 1

    def __str__(self):
        return "AngularVelocityTorqueRep"
    def __call__(self,G):
        return self.__class__(G)



def rel_err(a, b):
    return np.array(
        jnp.sqrt(((a - b) ** 2).mean())
        / (jnp.sqrt((a**2).mean()) + jnp.sqrt((b**2).mean()))
    )


def equivariance_err_actor(model, params, state, rin, rout, G):
    gs = G.samples(state.shape[0])
    rho_gin = jnp.stack([jnp.array(rin.rho_dense(g)) for g in gs])
    rho_gout = jnp.stack([jnp.array(rout.rho_dense(g)) for g in gs])

    y1 = model.apply(params, (rho_gin @ state[..., None]).squeeze(-1))
    y2 = model.apply(params, state)
    y2 = (rho_gout @ y2[..., None]).squeeze(-1)
    return rel_err(y1, y2)


def equivariance_err_qvalue(model, params, state, actions, rin, rout, G):
    gs = G.samples(state.shape[0])
    rho_gin = jnp.stack([jnp.array(rin.rho_dense(g)) for g in gs])
    rho_gout = jnp.stack([jnp.array(rout.rho_dense(g)) for g in gs])

    x = jnp.concatenate([state, actions], axis=1)
    x = x @ rho_gin

    y1 = model.apply(
        params, x[0, :, : state.shape[-1]], x[0, :, -actions.shape[-1] :]
    ).squeeze(-1)
    y2 = model.apply(params, state, actions)
    y2 = (rho_gout @ y2[..., None]).squeeze(-1)
    return rel_err(y1, y2)