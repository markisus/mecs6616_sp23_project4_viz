from arm_dynamics_base import ArmDynamicsBase

class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, time_step, device):
        # dummy impl
        pass

    def dynamics_step(self, state, action, dt):
        # dummy impl
        return state
