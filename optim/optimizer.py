class Optimizer:
    def state_dict(self):
        pass

    def step(self, *args, **kwargs):
        raise NotImplementedError("step method is not implemented")
