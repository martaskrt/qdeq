    def measure_big(self, bsz):
        states = self.q_device.get_states_1d()
        states = states.reshape((bsz, 1, -1))
        result = torch.zeros((bsz, self.num_classes,), dtype=torch.float)
        for k in range(self.num_classes):
            obs = torch.zeros((states.shape[2],), dtype=torch.cfloat)
            obs[k] = 1.
            obs = torch.einsum('i, j -> ij', obs, obs)

            result[:,k] = torch.bmm(states.conj(), torch.mm(obs, states.reshape((bsz,-1)).transpose(0, 1)).transpose(0, 1).reshape((bsz,-1,1))).squeeze()
