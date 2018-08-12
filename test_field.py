from QinCapsNet import QinCapsNet

qin_caps_net = QinCapsNet(caps1_n_maps=25, caps1_n_dims=6, caps2_n_caps=10, caps2_n_dims=18, n_epochs=10, batch_size=50,
                          restore_checkpoint=True)

qin_caps_net.create_a_net()
