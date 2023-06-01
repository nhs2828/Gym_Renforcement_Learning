cfgDDGP = {
    "len_buffer" : 100000,
    "batch" : 64,
    "gamma" : 0.99,
    "tau" : 0.001,
    "explore": 1.0,
    "explore_min": 0.01,
    "explore_decay": 0.995,
    "noise": 0.1,
    "noise_decay_step": 5000,
    "start_learning_step": 10000,
    "lr" : {
        "critic" : 0.0002,
        "actor" : 0.0003
    },
    "hidden_layer" : {
        "critic" : [256,256],
        "actor" : [256, 256]
    }
}